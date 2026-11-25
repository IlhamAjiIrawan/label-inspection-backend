"""
main.py — External Service - Label Inspection Status + OCR Pipeline

PIPELINE RINGKAS (10 TAHAP — Fix 2026-09-06 CLAHE+2pass):
 1. Validasi  (ukuran ≤5MB, ekstensi + magic bytes, sanitasi nama)
 2. Simpan    (tulis ke uploads/{generated_filename}, update metadata.json)
 3. Deteksi   (gunakan generated filename langsung — paling akurat)
 4. Preprocessing (grayscale → CLAHE 2.0 → autocontrast → upscale 2x bicubic → sharpen → medianBlur 3, fallback adaptiveThreshold 11,2) CPU-only, 2-pass
 5. OCR Engine (Tesseract local, lang=eng, --psm 6, fallback 11,3 jika blank, raw text)
 6. Parsing & Cleaning (strip, hapus baris kosong, pecah per baris) + is_empty guard
 7. Simpan hasil OCR (uploads/ocr/{filename}.json + in-memory LRU max 10) — status failed jika empty after 2-pass
 8. Endpoint GET (/api/test/ocr/latest, /api/test/ocr/{filename}) + is_empty/warning
 9. Callback PUSH (kirim cleaned_text+is_empty ke Laravel http://127.0.0.1:8000/api/ocr/callback via httpx + HMAC + retry, 401 no-retry with hint, empty-failed allowed)
10. Reconcile (Laravel bisa GET /api/test/ocr/latest sebagai fallback + POST /callback/retry manual)

KEAMANAN:
 - MAX_UPLOAD_SIZE = 5 MB, ALLOWED_EXTENSIONS = png/jpg/jpeg, magic bytes check
 - Filename di-generate internal (cegah path traversal), tidak percaya input client
 - Rate limit 1 gambar/menit → 429, tidak ada traceback bocor, no eval/exec input user
 - Callback: HMAC-SHA256 (X-Signature), X-API-Key, X-Timestamp TTL 5m, allowlist host cegah SSRF, timeout 5s
PERFORMA:
 - OCR via BackgroundTasks (tidak blokir response POST)
 - Callback via BackgroundTasks sync (httpx.Client) — tidak blokir POST, retry 3x exponential 2/4/8s + jitter
 - Preprocessing resize dibatasi max 6000px sisi terpanjang (cegah OOM)
 - Hasil di-cache di memory (max 10) + persist disk JSON
"""

from fastapi import FastAPI, HTTPException, File, UploadFile, Form, BackgroundTasks, Header, Request, Query
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from datetime import datetime, timezone, timedelta
from typing import Optional
import os
import json
import tempfile
import time
import logging
import mimetypes
from pathlib import Path

# --- Internal modules (modular structure) ---
from app.config import (
    UPLOAD_DIR,
    OCR_DIR,
    METADATA_FILE,
    MAX_UPLOAD_SIZE,
    TESSERACT_LANG,
    TESSERACT_CONFIG,
    OCR_ENGINE,
    CALLBACK_ENABLED,
    CALLBACK_LOG_FILE,
    CALLBACK_DEAD_LETTER_FILE,
    LARAVEL_CALLBACK_URL,
    LARAVEL_API_KEY,
    GET_RATE_LIMIT_REQUESTS,
    GET_RATE_LIMIT_WINDOW_SECONDS,
    DELETE_RATE_LIMIT_REQUESTS,
    DELETE_RATE_LIMIT_WINDOW_SECONDS,
)
from app.security import (
    validate_image_file,
    generate_safe_filename,
    is_safe_path,
    check_rate_limit,
    check_get_rate_limit,
    check_delete_rate_limit,
    get_rate_limit_headers,
    verify_delete_api_key,
    sanitize_filename_for_lookup,
)
from app.storage import load_metadata, save_metadata, get_latest_image, delete_by_filename
from app.ocr.preprocess import preprocess_image, preprocess_v1, preprocess_v2_adaptive
from app.ocr.parser import clean_ocr_text
from app.ocr.store import ocr_store
from app.ocr.engine import is_tesseract_available, extract_text_tesseract, get_tesseract_cmd

# ---------------------------------------------------------------------------
# App & Timezone + Lifespan warm-up (R1)
# ---------------------------------------------------------------------------
from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    # R1: warm-up engine di tiap worker OCR agar request pertama ≈ steady-state.
    try:
        from app import config as _cfg

        if bool(getattr(_cfg, "OCR_WARMUP", True)):
            try:
                from app.ocr.executor import OCR_EXECUTOR  # pastikan pool sudah ada
            except Exception:
                pass
            try:
                from app.ocr.paddle_engine import warmup_engines

                import asyncio as _aio

                await _aio.to_thread(warmup_engines)
            except Exception as e:
                print(f"[WARMUP] skipped: {e}")
    except Exception as e:
        print(f"[WARMUP] lifespan error: {e}")
    yield


app = FastAPI(title="External Service - Label Inspection Status + OCR", lifespan=lifespan)

JAKARTA_TZ = timezone(timedelta(hours=7))

# Pastikan folder ada (juga sudah di app.config, tapi jaga-jaga jika import order berubah)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OCR_DIR.mkdir(parents=True, exist_ok=True)

# --- Reference Library DB init (PLAN_REFERENCE_LIBRARY_FASTAPI + MYSQL_BACKED) ---
# Backend ikut flag REF_STORE: sqlite → SQLite lokal, mysql → tabel ref_* (init idempotent).
try:
    from app.config import REF_STORE as _REF_STORE_MAIN

    if str(_REF_STORE_MAIN or "sqlite").strip().lower() == "mysql":
        from app.db_mysql import init_db as _init_ref_db
    else:
        from app.references.db import init_db as _init_ref_db

    _init_ref_db()
    print(f"[REFERENCES] db ready (store={str(_REF_STORE_MAIN).strip().lower()})")
except Exception as e:
    print(f"[REFERENCES] db init failed: {e}")

logger = logging.getLogger(__name__)

# --- Check Batch Router (5 images) — PLAN_CHECK_5_IMAGES ---
try:
    from routers.check import router as check_router

    app.include_router(check_router)
    print("[CHECK] router mounted at /api/check/captures")
except ImportError as e:
    print(f"[CHECK] router not loaded: {e} — run without check endpoint")
except Exception as e:
    print(f"[CHECK] failed to mount router: {e}")

# --- Reference Library Router — PLAN_REFERENCE_LIBRARY_FASTAPI ---
try:
    from routers.references import router as references_router

    app.include_router(references_router)
    print("[REFERENCES] router mounted at /api/references")
except ImportError as e:
    print(f"[REFERENCES] router not loaded: {e} — run without references endpoint")
except Exception as e:
    print(f"[REFERENCES] failed to mount router: {e}")


# ---------------------------------------------------------------------------
# Helpers: Rate Limit 429 Ideal (§11) — unified response dengan Retry-After
# ---------------------------------------------------------------------------
def _rate_limit_429_response(wait_seconds: float, remaining: int = 0, limit: int = 60, message: Optional[str] = None) -> JSONResponse:
    """
    Bangun response 429 Too Many Requests ideal (RFC 6585).
    - Header Retry-After (detik) agar client (Laravel/Frontend) bisa backoff
    - X-RateLimit-Limit / Remaining / Reset untuk observability
    - Body JSON dengan success False + retry_after_seconds
    """
    wait_int = int(wait_seconds) + 1  # bulatkan ke atas
    headers = {
        "Retry-After": str(wait_int),
        "X-RateLimit-Limit": str(limit),
        "X-RateLimit-Remaining": str(max(0, remaining)),
        "X-RateLimit-Reset": str(wait_int),
        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
        "Pragma": "no-cache",
        "Expires": "0",
    }
    return JSONResponse(
        status_code=429,
        content={
            "success": False,
            "message": message or f"Rate limit exceeded. Please wait {wait_int} seconds.",
            "retry_after_seconds": wait_int,
            "error": "too_many_requests",
        },
        headers=headers,
    )


def _enforce_get_rate_limit():
    """
    Cek GET rate limit untuk polling. Jika melebihi, return JSONResponse 429, else None.
    Dipanggil di awal setiap GET handler pooling (ocr/latest, ocr/{filename}, status, files, image).
    Ideal: 60 req / 60s (config), cukup untuk 10× polling/30s =20/min, bahkan 2 paralel=40/min.
    Jika frontend/laravel masih spam 6 req per poll → 60/30s akan 429 → sinyal untuk optimasi poll (kurangi ke 1 req/poll).
    """
    allowed, wait, remaining = check_get_rate_limit()
    if not allowed:
        return _rate_limit_429_response(wait, remaining, GET_RATE_LIMIT_REQUESTS, f"GET rate limit exceeded. Max {GET_RATE_LIMIT_REQUESTS} req per {GET_RATE_LIMIT_WINDOW_SECONDS}s. Retry after {int(wait)+1}s.")
    return None


def _enforce_delete_rate_limit():
    """Rate limit untuk DELETE best-effort (30/min)."""
    allowed, wait, remaining = check_delete_rate_limit()
    if not allowed:
        return _rate_limit_429_response(wait, remaining, DELETE_RATE_LIMIT_REQUESTS, f"DELETE rate limit exceeded. Max {DELETE_RATE_LIMIT_REQUESTS} req per {DELETE_RATE_LIMIT_WINDOW_SECONDS}s. Retry after {int(wait)+1}s.")
    return None


def _extract_api_key_from_headers(request: Request, x_api_key_header: Optional[str] = None) -> Optional[str]:
    """
    Ambil API key dari berbagai sumber header (case-insensitive) untuk DELETE.
    Support: X-API-Key, x-api-key, X-Api-Key, atau query ?api_key=
    FastAPI Header alias sudah tangani, tapi juga cek request.headers langsung untuk toleransi.
    """
    if x_api_key_header:
        return x_api_key_header
    # Fallback cek langsung dari request.headers (case-insensitive dict)
    for k in ("X-API-Key", "x-api-key", "X-Api-Key", "X-API-KEY"):
        v = request.headers.get(k)
        if v:
            return v
    # Juga support query param ?api_key= (untuk testing curl mudah)
    qp = request.query_params.get("api_key")
    if qp:
        return qp
    return None

# ---------------------------------------------------------------------------
# In-memory latest status (compatible dengan versi sebelumnya)
# ---------------------------------------------------------------------------
latest_source: Optional[str] = None
latest_timestamp: Optional[str] = None
latest_template_id: Optional[str] = None
latest_template_name: Optional[str] = None
latest_template_width: Optional[str] = None
latest_template_height: Optional[str] = None
latest_filename: Optional[str] = None  # ini sekarang = generated filename (safe)
latest_resolution_w: Optional[str] = None
latest_resolution_h: Optional[str] = None
latest_filesize: Optional[str] = None
received_at: Optional[str] = None


class StatusResponse(BaseModel):
    source: str
    timestamp: str
    template_id: str
    template_name: str
    template_width: str
    template_height: str
    filename: str
    resolution_w: str
    resolution_h: str
    filesize: str
    received_at: str


# ---------------------------------------------------------------------------
# Helper: dispatch callback ke Laravel (Tahap 9) — dipanggil dari process_ocr_background
# ---------------------------------------------------------------------------
def _dispatch_callback_sync(result: dict, saved_filename: str):
    """
    Kirim cleaned_text ke Laravel via httpx sync dengan HMAC + retry.

    Dijalankan di background thread setelah ocr_store.save(). Tidak raise — failure hanya log + dead letter.
    Import lazy agar service tetap jalan meski httpx belum install / config belum lengkap.
    """
    try:
        from app.callback.client import send_callback_sync

        # Ambil metadata untuk enrichment (source, template_id, dll)
        try:
            meta_all = load_metadata()
            meta_entry = meta_all.get(saved_filename)
        except Exception:
            meta_entry = None

        # Kirim — blocking di thread background, tidak blokir response POST
        cb_result = send_callback_sync(result, meta_entry)
        # Simpan status callback ke metadata untuk observability (opsional)
        try:
            meta_all2 = load_metadata()
            if saved_filename in meta_all2:
                meta_all2[saved_filename]["callback_last_attempt"] = datetime.now(JAKARTA_TZ).isoformat()
                meta_all2[saved_filename]["callback_success"] = cb_result.get("success", False)
                meta_all2[saved_filename]["callback_attempts"] = cb_result.get("attempts", 0)
                if cb_result.get("error"):
                    meta_all2[saved_filename]["callback_error"] = str(cb_result.get("error"))[:500]
                else:
                    meta_all2[saved_filename].pop("callback_error", None)
                save_metadata(meta_all2)
        except Exception as e:
            logger.warning(f"Failed to update metadata callback status for {saved_filename}: {e}")

    except ImportError as e:
        logger.warning(f"Callback module not available (httpx not installed?): {e}")
        print(f"[CALLBACK] Skipped — module not available: {e}")
    except Exception as e:
        # Jangan crash OCR karena callback gagal
        logger.warning(f"Callback dispatch failed for {saved_filename}: {e}", exc_info=True)
        print(f"[CALLBACK] Dispatch error for {saved_filename}: {e}")


# ---------------------------------------------------------------------------
# Helper: proses OCR di background (Tahap 4-7 + 9 callback)
# ---------------------------------------------------------------------------
def _is_temp_capture(image_path: str) -> bool:
    """
    True jika path adalah tempfile capture F7 (boleh dihapus setelah OCR).

    Kriteria ketat (anti salah hapus): di dalam system tempdir DAN
    basename berprefix labelcap_. File uploads/ & check TIDAK pernah cocok.
    """
    try:
        p = str(image_path or "")
        if not p or os.path.basename(p)[:9] != "labelcap_":
            return False
        tmpdir = os.path.realpath(tempfile.gettempdir())
        return os.path.realpath(p).startswith(tmpdir + os.sep)
    except Exception:
        return False


def _is_empty_result(cleaned: dict) -> bool:
    """Helper: cek apakah hasil OCR kosong (no text) — Fix Blank 2026-09-06."""
    # cleaned = {"cleaned_text": str, "lines": list, "raw_text": str}
    if not cleaned:
        return True
    # OCR Engine Upgrade (2026-09-11): 'none' = hasil VALID (tidak ada teks),
    # BUKAN kosong. Case-insensitive sesuai kontrak plans/OCR_ENGINE_UPGRADE_PADDLE_PLAN.md §6.
    try:
        from app.ocr.parser import is_none_text

        if is_none_text(cleaned.get("cleaned_text", "")):
            return False
    except ImportError:
        if isinstance(cleaned.get("cleaned_text", ""), str) and cleaned.get("cleaned_text", "").strip().lower() == "none":
            return False
    ct = cleaned.get("cleaned_text", "")
    lines = cleaned.get("lines", [])
    # kosong jika cleaned_text strip kosong dan lines kosong atau hanya whitespace
    if not ct or not ct.strip():
        return True
    if not lines or not any(ln.strip() for ln in lines if isinstance(ln, str)):
        return True
    return False


def _build_ocr_result_from_dispatch(ocr_out: dict, original_filename: str, saved_filename: str,
                                   file_size: int, elapsed: float, queue_wait_s: float = 0.0) -> dict:
    """Bangun result standar dari ocr_out dispatcher + timings aditif (R0/R1/R2)."""
    engine = ocr_out.get("engine", "tesseract")
    cleaned = {
        "raw_text": ocr_out.get("raw_text", ""),
        "cleaned_text": ocr_out.get("cleaned_text", ""),
        "lines": ocr_out.get("lines", []),
    }
    text_present = ocr_out.get("text_present")
    if engine == "rapidocr":
        rec_model = ocr_out.get("rec_model") or "PP-OCRv6_rec_small"
        det_model = ocr_out.get("det_model") or "PP-OCRv6_det_small"
        try:
            from app import config as _cfg

            _max_side = int(getattr(_cfg, "OCR_MAX_SIDE_PX", 0) or 0)
        except Exception:
            _max_side = 0
        if _max_side:
            preprocessing_steps = [f"rapidocr_det_rec({det_model}+{rec_model})", f"resize_max_side_{_max_side}"]
        else:
            preprocessing_steps = [f"rapidocr_det_rec({det_model}+{rec_model})", "original_no_upscale"]
        preprocessing_attempts = [f"rapidocr_det_rec({rec_model})"]
        used_pass = 1
        print(f"[OCR] RapidOCR({rec_model}): {saved_filename} -> {len(cleaned['lines'])} lines, text_present={text_present}")
    elif engine == "surya":
        preprocessing_steps = ["surya_segformer_det3_rec2", "rgb_normalize"]
        preprocessing_attempts = ["surya_multilingual"]
        used_pass = 1
        print(f"[OCR] Surya OCR: {saved_filename} -> {len(cleaned['lines'])} lines, text_present={text_present}")
    elif engine == "vitstr":
        preprocessing_steps = ["vitstr_parseq_tiny", "resize_normalize"]
        preprocessing_attempts = ["vitstr_strhub"]
        used_pass = 1
        print(f"[OCR] ViTSTR-Tiny: {saved_filename} -> {len(cleaned['lines'])} lines, text_present={text_present}")
    else:
        preprocessing_steps = ocr_out.get("preprocessing_steps", ["grayscale", "clahe", "autocontrast", "upscale_2x", "sharpen", "medianBlur_3"])
        preprocessing_attempts = ocr_out.get("preprocessing_attempts", ["v1_psm6"])
        used_pass = ocr_out.get("preprocessed_pass", 1)
        print(f"[OCR] Tesseract: {saved_filename} -> {len(cleaned['lines'])} lines, pass {used_pass}")

    if engine == "rapidocr" and text_present is False:
        status = "success"
        error_msg = None
        is_empty = False
        empty_reason = "no_text_detected"
        print(f"[OCR] No text detected: {saved_filename} -> cleaned_text='none' ({elapsed:.2f}s)")
        logger.info(f"[OCR] No text detected (none) {saved_filename}")
    else:
        is_empty = _is_empty_result(cleaned)
        if is_empty:
            status = "failed"
            error_msg = "empty cleaned_text after 2-pass — dark low-contrast, need retake or better lighting (preprocess fallback required)"
            empty_reason = "no_text_detected_after_2pass"
            print(f"[OCR] Failed (empty): {saved_filename} -> 0 lines after 2-pass ({elapsed:.2f}s) — {error_msg}")
            logger.warning(f"[OCR] Failed (empty) {saved_filename}: {error_msg}")
        else:
            status = "success"
            error_msg = None
            empty_reason = None

    timings = dict(ocr_out.get("timings") or {})
    try:
        from app import config as _cfg2

        if bool(getattr(_cfg2, "OCR_TIMINGS_ENABLED", True)):
            timings["queue_wait_s"] = round(float(queue_wait_s or 0.0), 4)
            timings["total_s"] = round(float(elapsed), 4)
    except Exception:
        pass

    try:
        from app import config as _cfg3

        _ocr_cfg = {
            "max_side_px": int(getattr(_cfg3, "OCR_MAX_SIDE_PX", 0) or 0),
            "workers": int(getattr(_cfg3, "OCR_WORKERS", 1) or 1),
            "cascade": bool(getattr(_cfg3, "OCR_CASCADE_ENABLED", False)),
        }
    except Exception:
        _ocr_cfg = {}

    result = {
        "filename": saved_filename,
        "original_filename": original_filename,
        "engine": engine,
        "language": "en" if engine == "rapidocr" else TESSERACT_LANG,
        "text_present": text_present,
        "det_threshold": ocr_out.get("det_threshold"),
        "rec_threshold": ocr_out.get("rec_threshold"),
        "det_model": ocr_out.get("det_model"),
        "rec_model": ocr_out.get("rec_model"),
        "preprocessed": True,
        "preprocessing_steps": preprocessing_steps,
        "preprocessing_attempts": preprocessing_attempts,
        "preprocessed_pass": used_pass,
        "raw_text": cleaned["raw_text"] if cleaned else "",
        "cleaned_text": cleaned["cleaned_text"] if cleaned else "",
        "lines": cleaned["lines"] if cleaned else [],
        "is_empty": is_empty,
        "empty_reason": empty_reason,
        "processed_at": datetime.now(JAKARTA_TZ).isoformat(),
        "status": status,
        "error": error_msg,
        "file_size_bytes": file_size,
        "processing_time_seconds": round(elapsed, 3),
        "image_scale": ocr_out.get("image_scale", 1.0),
        "timings": timings,
        "ocr_config": _ocr_cfg,
    }
    return result


def process_ocr_job(image_bytes: bytes, original_filename: str, saved_filename: str, enqueued_at: float | None = None):
    """
    Job OCR baru (R1-R6): fungsi sync, dijalankan di OCR_EXECUTOR.

    - Decode dari bytes (R3), engine thread-local warm (R1), resize R2.
    - store.put (bangunkan long-poll R4) → submit callback non-blocking (R5).
    - Selalu simpan success/failed agar polling tidak 404 buta.
    """
    queue_wait = max(0.0, time.time() - float(enqueued_at or time.time()))
    start = time.time()
    try:
        # Low-res warning dari bytes (tanpa file).
        try:
            import io as _io

            from PIL import Image as _PILImage

            with _PILImage.open(_io.BytesIO(image_bytes)) as _tmp:
                _w, _h = _tmp.size
            if _w < 400 or _h < 400:
                print(f"[OCR WARN] Low-res capture {_w}x{_h} for {saved_filename} — DPI may be insufficient, suggest retake with higher res")
                logger.warning(f"Low-res image {_w}x{_h} for {saved_filename} — may cause blank OCR")
        except Exception:
            pass

        try:
            from app.ocr.dispatch import extract_label_text_from_bytes

            ocr_out = extract_label_text_from_bytes(image_bytes, original_filename)
        except Exception as e_pre:
            logger.error(f"OCR engine failed for {saved_filename}: {e_pre}", exc_info=True)
            raise

        elapsed = time.time() - start
        result = _build_ocr_result_from_dispatch(ocr_out, original_filename, saved_filename, len(image_bytes or b""), elapsed, queue_wait)
        ocr_store.save(result)

        try:
            metadata = load_metadata()
            if saved_filename in metadata:
                metadata[saved_filename]["ocr_status"] = result["status"]
                metadata[saved_filename]["ocr_processed_at"] = result["processed_at"]
                metadata[saved_filename]["ocr_is_empty"] = result["is_empty"]
                if result["is_empty"]:
                    metadata[saved_filename]["ocr_error"] = result["error"]
                else:
                    metadata[saved_filename].pop("ocr_error", None)
                save_metadata(metadata)
        except Exception as e:
            logger.warning(f"Failed to update metadata ocr_status {result['status']} for {saved_filename}: {e}")

        if not result["is_empty"]:
            print(f"[OCR] Success: {saved_filename} -> {len(result['cleaned_text'])} chars, {elapsed:.2f}s (queue {queue_wait:.2f}s)")
            print(f"[OCR] File: {saved_filename} (original: {original_filename})")
            print("=" * 60)
            print(f"[OCR RAW] file={saved_filename} original={original_filename} engine={result['engine']} text_present={result['text_present']} chars={len(result['cleaned_text'])}")
            print(f"[OCR RAW] cleaned_text: {repr(result['cleaned_text'])}")
            print("=" * 60)
            try:
                tm = result.get("timings") or {}
                print(f"OCR_DONE file={saved_filename} total={elapsed:.2f}s queue={queue_wait:.2f}s decode={tm.get('decode_s')} det={tm.get('det_s')} rec={tm.get('rec_s')} lines={len(result['lines'])} engine={result['engine']}")
            except Exception:
                pass

        # R5: callback non-blocking di pool sendiri.
        try:
            from app.callback.client import submit_callback

            submit_callback(saved_filename, result)
        except Exception as e:
            logger.warning(f"Callback outer error for {saved_filename}: {e}", exc_info=True)

    except Exception as e:
        elapsed = time.time() - start
        logger.error(f"OCR failed for {saved_filename}: {e}", exc_info=True)
        print(f"[OCR] Failed: {saved_filename} -> {e}")
        result = {
            "filename": saved_filename,
            "original_filename": original_filename,
            "engine": OCR_ENGINE if OCR_ENGINE in ("vitstr", "surya") else ("rapidocr" if OCR_ENGINE == "paddle" else "tesseract"),
            "language": "en" if OCR_ENGINE in ("paddle", "vitstr", "surya") else TESSERACT_LANG,
            "text_present": False,
            "det_threshold": None,
            "rec_threshold": None,
            "preprocessed": True,
            "preprocessing_steps": ["grayscale", "clahe", "autocontrast", "upscale_2x", "sharpen", "medianBlur_3"],
            "preprocessing_attempts": ["exception"],
            "preprocessed_pass": 0,
            "raw_text": "",
            "cleaned_text": "",
            "lines": [],
            "is_empty": True,
            "empty_reason": "exception",
            "processed_at": datetime.now(JAKARTA_TZ).isoformat(),
            "status": "failed",
            "error": str(e),
            "file_size_bytes": len(image_bytes or b""),
            "processing_time_seconds": round(elapsed, 3),
            "timings": {"queue_wait_s": round(queue_wait, 4), "total_s": round(elapsed, 4)},
        }
        ocr_store.save(result)
        try:
            metadata = load_metadata()
            if saved_filename in metadata:
                metadata[saved_filename]["ocr_status"] = "failed"
                metadata[saved_filename]["ocr_error"] = str(e)
                metadata[saved_filename]["ocr_processed_at"] = result["processed_at"]
                metadata[saved_filename]["ocr_is_empty"] = True
                save_metadata(metadata)
        except Exception as ex:
            logger.warning(f"Failed to update metadata ocr_status failed for {saved_filename}: {ex}")
        try:
            from app.callback.client import submit_callback as _sc

            _sc(saved_filename, result)
        except Exception as e2:
            logger.warning(f"Callback outer error (failed branch) for {saved_filename}: {e2}", exc_info=True)


def process_ocr_background(image_path: str, original_filename: str, saved_filename: str):
    """
    Background task LEGACY (path) — dipertahankan untuk kompatibilitas.

    Mendelegasikan ke process_ocr_job(bytes) agar R1+R2 otomatis berlaku,
    lalu hapus tempfile bila memenuhi _is_temp_capture (F7).
    """
    try:
        try:
            _data = Path(image_path).read_bytes()
        except Exception as e:
            logger.error(f"OCR failed for {saved_filename} ({image_path}): cannot read file: {e}", exc_info=True)
            process_ocr_job(b"", original_filename, saved_filename, enqueued_at=time.time())
            return
        process_ocr_job(_data, original_filename, saved_filename, enqueued_at=time.time())
    finally:
        # F7 (2026-09-15): hapus tempfile capture — sukses maupun gagal.
        # File uploads()/check TIDAK pernah cocok _is_temp_capture → aman dari salah hapus.
        try:
            if _is_temp_capture(image_path):
                os.unlink(image_path)
                print(f"[CLEANUP] tempfile removed: {saved_filename}")
        except Exception as e_unlink:
            logger.warning(f"Tempfile unlink failed for {saved_filename}: {e_unlink}")


# ---------------------------------------------------------------------------
# POST /api/test/status — Upload + Validasi + Simpan + Trigger OCR background
# ---------------------------------------------------------------------------
@app.post("/api/test/status")
async def post_status(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...),
    source: str = Form(...),
    timestamp: str = Form(...),
    template_id: str = Form(...),
    template_name: str = Form(...),
    template_width: str = Form(...),
    template_height: str = Form(...),
    filename: str = Form(...),
    resolution_w: str = Form(...),
    resolution_h: str = Form(...),
    filesize: str = Form(...),
    # --- Checking Label 2-capture enrichment (PLAN_CHECKING_LABEL_2_CAPTURE_OCR_CALLBACK §5.1) ---
    # Opsional, backward-compat: tanpa field ini alur lama tetap jalan.
    # Laravel Checking kirim check_session_id (CHK-...) + label_slot (1/2) agar
    # callback + polling bisa korelasikan Label 1 vs Label 2 (anti-tertukar/stale).
    check_session_id: Optional[str] = Form(None),
    label_slot: Optional[str] = Form(None),
    sheet_no: Optional[str] = Form(None),
    reference_id: Optional[str] = Form(None),
):
    global latest_source, latest_timestamp, latest_template_id, latest_template_name
    global latest_template_width, latest_template_height, latest_filename
    global latest_resolution_w, latest_resolution_h, latest_filesize
    global received_at

    # --- Rate Limit POST burst (FIX 429 dual-side front+back, 2026-09-15) ---
    # Dulu: 1 gambar/menit → BACK selalu 429. Sekarang: 5 req / 60s sliding-window
    # (preview front+back 2 + forward save 2 + 1 retake cadangan). Spam 6+/menit masih 429.
    from app.security import check_post_rate_limit
    from app.config import POST_RATE_LIMIT_REQUESTS, POST_RATE_LIMIT_WINDOW_SECONDS
    allowed, wait, remaining = check_post_rate_limit()
    if not allowed:
        # Return ideal 429 dengan Retry-After + X-RateLimit headers agar Laravel/Frontend bisa backoff
        # 429 di sini expected hanya jika spam >5/menit; alur normal front→back tidak kena.
        return _rate_limit_429_response(
            wait, remaining, POST_RATE_LIMIT_REQUESTS,
            f"Rate limit exceeded. Maximum {POST_RATE_LIMIT_REQUESTS} images per {POST_RATE_LIMIT_WINDOW_SECONDS} seconds. Please wait {int(wait)+1} seconds.",
        )

    # --- Baca bytes (sudah ada MAX size check setelahnya, tapi baca dulu) ---
    try:
        image_bytes = await image.read()
    except Exception:
        return JSONResponse(
            status_code=400,
            content={"success": False, "message": "Failed to read uploaded file"},
        )

    # --- Tahap 1: Validasi (ukuran, ekstensi, magic bytes) ---
    # Gunakan filename dari form (atau dari UploadFile.filename sebagai fallback)
    client_filename = filename or (image.filename or "upload.png")
    valid, err_msg = validate_image_file(image_bytes, client_filename)
    if not valid:
        # Tentukan status code: ukuran & format -> 400
        # Konsisten dengan plan: semua validasi upload = 400 kecuali rate limit 429
        return JSONResponse(
            status_code=400,
            content={"success": False, "message": err_msg},
        )

    # --- Sanitasi & generate nama aman (Tahap 1) ---
    # R3: tanpa tempfile — bytes diteruskan langsung ke worker OCR (memori).
    # saved_filename tetap di-generate (KEY hasil OCR + poll URL).
    saved_filename = generate_safe_filename(client_filename)
    file_path: object = f"<bytes:{len(image_bytes)}>"

    # --- Update in-memory latest_* (pakai saved_filename sebagai filename efektif) ---
    latest_source = source
    latest_timestamp = timestamp
    latest_template_id = template_id
    latest_template_name = template_name
    latest_template_width = template_width
    latest_template_height = template_height
    latest_filename = saved_filename
    latest_resolution_w = resolution_w
    latest_resolution_h = resolution_h
    latest_filesize = str(len(image_bytes))
    received_at = datetime.now(JAKARTA_TZ).isoformat()

    # --- Sanitasi enrichment Checking 2-label (opsional, aditif, tidak 422) ---
    # check_session_id: alfanumerik + dash/underscore, max 64 (contoh CHK-20261001-093015-a1b2)
    # label_slot: "1"/"2" (string form) — simpan apa adanya max 8 bila tak valid
    # reference_id: opsional audit, max 32
    import re as _re_check
    _check_session_clean = ""
    try:
        _check_session_clean = (check_session_id or "").strip()[:64]
        if _check_session_clean and not _re_check.match(r"^[A-Za-z0-9\-_]+$", _check_session_clean):
            # Jangan tolak request — simpan versi ter-sanitasi agar tidak merusak path/JSON
            _check_session_clean = _re_check.sub(r"[^A-Za-z0-9\-_]", "", _check_session_clean)[:64]
    except Exception:
        _check_session_clean = ""
    _label_slot_clean: object = ""
    try:
        _slot_raw = (label_slot or "").strip()[:8]
        if _slot_raw in ("1", "2", "1.0", "2.0"):
            _label_slot_clean = int(float(_slot_raw))
        elif _slot_raw:
            try:
                _label_slot_clean = int(_slot_raw)
            except ValueError:
                _label_slot_clean = _slot_raw
        else:
            _label_slot_clean = ""
    except Exception:
        _label_slot_clean = ""
    _reference_id_clean = ""
    try:
        _reference_id_clean = (reference_id or "").strip()[:32]
    except Exception:
        _reference_id_clean = ""
    _sheet_no_clean: object = ""
    try:
        _sn_raw = (sheet_no or "").strip()[:8]
        if _sn_raw:
            _sn_val = int(float(_sn_raw))
            if 1 <= _sn_val <= 4:
                _sheet_no_clean = _sn_val
    except Exception:
        _sheet_no_clean = ""

    # --- Update metadata.json (Tahap 2) ---
    try:
        metadata = load_metadata()
        metadata[saved_filename] = {
            "source": source,
            "timestamp": timestamp,
            "template_id": template_id,
            "template_name": template_name,
            "template_width": template_width,
            "template_height": template_height,
            "original_filename": client_filename,
            "saved_as": saved_filename,
            "received_at": received_at,
            "file_size": len(image_bytes),
            "resolution_w": resolution_w,
            "resolution_h": resolution_h,
            # Checking Label 2-capture korelasi (PLAN_CHECKING_LABEL_2_CAPTURE_OCR_CALLBACK §5.1).
            # Kosong (""/"") bila alur lama tanpa field ini — callback/polling tetap jalan.
            "check_session_id": _check_session_clean,
            "label_slot": _label_slot_clean,
            "sheet_no": _sheet_no_clean,
            "reference_id": _reference_id_clean,
            # [OCR-REENABLE 2026-09-11 / PLAN_REVERT_TO_OCR] — kembali ke OCR awal.
            #     Marker lama [OCR-DISABLED 2026-09-10] dibiarkan sebagai jejak di bawah.
            # [OCR-DISABLED 2026-09-10 / PLAN_VISUAL_COMPARE_MODE]
            #     Mode Komparasi Visual: OCR tidak dijalankan.
            #     CARA AKTIFKAN KEMBALI: ganti "skipped-visual-compare-mode" → "pending".
            "ocr_status": "pending",
        }
        save_metadata(metadata)
    except Exception as e:
        logger.warning(f"Failed to save metadata for {saved_filename}: {e}")

    # --- CLEANUP RETAKE dipindah ke background (R9) agar POST cepat ---
    # Tujuan sama (hemat storage + cegah stale), tapi tidak blokir response.
    # retain_last_n tetap dipakai (aman untuk split Label1/2, lihat FIX 2026-09-30).
    def _cleanup_retake_background(_keep: str):
        try:
            from app.storage import cleanup_old_files
            from app.config import CLEANUP_RETAKE, RETAIN_COUNT

            if not CLEANUP_RETAKE:
                print(f"[CLEANUP] Skipped (CLEANUP_RETAKE=false) for {_keep}")
                return
            stats = cleanup_old_files(_keep)
            try:
                try:
                    removed = ocr_store.retain_last_n(RETAIN_COUNT)
                except AttributeError:
                    try:
                        removed = ocr_store.retain_pair(_keep)
                    except AttributeError:
                        removed = ocr_store.retain_only(_keep)
                if removed > 0:
                    print(f"[CLEANUP] ocr_store memory: removed {removed} old entries, kept last {RETAIN_COUNT} (latest {_keep})")
            except AttributeError:
                try:
                    ocr_store.clear_all()
                    print("[CLEANUP] ocr_store memory cleared (fallback)")
                except Exception as e2:
                    logger.warning(f"ocr_store cleanup failed: {e2}")
            print(f"[CLEANUP] Retake cleanup done for {_keep}: {stats}")
        except Exception as e:
            logger.warning(f"Retake cleanup failed for {_keep}: {e}", exc_info=True)
            print(f"[CLEANUP ERROR] {_keep}: {e}")

    print(f"[TEST SIGNAL] Source: {source}")
    print(f"[TEST SIGNAL] Template: {template_name} ({template_id})")
    print(f"[TEST SIGNAL] Size: {template_width} x {template_height} mm")
    print(f"[TEST SIGNAL] Filename (client): {client_filename}")
    print(f"[TEST SIGNAL] Filename (saved): {saved_filename}")
    print(f"[TEST SIGNAL] -> Mapping: {client_filename} => {saved_filename}")
    print(f"[TEST SIGNAL] Resolution: {resolution_w} x {resolution_h} px")
    print(f"[TEST SIGNAL] Filesize: {len(image_bytes)} bytes")
    print(f"[TEST SIGNAL] Timestamp: {timestamp}")
    print(f"[TEST SIGNAL] Image: <received> -> {file_path}")
    if _check_session_clean or _label_slot_clean != "":
        print(f"[CHECK] session={_check_session_clean or '-'} slot={_label_slot_clean if _label_slot_clean != '' else '-'} ref={_reference_id_clean or '-'} -> {saved_filename}")
    print(f"[OCR] Queued: {saved_filename} (original: {client_filename})")

    # --- Tahap 3 + 4-7: Trigger OCR via executor terkontrol (R6), cleanup via background (R9) ---
    # R3: bytes langsung (tanpa tempfile). R6: antrean terbatas → 503 bila penuh.
    try:
        from app.ocr.executor import QueueFull, submit_ocr

        _enq = time.time()
        try:
            submit_ocr(process_ocr_job, image_bytes, client_filename, saved_filename, _enq)
        except QueueFull:
            return JSONResponse(
                status_code=503,
                headers={"Retry-After": "3"},
                content={"success": False, "message": "OCR queue full, retry shortly", "error": "queue_full"},
            )
    except Exception as e:
        logger.warning(f"submit_ocr failed, fallback BackgroundTasks: {e}")
        background_tasks.add_task(process_ocr_job, image_bytes, client_filename, saved_filename, time.time())
    # R9: cleanup + persist metadata lanjutan di background (setelah response).
    background_tasks.add_task(_cleanup_retake_background, saved_filename)

    # request_id = saved tanpa ekstensi, untuk korelasi anti-stale
    request_id = saved_filename.rsplit(".", 1)[0] if "." in saved_filename else saved_filename
    return JSONResponse(
        content={
            "success": True,
            "message": "Image uploaded successfully, OCR processing started",
            "filename": saved_filename,
            "original_filename": client_filename,
            "received_at": received_at,
            "request_id": request_id,
            "ocr_status": "pending",
            "source": source,
            "template_id": template_id,
            # Checking Label 2-capture echo (aditif; "" bila alur lama tanpa field ini)
            "check_session_id": _check_session_clean,
            "label_slot": _label_slot_clean,
            "sheet_no": _sheet_no_clean,
            "reference_id": _reference_id_clean,
            "poll_url": f"/api/test/ocr/{saved_filename}",
            "image_url": f"/api/test/image/{saved_filename}",
            # FIX timeout Label 1/2 (2026-10-06): beri tahu client berapa lama
            # polling retryable agar tidak timeout dini. Aditif.
            "retryable": True,
            "retry_after_seconds": 2,
            "pending_timeout_seconds": 180,
        },
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


# ---------------------------------------------------------------------------
# GET /api/test/status — Ambil status (support ?filename= untuk filter spesifik)
# ---------------------------------------------------------------------------
@app.get("/api/test/status")
async def get_status(filename: Optional[str] = None):
    """
    Ambil status.
    - Tanpa query: return latest (behavior lama, tetap support) — FIX RC-3 tetap kompatibel
    - Dengan ?filename=xxx: return status untuk file spesifik (resolve original/saved via metadata.json).
      Menerima BAIK saved_filename MAUPUN original_filename.
    Ideal polling 429 handling: endpoint ini termasuk GET rate limit 60/min. Client harus pakai ?filename= spesifik,
    bukan polling tanpa param yang bisa stale. Frontend poll 3s×10 =10 req/30s aman (20/min). Jangan spam 6 URL/poll.
    """
    # --- 429 Ideal: cek GET rate limit (60/min) ---
    rl = _enforce_get_rate_limit()
    if rl:
        return rl
    # Jika ada query param ?filename=xxx → filter spesifik
    if filename:
        from app.storage import resolve_filename

        safe_req = sanitize_filename_for_lookup(filename)
        resolved = resolve_filename(safe_req)
        lookup = resolved if resolved else safe_req

        metadata = load_metadata()
        # Coba lookup langsung
        meta = metadata.get(lookup) or metadata.get(safe_req)
        # Fallback reverse lookup jika resolve gagal tapi ada sebagai original
        if not meta:
            for k, v in metadata.items():
                if sanitize_filename_for_lookup(v.get("original_filename", "")) == safe_req:
                    meta = v
                    lookup = k
                    break
                if sanitize_filename_for_lookup(v.get("saved_as", "")) == safe_req:
                    meta = v
                    lookup = k
                    break

        if not meta:
            print(f"[TEST SIGNAL] GET status?filename={filename} -> 404 not found (resolved: {lookup})")
            return JSONResponse(
                status_code=404,
                content={"detail": f"No status data for filename: {filename}"},
                headers={
                    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                    "Pragma": "no-cache",
                    "Expires": "0",
                },
            )

        # Return data spesifik (bukan latest)
        print(f"[TEST SIGNAL] GET status?filename={filename} -> {lookup} (original: {meta.get('original_filename','')})")
        print(f"[TEST SIGNAL] Source: {meta.get('source','')}")
        print(f"[TEST SIGNAL] Template: {meta.get('template_name','')} ({meta.get('template_id','')})")
        print(f"[TEST SIGNAL] Timestamp: {meta.get('timestamp','')}")

        return JSONResponse(
            content={
                "source": meta.get("source", ""),
                "timestamp": meta.get("timestamp", ""),
                "template_id": meta.get("template_id", ""),
                "template_name": meta.get("template_name", ""),
                "template_width": meta.get("template_width", ""),
                "template_height": meta.get("template_height", ""),
                "filename": meta.get("saved_as", lookup),
                "original_filename": meta.get("original_filename", ""),
                "resolution_w": meta.get("resolution_w", ""),
                "resolution_h": meta.get("resolution_h", ""),
                "filesize": str(meta.get("file_size", "")),
                "received_at": meta.get("received_at", ""),
                "ocr_status": meta.get("ocr_status", "unknown"),
                "check_session_id": meta.get("check_session_id", ""),
                "label_slot": meta.get("label_slot", ""),
                "reference_id": meta.get("reference_id", ""),
            },
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )

    # Tanpa query: behavior lama (latest)
    if latest_source is None:
        return JSONResponse(
            status_code=404,
            content={"detail": "No status data available yet"},
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )

    print(f"[TEST SIGNAL] GET requested (latest)")
    print(f"[TEST SIGNAL] Source: {latest_source}")
    print(f"[TEST SIGNAL] Template: {latest_template_name} ({latest_template_id})")
    print(f"[TEST SIGNAL] Timestamp: {latest_timestamp}")

    return JSONResponse(
        content={
            "source": latest_source,
            "timestamp": latest_timestamp,
            "template_id": latest_template_id,
            "template_name": latest_template_name,
            "template_width": latest_template_width,
            "template_height": latest_template_height,
            "filename": latest_filename,
            "resolution_w": latest_resolution_w,
            "resolution_h": latest_resolution_h,
            "filesize": latest_filesize,
            "received_at": received_at,
        },
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


# ---------------------------------------------------------------------------
# GET /api/test/image/{filename} — Download image (support original→saved, path traversal safe)
# ---------------------------------------------------------------------------
@app.get("/api/test/image/{filename}")
async def get_image(filename: str):
    # --- 429 Ideal: GET rate limit ---
    rl = _enforce_get_rate_limit()
    if rl:
        return rl
    from app.storage import resolve_filename

    # Sanitasi: hanya basename
    safe_name = sanitize_filename_for_lookup(filename)
    if not safe_name or safe_name != os.path.basename(filename):
        # Jika ada percobaan traversal, tolak
        # Tapi tetap izinkan normal file dengan karakter aman
        pass

    # Resolve original -> saved (FIX: support original filename)
    resolved = resolve_filename(safe_name)
    lookup_name = resolved if resolved else safe_name
    if lookup_name != safe_name:
        print(f"[IMAGE LOOKUP] {safe_name} -> {lookup_name}")

    # Cegah path traversal
    file_path = UPLOAD_DIR / lookup_name
    if not is_safe_path(UPLOAD_DIR, file_path):
        return JSONResponse(
            status_code=404,
            content={"success": False, "message": "File not found"},
        )

    if not file_path.exists() or not file_path.is_file():
        # Fallback: coba safe_name langsung (untuk file tanpa metadata)
        fallback = UPLOAD_DIR / safe_name
        if fallback.exists() and fallback.is_file() and is_safe_path(UPLOAD_DIR, fallback):
            file_path = fallback
            lookup_name = safe_name
        else:
            return JSONResponse(
                status_code=404,
                content={"success": False, "message": "File not found"},
            )

    try:
        # Tentukan media_type berdasarkan ekstensi
        mime, _ = mimetypes.guess_type(str(file_path))
        media_type = mime or "image/png"
        return FileResponse(
            path=str(file_path),
            media_type=media_type,
            filename=lookup_name,
        )
    except Exception as e:
        logger.error(f"Failed to serve image {lookup_name}: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": "Internal error"},
        )


# ---------------------------------------------------------------------------
# GET /api/test/files — List semua file yang tersimpan
# ---------------------------------------------------------------------------
@app.get("/api/test/files")
async def get_files():
    # --- 429 Ideal: GET rate limit (files list jangan di-poll spam; cukup sekali per capture) ---
    rl = _enforce_get_rate_limit()
    if rl:
        return rl
    metadata = load_metadata()
    files = []

    for fname, meta in metadata.items():
        # Sanitasi fname dari metadata (jaga-jaga metadata lama corrupt)
        safe_fname = sanitize_filename_for_lookup(fname)
        file_path = UPLOAD_DIR / safe_fname
        # Pastikan path aman
        if not is_safe_path(UPLOAD_DIR, file_path):
            continue
        if file_path.exists() and file_path.is_file():
            files.append(
                {
                    "filename": safe_fname,
                    "original_filename": meta.get("original_filename", ""),
                    "saved_as": meta.get("saved_as", safe_fname),
                    "size": file_path.stat().st_size,
                    "timestamp": meta.get("timestamp", ""),
                    "template_id": meta.get("template_id", ""),
                    "source": meta.get("source", ""),
                    "ocr_status": meta.get("ocr_status", "unknown"),
                }
            )

    return JSONResponse(
        content={"success": True, "files": files},
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


# ---------------------------------------------------------------------------
# OCR Endpoints (Tahap 8)
# ---------------------------------------------------------------------------
@app.get("/api/test/ocr/latest")
async def get_ocr_latest(filename: Optional[str] = Query(None, description="Optional filename filter — support saved or original (10.4 Filesystem External). Jika diisi, return OCR untuk file spesifik bukan latest.")):
    """
    Mengembalikan hasil OCR untuk gambar terbaru yang berhasil diproses.
    Cek memory dulu (ocr_store), fallback tidak perlu karena store sudah load disk saat init.
    DEPRECATED untuk scan flow — pakai GET /api/test/ocr/{filename} dengan saved_filename spesifik.
    Namun support ?filename= untuk Filesystem External spec (10.4): ocr/latest?filename= -> query spesifik.
    Jika ?filename= diisi, akan resolve original→saved dan return spesifik (tidak poll latest yang bisa stale).
    """
    # --- 429 Ideal: GET rate limit ---
    rl = _enforce_get_rate_limit()
    if rl:
        return rl

    # --- 10.4 Filesystem External: support ?filename= query ---
    if filename:
        # Delegasikan ke lookup spesifik (anti-stale)
        from app.storage import resolve_filename

        safe_req = sanitize_filename_for_lookup(filename)
        resolved = resolve_filename(safe_req)
        lookup = resolved if resolved else safe_req
        if resolved and resolved != safe_req:
            print(f"[OCR LATEST QUERY] ?filename={safe_req} -> {resolved} (saved)")

        # Coba ambil spesifik
        specific = ocr_store.get_by_filename(lookup)
        if specific is not None:
            # Jika ketemu, return spesifik dengan header sama seperti /ocr/{filename}
            headers = {
                "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                "Pragma": "no-cache",
                "Expires": "0",
                "ETag": f'W/"{specific.get("filename","")}-{specific.get("processed_at","")}"',
                "X-Resolved-From": filename if resolved and resolved != safe_req else "",
            }
            if specific.get("processed_at"):
                headers["Last-Modified"] = specific.get("processed_at")
            # Ambil metadata untuk received_at + korelasi Checking 2-label
            try:
                meta = load_metadata().get(specific.get("filename", lookup), {})
                effective_original = specific.get("original_filename", meta.get("original_filename", ""))
                received_at_meta = meta.get("received_at", "")
                check_session_meta_l = meta.get("check_session_id", "")
                label_slot_meta_l = meta.get("label_slot", "")
                reference_id_meta_l = meta.get("reference_id", "")
            except Exception:
                effective_original = specific.get("original_filename", "")
                received_at_meta = ""
                check_session_meta_l = ""
                label_slot_meta_l = ""
                reference_id_meta_l = ""
            fname_for_id = specific.get("filename", lookup)
            request_id = fname_for_id.rsplit(".", 1)[0] if "." in fname_for_id else fname_for_id
            _is_empty_s = specific.get("is_empty")
            if _is_empty_s is None:
                _ct_s_raw = specific.get("cleaned_text") or ""
                if isinstance(_ct_s_raw, str) and _ct_s_raw.strip().lower() == "none":
                    _is_empty_s = False  # kontrak 'none' = valid, bukan kosong
                else:
                    _ct_s = (_ct_s_raw.strip() if isinstance(_ct_s_raw, str) else "")
                    _lines_s = specific.get("lines") or []
                    _is_empty_s = not _ct_s and not any((ln or "").strip() for ln in _lines_s)
            return JSONResponse(
                content={
                    "success": True,
                    "filename": specific.get("filename", lookup),
                    "original_filename": effective_original,
                    "request_id": request_id,
                    "received_at": received_at_meta,
                    "check_session_id": check_session_meta_l,
                    "label_slot": label_slot_meta_l,
                    "reference_id": reference_id_meta_l,
                    "resolved_from": filename if resolved and resolved != safe_req else None,
                    "engine": specific.get("engine", "tesseract"),
                    "language": specific.get("language", "eng"),
                    "preprocessed": specific.get("preprocessed", True),
                    "preprocessing_steps": specific.get("preprocessing_steps", []),
                    "preprocessing_attempts": specific.get("preprocessing_attempts", []),
                    "preprocessed_pass": specific.get("preprocessed_pass"),
                    "is_empty": _is_empty_s,
                    "empty_reason": specific.get("empty_reason"),
                    "text_present": specific.get("text_present"),
                    "det_model": specific.get("det_model"),
                    "rec_model": specific.get("rec_model"),
                    "raw_text": specific.get("raw_text", ""),
                    "cleaned_text": specific.get("cleaned_text", ""),
                    "lines": specific.get("lines", []),
                    "processed_at": specific.get("processed_at", ""),
                    "status": specific.get("status", "unknown"),
                    "processing_time_seconds": specific.get("processing_time_seconds"),
                    "error": specific.get("error"),
                    "warning": specific.get("error") if _is_empty_s else None,
                    "empty_hint": "No text detected — try retake with better lighting" if _is_empty_s else None,
                    "queried_via": "latest?filename=",
                },
                headers=headers,
            )
        # Jika tidak ketemu spesifik, cek pending
        try:
            meta = load_metadata().get(lookup) or load_metadata().get(safe_req)
            if not meta:
                for k, v in load_metadata().items():
                    if sanitize_filename_for_lookup(v.get("original_filename", "")) == safe_req:
                        meta = v
                        break
            if meta and meta.get("ocr_status") == "pending":
                return JSONResponse(
                    status_code=200,
                    content={
                        "success": False,
                        "status": "pending",
                        "message": f"OCR still processing for file: {safe_req}, retry in 2 seconds",
                        "filename": lookup,
                        "original_filename": meta.get("original_filename", ""),
                        "check_session_id": meta.get("check_session_id", ""),
                        "label_slot": meta.get("label_slot", ""),
                        "reference_id": meta.get("reference_id", ""),
                        "retry_after_seconds": 2,
                        "retryable": True,
                        "poll_url": f"/api/test/ocr/{lookup}",
                        "pending_timeout_seconds": 180,
                        "queried_via": "latest?filename=",
                    },
                    headers={
                        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                        "Pragma": "no-cache",
                        "Expires": "0",
                        "Retry-After": "2",
                    },
                )
        except Exception:
            pass
        # Benar-benar tidak ada → 404
        return JSONResponse(
            status_code=404,
            content={
                "success": False,
                "status": "not_found",
                "message": f"OCR result not found for file: {safe_req} (queried via latest?filename=)",
                "resolved_as": lookup if resolved else None,
                "queried_via": "latest?filename=",
                "hint": "Poll poll_url (saved filename label-*.png) dari response POST, bukan original label-ref-*.png.",
            },
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )

    result = ocr_store.get_latest()
    if result is None:
        return JSONResponse(
            status_code=404,
            content={"success": False, "message": "No OCR result available yet"},
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )

    # Jika status failed, tetap return 200 dengan status failed agar client tahu
    headers = {
        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
        "Pragma": "no-cache",
        "Expires": "0",
        "X-Deprecated": "Use GET /api/test/ocr/{filename} with saved_filename from POST instead of /latest for scan flow",
        "ETag": f'W/"{result.get("filename","")}-{result.get("processed_at","")}"',
    }
    # Last-Modified jika ada
    if result.get("processed_at"):
        headers["Last-Modified"] = result.get("processed_at")

    _is_empty_l = result.get("is_empty")
    if _is_empty_l is None:
        _ct_l_raw = result.get("cleaned_text") or ""
        if isinstance(_ct_l_raw, str) and _ct_l_raw.strip().lower() == "none":
            _is_empty_l = False  # kontrak 'none' = valid, bukan kosong
        else:
            _ct_l = (_ct_l_raw.strip() if isinstance(_ct_l_raw, str) else "")
            _lines_l = result.get("lines") or []
            _is_empty_l = not _ct_l and not any((ln or "").strip() for ln in _lines_l)
    return JSONResponse(
        content={
            "success": True,
            "filename": result.get("filename", ""),
            "original_filename": result.get("original_filename", ""),
            "engine": result.get("engine", "tesseract"),
            "language": result.get("language", "eng"),
            "preprocessed": result.get("preprocessed", True),
            "preprocessing_steps": result.get("preprocessing_steps", []),
            "preprocessing_attempts": result.get("preprocessing_attempts", []),
            "preprocessed_pass": result.get("preprocessed_pass"),
            "is_empty": _is_empty_l,
            "empty_reason": result.get("empty_reason"),
            "text_present": result.get("text_present"),
            "det_model": result.get("det_model"),
            "rec_model": result.get("rec_model"),
            "raw_text": result.get("raw_text", ""),
            "cleaned_text": result.get("cleaned_text", ""),
            "lines": result.get("lines", []),
            "processed_at": result.get("processed_at", ""),
            "status": result.get("status", "unknown"),
            "processing_time_seconds": result.get("processing_time_seconds"),
            "error": result.get("error"),
            "warning": result.get("error") if _is_empty_l else "This endpoint is deprecated for scan flow; use GET /api/test/ocr/{filename} with saved filename from POST",
            "empty_hint": "No text detected — try retake" if _is_empty_l else None,
        },
        headers=headers,
    )


@app.get("/api/test/ocr/reconcile")
async def reconcile_ocr(since: Optional[str] = None, limit: int = 10):
    """
    Reconcile untuk Laravel: list OCR yang processed_at > since (anti-stale batch).
    Query: ?since=2026-09-01T10:00:00+07:00&limit=10
    Return max 10 terbaru, no-cache, hanya success.
    Berguna jika callback hilang dan Laravel mau sync batch.
    Penting: endpoint ini harus didefinisikan SEBELUM /{filename} agar tidak tertangkap sebagai filename="reconcile".
    """
    rl = _enforce_get_rate_limit()
    if rl:
        return rl
    limit = max(1, min(limit, 20))
    metadata = load_metadata()
    results = []
    for fname in metadata.keys():
        ocr = ocr_store.get_by_filename(fname)
        if ocr and ocr.get("status") == "success":
            if since and ocr.get("processed_at", "") <= since:
                continue
            results.append(ocr)
    results.sort(key=lambda x: x.get("processed_at", ""), reverse=True)
    results = results[:limit]
    return JSONResponse(
        content={"success": True, "count": len(results), "results": results, "since": since},
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@app.get("/api/test/ocr/{filename}")
async def get_ocr_by_filename(filename: str, wait: float = Query(0.0, description="Long-poll R4: tahan hingga hasil siap, maks OCR_LONGPOLL_MAX_SECONDS. 0 = perilaku lama.")):
    """
    Mengembalikan hasil OCR spesifik per file.
    Menerima BAIK saved_filename MAUPUN original_filename (resolve via metadata.json).
    Jika OCR masih pending, return 200 dengan status pending agar client tahu untuk retry.
    R4: bila ?wait>0 dan masih pending, server menunggu (long-poll) hingga hasil siap.
    Filename disanitasi (basename, hapus karakter berbahaya).
    Cek memory dulu, lalu disk (uploads/ocr/{filename}.json).
    Ideal 429: endpoint ini adalah TARGET UTAMA untuk polling spesifik (1 req/poll), rate limit 60/min aman.
    Laravel/Frontend harus poll endpoint ini saja (jangan 6 URL), interval 3s ×10 =10 req/30s → 20/min → tidak 429.
    """
    # --- 429 Ideal: GET rate limit ---
    rl = _enforce_get_rate_limit()
    if rl:
        return rl
    from app.storage import resolve_filename

    safe_name = sanitize_filename_for_lookup(filename)
    if not safe_name:
        return JSONResponse(
            status_code=404,
            content={"success": False, "message": f"OCR result not found for file: {filename}"},
        )

    # --- Resolve original -> saved (FIX RC-1) ---
    resolved = resolve_filename(safe_name)
    lookup_name = resolved if resolved else safe_name

    # Debug log: cetak mapping agar terlihat di console
    if resolved and resolved != safe_name:
        print(f"[OCR LOOKUP] {safe_name} (original) -> {resolved} (saved)")

    result = ocr_store.get_by_filename(lookup_name)

    # R4 long-poll: bila masih pending dan ?wait>0, tunggu hingga Event set.
    def _is_pending_key(_key: str) -> bool:
        try:
            _m = load_metadata().get(_key)
            return bool(_m and _m.get("ocr_status") == "pending")
        except Exception:
            return False

    if result is None and wait and wait > 0 and _is_pending_key(lookup_name):
        try:
            from app import config as _cfg

            _max_wait = float(getattr(_cfg, "OCR_LONGPOLL_MAX_SECONDS", 10) or 10)
        except Exception:
            _max_wait = 10.0
        try:
            from starlette.concurrency import run_in_threadpool as _rit

            await _rit(ocr_store.wait, lookup_name, min(float(wait), _max_wait))
            result = ocr_store.get_by_filename(lookup_name)
        except Exception:
            pass

    # Jika belum ada hasil di store/disk, cek apakah file ada di metadata dengan status pending
    if result is None:
        # Cek metadata untuk bedakan "pending" vs "not found"
        try:
            meta_all = load_metadata()
            meta = meta_all.get(lookup_name) or meta_all.get(safe_name)
            # Fallback reverse lookup jika resolve gagal tapi ada sebagai original
            if not meta:
                for k, v in meta_all.items():
                    if sanitize_filename_for_lookup(v.get("original_filename", "")) == safe_name:
                        meta = v
                        lookup_name = k
                        break
            if meta and meta.get("ocr_status") == "pending":
                print(f"[OCR] Pending: {lookup_name} (requested: {safe_name})")
                # FIX timeout Label 1/2 (2026-10-06): bedakan "masih processing,
                # boleh sambung lagi" vs "gagal". Tambah retryable + poll_url +
                # elapsed agar Laravel/JS bisa lanjut polling (Cek Lagi) tanpa
                # capture ulang. Aditif, kontrak lama tidak berubah.
                try:
                    _recv_iso = str(meta.get("received_at", "") or "")
                    _recv_dt = datetime.fromisoformat(_recv_iso) if _recv_iso else None
                    _elapsed_s = round((datetime.now(JAKARTA_TZ) - _recv_dt).total_seconds(), 1) if _recv_dt else None
                    if _elapsed_s is not None and _elapsed_s < 0:
                        _elapsed_s = 0.0
                except Exception:
                    _elapsed_s = None
                return JSONResponse(
                    status_code=200,
                    content={
                        "success": False,
                        "status": "pending",
                        "message": f"OCR still processing for file: {safe_name}, retry in 2 seconds",
                        "filename": lookup_name,
                        "original_filename": meta.get("original_filename", ""),
                        "check_session_id": meta.get("check_session_id", ""),
                        "label_slot": meta.get("label_slot", ""),
                        "reference_id": meta.get("reference_id", ""),
                        "retry_after_seconds": 2,
                        "retryable": True,
                        "poll_url": f"/api/test/ocr/{lookup_name}",
                        "elapsed_seconds": _elapsed_s,
                        "pending_timeout_seconds": 180,
                    },
                    headers={
                        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                        "Pragma": "no-cache",
                        "Expires": "0",
                        "Retry-After": "2",
                    },
                )
            # FIX 2026-09-11 — polling race original→saved:
            # Frontend (reference.js) sempat poll pakai original_filename
            # (label-ref-*.png) sementara POST baru return saved_filename
            # (label-*.png). Jika ada job pending lain yang masih fresh
            # (<OCR_PENDING_FRESH_SECONDS, default 180s), jangan 404
            # (browser log GET 404 sebagai error) —
            # return 200 pending + arahkan ke poll_url yang benar.
            # FIX timeout Label 1/2 (2026-10-06): window 120s -> 180s agar
            # menutup polling panjang frontend 120s + Laravel 150s.
            if not meta:
                try:
                    try:
                        from app import config as _cfg_fresh

                        _fresh_window = float(getattr(_cfg_fresh, "OCR_PENDING_FRESH_SECONDS", 180) or 180)
                    except Exception:
                        _fresh_window = 180.0
                    now_jkt = datetime.now(JAKARTA_TZ)
                    fresh_pending = None
                    for k, v in meta_all.items():
                        if v.get("ocr_status") != "pending":
                            continue
                        try:
                            recv = datetime.fromisoformat(str(v.get("received_at", "")))
                            age = (now_jkt - recv).total_seconds()
                        except Exception:
                            age = 9999
                        if age < _fresh_window:
                            if fresh_pending is None or str(v.get("received_at", "")) > str(fresh_pending[1].get("received_at", "")):
                                fresh_pending = (k, v)
                    if fresh_pending:
                        fk, fv = fresh_pending
                        print(f"[OCR] Pending-race: requested {safe_name} not found, redirect poll to fresh pending {fk}")
                        return JSONResponse(
                            status_code=200,
                            content={
                                "success": False,
                                "status": "pending",
                                "message": (
                                    f"OCR still processing (requested {safe_name} not yet resolved). "
                                    f"Poll saved file instead: {fk}. Retry in 2 seconds."
                                ),
                                "filename": fk,
                                "original_filename": fv.get("original_filename", ""),
                                "requested_filename": safe_name,
                                "resolved_as": lookup_name if resolved else None,
                                "poll_url": f"/api/test/ocr/{fk}",
                                "retry_after_seconds": 2,
                                "retryable": True,
                                "pending_timeout_seconds": 180,
                            },
                            headers={
                                "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                                "Pragma": "no-cache",
                                "Expires": "0",
                                "Retry-After": "2",
                            },
                        )
                except Exception:
                    pass
        except Exception:
            pass

        # Benar-benar tidak ada — beri status + hint agar frontend bedakan
        # transient (retry) vs permanen. Frontend harus treat 404 ini sebagai
        # "belum ada, jangan error keras": polling pakai saved_filename dari
        # POST /api/test/status (field poll_url), bukan original_filename.
        return JSONResponse(
            status_code=404,
            content={
                "success": False,
                "status": "not_found",
                "message": f"OCR result not found for file: {safe_name}",
                "resolved_as": lookup_name if resolved else None,
                "hint": "Jika baru upload, poll poll_url (saved filename label-*.png) dari response POST, bukan original label-ref-*.png. Tunggu callback / retry 2s.",
            },
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )

    # Sukses: kembalikan dengan info tambahan original_filename + received_at/request_id untuk anti-stale
    original = ""
    received_at_meta = ""
    check_session_meta: object = ""
    label_slot_meta: object = ""
    reference_id_meta = ""
    try:
        meta = load_metadata().get(result.get("filename", lookup_name), {})
        original = meta.get("original_filename", "")
        received_at_meta = meta.get("received_at", "")
        # Checking Label 2-capture korelasi (aditif, "" bila alur lama)
        check_session_meta = meta.get("check_session_id", "")
        label_slot_meta = meta.get("label_slot", "")
        reference_id_meta = meta.get("reference_id", "")
    except Exception:
        pass

    # Jika result sudah punya original_filename, pakai itu; fallback ke metadata
    effective_original = result.get("original_filename", original)
    # request_id = filename tanpa ekstensi
    fname_for_id = result.get("filename", lookup_name)
    request_id = fname_for_id.rsplit(".", 1)[0] if "." in fname_for_id else fname_for_id

    headers = {
        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
        "Pragma": "no-cache",
        "Expires": "0",
        "ETag": f'W/"{fname_for_id}-{result.get("processed_at","")}"',
    }
    if result.get("processed_at"):
        headers["Last-Modified"] = result.get("processed_at")

    # Fix Blank: tambah is_empty/warning untuk UI retake
    # Kontrak 'none' (2026-09-11): cleaned_text 'none' = VALID, bukan kosong.
    _is_empty = result.get("is_empty")
    if _is_empty is None:
        # Fallback untuk data lama sebelum fix — kecualikan 'none' eksplisit
        try:
            from app.ocr.parser import is_none_text as _is_none

            if _is_none(result.get("cleaned_text", "")):
                _is_empty = False
            else:
                _ct = (result.get("cleaned_text") or "").strip()
                _lines = result.get("lines") or []
                _is_empty = not _ct and not any((ln or "").strip() for ln in _lines)
        except ImportError:
            _ct_raw = result.get("cleaned_text") or ""
            if isinstance(_ct_raw, str) and _ct_raw.strip().lower() == "none":
                _is_empty = False
            else:
                _ct = (_ct_raw.strip() if isinstance(_ct_raw, str) else "")
                _lines = result.get("lines") or []
                _is_empty = not _ct and not any((ln or "").strip() for ln in _lines)
    _empty_reason = result.get("empty_reason")
    _attempts = result.get("preprocessing_attempts", [])
    _pass = result.get("preprocessed_pass")
    _warning = None
    _hint = None
    if _is_empty:
        _warning = result.get("error") or "OCR completed but no text detected. Image may be blank, too dark, or low resolution. Try retake with better lighting."
        _hint = "No text detected — check image quality, lighting, or preprocessing. 2-pass CLAHE+adaptive was tried."
    # Untuk failed empty, beri pesan lebih jelas
    if result.get("status") == "failed" and _is_empty:
        _warning = result.get("error") or _warning

    return JSONResponse(
        content={
            "success": True,
            "filename": result.get("filename", lookup_name),
            "original_filename": effective_original,
            "request_id": request_id,
            "poll_url": f"/api/test/ocr/{result.get('filename', lookup_name)}",
            "received_at": received_at_meta,
            "check_session_id": check_session_meta,
            "label_slot": label_slot_meta,
            "reference_id": reference_id_meta,
            "resolved_from": safe_name if resolved and resolved != safe_name else None,
            "engine": result.get("engine", "tesseract"),
            "language": result.get("language", "eng"),
            "preprocessed": result.get("preprocessed", True),
            "preprocessing_steps": result.get("preprocessing_steps", []),
            "preprocessing_attempts": _attempts,
            "preprocessed_pass": _pass,
            "is_empty": _is_empty,
            "empty_reason": _empty_reason,
            "text_present": result.get("text_present"),
            "det_model": result.get("det_model"),
            "rec_model": result.get("rec_model"),
            "raw_text": result.get("raw_text", ""),
            "cleaned_text": result.get("cleaned_text", ""),
            "lines": result.get("lines", []),
            "processed_at": result.get("processed_at", ""),
            "status": result.get("status", "unknown"),
            "processing_time_seconds": result.get("processing_time_seconds"),
            "error": result.get("error"),
            "warning": _warning,
            "empty_hint": _hint,
            "timings": result.get("timings"),
            "image_scale": result.get("image_scale", 1.0),
            "ocr_config": result.get("ocr_config"),
        },
        headers=headers,
    )


@app.get("/api/test/result/{filename}")
async def get_result_by_filename(filename: str, wait: float = Query(0.0)):
    """
    ALIAS untuk GET /api/test/ocr/{filename} — backward compatible.

    Dibuat karena client lama (Laravel/test script) memanggil /api/test/result/* yang belum ada → 404.
    Sekarang arahkan ke handler yang sama agar tidak 404 (FIX RC-2).
    """
    return await get_ocr_by_filename(filename, wait=wait)


# ---------------------------------------------------------------------------
# Alias Laravel-style tanpa /test — FIX 2026-09-11 untuk 404 polling
# ---------------------------------------------------------------------------
# Log frontend: GET http://127.0.0.1:8000/api/ocr/label-ref-*.png → 404.
# Penyebab: frontend poll path Laravel (/api/ocr/...) pakai original_filename,
# sementara FastAPI hanya serve /api/test/ocr/... pakai saved_filename.
# Alias ini membuat FastAPI toleran bila frontend/proxy salah alamat ke
# FastAPI langsung dengan path pendek: /api/ocr/{filename} → handler sama
# (resolve original→saved + 200 pending, bukan 404).
# PENTING: /latest dan /reconcile harus didefinisikan SEBELUM /{filename}
# agar tidak tertangkap sebagai filename="latest"/"reconcile".
@app.get("/api/ocr/latest")
async def get_ocr_alias_latest(filename: Optional[str] = Query(None, description="Optional filename filter — sama seperti /api/test/ocr/latest?filename=")):
    """Alias Laravel-style untuk GET /api/test/ocr/latest (support ?filename=)."""
    return await get_ocr_latest(filename=filename)


@app.get("/api/ocr/reconcile")
async def reconcile_ocr_alias(since: Optional[str] = None, limit: int = 10):
    """Alias Laravel-style untuk GET /api/test/ocr/reconcile."""
    return await reconcile_ocr(since=since, limit=limit)


@app.get("/api/ocr/{filename}")
async def get_ocr_alias(filename: str, wait: float = Query(0.0)):
    """
    Alias Laravel-style untuk GET /api/test/ocr/{filename}.

    Menerima saved ATAU original filename, return 200 pending bila masih
    diproses (agar browser tidak log GET 404 sebagai error).
    Query cache-buster seperti ?_t=... diabaikan (tidak divalidasi).
    """
    return await get_ocr_by_filename(filename, wait=wait)


@app.post("/api/test/ocr/run")
async def run_ocr_manual(background_tasks: BackgroundTasks):
    """
    Endpoint manual untuk memicu OCR ulang pada file terbaru (fallback via get_latest_image).
    Berguna saat testing tanpa upload baru.

    Rate limit TIDAK berlaku untuk endpoint ini (manual trigger).
    Return 404 jika tidak ada gambar sama sekali.
    """
    # [OCR-REENABLE 2026-09-11 / PLAN_REVERT_TO_OCR] — gate 503 di bawah DINONAKTIFKAN,
    #     OCR manual aktif kembali. Jejak marker lama dibiarkan.
    # [OCR-DISABLED 2026-09-10 / PLAN_VISUAL_COMPARE_MODE]
    #     Mode Komparasi Visual: trigger OCR manual dinonaktifkan sementara.
    #     CARA AKTIFKAN KEMBALI: hapus blok return 503 di bawah ini.
    # return JSONResponse(
    #     status_code=503,
    #     content={
    #         "success": False,
    #         "message": "OCR disabled (visual compare mode). Aktifkan kembali: lihat marker [OCR-DISABLED] di main.py.",
    #         "error": "ocr_disabled",
    #     },
    # )
    latest_path = get_latest_image()
    if not latest_path:
        return JSONResponse(
            status_code=404,
            content={"success": False, "message": "No image file found in uploads/"},
        )

    saved_filename = os.path.basename(latest_path)
    # Untuk manual run, kita tidak punya original_filename, pakai saved_filename
    original_filename = saved_filename

    # Update metadata ocr_status pending jika ada
    try:
        metadata = load_metadata()
        if saved_filename in metadata:
            metadata[saved_filename]["ocr_status"] = "pending"
            save_metadata(metadata)
    except Exception:
        pass

    background_tasks.add_task(process_ocr_background, latest_path, original_filename, saved_filename)

    return {
        "success": True,
        "message": f"OCR processing started for: {saved_filename}",
        "filename": saved_filename,
    }


# ---------------------------------------------------------------------------
# DELETE Endpoints — Filesystem External (10.4) for Laravel retakeCleanup
# ---------------------------------------------------------------------------
# Laravel retakeCleanup harus DELETE (best effort, external bisa 404 tapi DB tetap 200):
#   DELETE /api/test/ocr/{original_filename}   — hapus image + ocr json + metadata
#   DELETE /api/test/status?filename={fn}      — hapus status/metadata (alias samma logic)
# Implementasi FastAPI (saran planning):
#   @app.delete("/api/test/ocr/{filename}")
#   async def delete_ocr(filename: str, api_key: str = Header(None)):
#       verify api_key == LARAVEL_API_KEY, sanitize basename, delete uploads/filename + ocr/*.json
# Keamanan: sanitasi basename + is_safe_path + verify X-API-Key jika LARAVEL_API_KEY ter-set.
# Rate limit: DELETE 30/min (best effort, cegah abuse) — return 429 dengan Retry-After.
# Return: 200 jika terhapus, 404 jika tidak ketemu, 401 jika api key salah, 429 jika rate limit.
@app.delete("/api/test/ocr/{filename}")
async def delete_ocr(request: Request, filename: str, x_api_key: Optional[str] = Header(None, alias="X-API-Key")):
    """
    DELETE /api/test/ocr/{filename} — Hapus file capture + hasil OCR + metadata.

    Dipanggil oleh Laravel retakeCleanup (best effort). Menerima BAIK saved_filename MAUPUN original_filename.
    Header: X-API-Key harus == LARAVEL_API_KEY (jika LARAVEL_API_KEY ter-set di .env). Jika kosong, allow tanpa auth (dev).
    Filesystem: uploads/{resolved}.png + uploads/ocr/{resolved}.png.json + metadata.json entri + memory ocr_store.
    """
    global latest_filename, latest_source, latest_timestamp, latest_template_id, latest_template_name
    global latest_template_width, latest_template_height, latest_resolution_w, latest_resolution_h, latest_filesize, received_at

    # --- Rate limit DELETE (30/min) ---
    rl = _enforce_delete_rate_limit()
    if rl:
        return rl

    # --- Verify API Key (jika LARAVEL_API_KEY ter-set) ---
    provided_key = _extract_api_key_from_headers(request, x_api_key)
    valid, reason = verify_delete_api_key(provided_key)
    if not valid:
        # Jangan bocorkan expected key, hanya reason generik
        return JSONResponse(
            status_code=401,
            content={"success": False, "message": f"Unauthorized: {reason}", "error": "unauthorized"},
            headers={"Cache-Control": "no-store"},
        )
    if reason == "no_api_key_configured":
        print(f"[DELETE WARN] No LARAVEL_API_KEY configured — allowing DELETE without auth (dev mode) for {filename}")

    # --- Sanitasi & resolve ---
    safe_name = sanitize_filename_for_lookup(filename)
    if not safe_name:
        return JSONResponse(
            status_code=400,
            content={"success": False, "message": f"Invalid filename: {filename}"},
        )

    # --- Delete via storage helper (handles original→saved, is_safe_path) ---
    stats = delete_by_filename(filename)  # internal sanitasi lagi

    # --- Bersihkan memory ocr_store ---
    lookup = stats.get("lookup") or stats.get("resolved_as") or safe_name
    try:
        removed_mem = ocr_store.remove(lookup)
        if removed_mem:
            print(f"[DELETE] Removed from ocr_store memory: {lookup}")
        # Jika lookup beda dengan safe_name, coba juga hapus safe_name
        if lookup != safe_name:
            if ocr_store.remove(safe_name):
                print(f"[DELETE] Removed from ocr_store memory (fallback): {safe_name}")
    except Exception as e:
        logger.warning(f"ocr_store remove failed for {lookup}: {e}")
        stats["errors"].append(f"ocr_store: {e}")

    # --- Update in-memory latest_* jika yang dihapus adalah latest ---
    try:
        if latest_filename and (latest_filename == lookup or latest_filename == safe_name or latest_filename == stats.get("resolved_as")):
            # Reset latest ke None atau ke metadata terbaru tersisa (jika ada)
            remaining_meta = load_metadata()
            if remaining_meta:
                # Ambil yang received_at terbaru
                latest_key, latest_val = sorted(remaining_meta.items(), key=lambda kv: kv[1].get("received_at", ""), reverse=True)[0]
                latest_filename = latest_key
                latest_source = latest_val.get("source", "")
                latest_timestamp = latest_val.get("timestamp", "")
                latest_template_id = latest_val.get("template_id", "")
                latest_template_name = latest_val.get("template_name", "")
                latest_template_width = latest_val.get("template_width", "")
                latest_template_height = latest_val.get("template_height", "")
                latest_resolution_w = latest_val.get("resolution_w", "")
                latest_resolution_h = latest_val.get("resolution_h", "")
                latest_filesize = str(latest_val.get("file_size", ""))
                received_at = latest_val.get("received_at", "")
                print(f"[DELETE] Updated latest_* to remaining file: {latest_key}")
            else:
                latest_filename = None
                latest_source = None
                latest_timestamp = None
                latest_template_id = None
                latest_template_name = None
                latest_template_width = None
                latest_template_height = None
                latest_resolution_w = None
                latest_resolution_h = None
                latest_filesize = None
                received_at = None
                print(f"[DELETE] Cleared latest_* (no files remain) after deleting {lookup}")
    except Exception as e:
        logger.warning(f"Failed to update latest_* after delete {lookup}: {e}")

    # --- Response ---
    if not stats.get("found"):
        return JSONResponse(
            status_code=404,
            content={
                "success": False,
                "message": f"File not found for delete: {filename}",
                "resolved_as": stats.get("resolved_as"),
                "lookup": lookup,
            },
            headers={"Cache-Control": "no-store"},
        )

    return JSONResponse(
        content={
            "success": True,
            "message": f"Deleted {lookup} (requested: {filename})",
            "requested_filename": filename,
            "resolved_as": stats.get("resolved_as"),
            "lookup": lookup,
            "deleted_images": stats.get("deleted_images", 0),
            "deleted_ocr": stats.get("deleted_ocr", 0),
            "deleted_metadata": stats.get("deleted_metadata", 0),
            "deleted_ocr_files": stats.get("deleted_ocr_files", []),
            "errors": stats.get("errors", []),
            "api_key_verified": reason != "no_api_key_configured",
        },
        headers={"Cache-Control": "no-store"},
    )


@app.delete("/api/test/status")
async def delete_status(request: Request, filename: Optional[str] = Query(None, description="Filename to delete (saved or original). Jika tidak diisi -> 400"), x_api_key: Optional[str] = Header(None, alias="X-API-Key")):
    """
    DELETE /api/test/status?filename={fn} — Hapus status/metadata untuk file spesifik.

    Alias untuk DELETE /api/test/ocr/{filename} tapi path sesuai spec Laravel: DELETE /api/test/status?filename=
    Logic sama persis: hapus uploads/* + ocr/*.json + metadata + memory.
    Laravel akan best-effort: jika FastAPI 404, Laravel tetap sukses 200 untuk DB delete lokal.
    """
    global latest_filename, latest_source, latest_timestamp, latest_template_id, latest_template_name
    global latest_template_width, latest_template_height, latest_resolution_w, latest_resolution_h, latest_filesize, received_at

    # --- Rate limit ---
    rl = _enforce_delete_rate_limit()
    if rl:
        return rl

    # --- Verify API Key ---
    provided_key = _extract_api_key_from_headers(request, x_api_key)
    valid, reason = verify_delete_api_key(provided_key)
    if not valid:
        return JSONResponse(
            status_code=401,
            content={"success": False, "message": f"Unauthorized: {reason}", "error": "unauthorized"},
            headers={"Cache-Control": "no-store"},
        )
    if reason == "no_api_key_configured":
        print(f"[DELETE WARN] No LARAVEL_API_KEY configured — allowing DELETE /status without auth for filename={filename}")

    if not filename:
        return JSONResponse(
            status_code=400,
            content={"success": False, "message": "Query param ?filename= is required for DELETE /api/test/status"},
        )

    safe_name = sanitize_filename_for_lookup(filename)
    if not safe_name:
        return JSONResponse(
            status_code=400,
            content={"success": False, "message": f"Invalid filename: {filename}"},
        )

    stats = delete_by_filename(filename)

    # Bersihkan memory ocr_store (sama seperti delete_ocr)
    lookup = stats.get("lookup") or stats.get("resolved_as") or safe_name
    try:
        removed_mem = ocr_store.remove(lookup)
        if removed_mem:
            print(f"[DELETE STATUS] Removed from ocr_store: {lookup}")
        if lookup != safe_name and ocr_store.remove(safe_name):
            print(f"[DELETE STATUS] Removed from ocr_store fallback: {safe_name}")
    except Exception as e:
        logger.warning(f"ocr_store remove failed for {lookup}: {e}")
        stats["errors"].append(f"ocr_store: {e}")

    # Update latest_* jika needed
    try:
        if latest_filename and (latest_filename == lookup or latest_filename == safe_name or latest_filename == stats.get("resolved_as")):
            remaining_meta = load_metadata()
            if remaining_meta:
                latest_key, latest_val = sorted(remaining_meta.items(), key=lambda kv: kv[1].get("received_at", ""), reverse=True)[0]
                latest_filename = latest_key
                latest_source = latest_val.get("source", "")
                latest_timestamp = latest_val.get("timestamp", "")
                latest_template_id = latest_val.get("template_id", "")
                latest_template_name = latest_val.get("template_name", "")
                latest_template_width = latest_val.get("template_width", "")
                latest_template_height = latest_val.get("template_height", "")
                latest_resolution_w = latest_val.get("resolution_w", "")
                latest_resolution_h = latest_val.get("resolution_h", "")
                latest_filesize = str(latest_val.get("file_size", ""))
                received_at = latest_val.get("received_at", "")
                print(f"[DELETE STATUS] Updated latest_* to {latest_key}")
            else:
                latest_filename = None
                latest_source = None
                latest_timestamp = None
                latest_template_id = None
                latest_template_name = None
                latest_template_width = None
                latest_template_height = None
                latest_resolution_w = None
                latest_resolution_h = None
                latest_filesize = None
                received_at = None
                print(f"[DELETE STATUS] Cleared latest_* after deleting {lookup}")
    except Exception as e:
        logger.warning(f"Failed to update latest_* after delete_status {lookup}: {e}")

    if not stats.get("found"):
        return JSONResponse(
            status_code=404,
            content={
                "success": False,
                "message": f"File not found for delete: {filename}",
                "resolved_as": stats.get("resolved_as"),
                "lookup": lookup,
            },
            headers={"Cache-Control": "no-store"},
        )

    return JSONResponse(
        content={
            "success": True,
            "message": f"Deleted status for {lookup} (requested: {filename})",
            "requested_filename": filename,
            "resolved_as": stats.get("resolved_as"),
            "lookup": lookup,
            "deleted_images": stats.get("deleted_images", 0),
            "deleted_ocr": stats.get("deleted_ocr", 0),
            "deleted_metadata": stats.get("deleted_metadata", 0),
            "deleted_ocr_files": stats.get("deleted_ocr_files", []),
            "errors": stats.get("errors", []),
            "api_key_verified": reason != "no_api_key_configured",
        },
        headers={"Cache-Control": "no-store"},
    )


@app.delete("/api/test/status/{filename}")
async def delete_status_path(request: Request, filename: str, x_api_key: Optional[str] = Header(None, alias="X-API-Key")):
    """
    ALIAS path param untuk DELETE /api/test/status/{filename} — support both ?filename= dan /{filename}.
    """
    # Reuse logic dengan membuat Query-like param
    # Panggil delete_status dengan filename path
    return await delete_status(request, filename=filename, x_api_key=x_api_key)


# ---------------------------------------------------------------------------
# Callback Endpoints (Tahap 9-10) — Retry manual & observability
# ---------------------------------------------------------------------------
@app.post("/api/test/ocr/{filename}/callback/retry")
async def retry_callback(filename: str, background_tasks: BackgroundTasks):
    """
    Endpoint manual untuk replay callback ke Laravel untuk file tertentu.

    Berguna jika callback gagal 3x dan masuk dead letter, atau Laravel sempat down.
    Cek uploads/ocr/{filename}.json, lalu kirim ulang via BackgroundTasks (tidak blokir).
    Menerima BAIK saved maupun original filename (resolve via metadata).

    Rate limit tidak berlaku untuk endpoint ini.
    """
    from app.storage import resolve_filename

    safe_name = sanitize_filename_for_lookup(filename)
    if not safe_name:
        return JSONResponse(
            status_code=404,
            content={"success": False, "message": f"OCR result not found for file: {filename}"},
        )

    # Resolve original -> saved
    resolved = resolve_filename(safe_name)
    lookup_name = resolved if resolved else safe_name
    if resolved and resolved != safe_name:
        print(f"[CALLBACK RETRY LOOKUP] {safe_name} -> {resolved}")

    # Validasi allowlist & config
    if not CALLBACK_ENABLED:
        return JSONResponse(
            status_code=400,
            content={"success": False, "message": "Callback disabled (CALLBACK_ENABLED=false)"},
        )
    if not LARAVEL_CALLBACK_URL:
        return JSONResponse(
            status_code=400,
            content={"success": False, "message": "Callback URL not configured (LARAVEL_CALLBACK_URL empty)"},
        )

    result = ocr_store.get_by_filename(lookup_name)
    # Fallback cek safe_name jika resolve gagal
    if result is None and lookup_name != safe_name:
        result = ocr_store.get_by_filename(safe_name)
        if result is not None:
            lookup_name = safe_name
    if result is None:
        return JSONResponse(
            status_code=404,
            content={"success": False, "message": f"OCR result not found for file: {safe_name}", "resolved_as": lookup_name if resolved else None},
        )

    # Jangan retry jika status pending (belum selesai OCR)
    if result.get("status") == "pending":
        return JSONResponse(
            status_code=400,
            content={"success": False, "message": f"OCR still pending for {lookup_name}, try again after OCR completes"},
        )

    # Jalankan di background agar tidak blokir
    def _retry_task():
        _dispatch_callback_sync(result, lookup_name)

    background_tasks.add_task(_retry_task)

    return {
        "success": True,
        "message": f"Callback retry queued for: {lookup_name}",
        "filename": lookup_name,
        "original_filename": result.get("original_filename", ""),
        "resolved_from": safe_name if resolved and resolved != safe_name else None,
        "callback_url": LARAVEL_CALLBACK_URL,
    }


@app.post("/api/test/result/{filename}/callback/retry")
async def retry_callback_alias(filename: str, background_tasks: BackgroundTasks):
    """
    ALIAS untuk POST /api/test/ocr/{filename}/callback/retry — backward compatible.
    """
    return await retry_callback(filename, background_tasks)


@app.get("/api/test/callback/logs")
async def get_callback_logs(limit: int = 20):
    """
    Lihat log callback terakhir (untuk debugging). Query param ?limit=20 (max 100).

    Return JSON Lines terakhir dari uploads/callback_log.jsonl (jika ada).
    """
    limit = max(1, min(limit, 100))
    log_path = CALLBACK_LOG_FILE
    if not log_path.exists():
        return {"success": True, "logs": [], "message": "No callback logs yet"}

    try:
        # Baca terakhir N baris (efisien untuk file kecil <10MB)
        with open(log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        # Ambil N terakhir
        tail = lines[-limit:]
        logs = []
        for line in tail:
            line = line.strip()
            if not line:
                continue
            try:
                logs.append(json.loads(line))
            except json.JSONDecodeError:
                logs.append({"raw": line})
        # Return terbaru di atas
        logs.reverse()
        return {"success": True, "count": len(logs), "logs": logs, "log_file": str(log_path)}
    except Exception as e:
        logger.error(f"Failed to read callback logs: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": f"Failed to read logs: {e}"},
        )


@app.get("/api/test/callback/dead-letter")
async def get_dead_letter(limit: int = 20):
    """
    Lihat dead letter (callback gagal 3x) untuk reconcile manual.
    Query param ?limit=20 (max 100).
    """
    limit = max(1, min(limit, 100))
    dl_path = CALLBACK_DEAD_LETTER_FILE
    if not dl_path.exists():
        return {"success": True, "dead_letters": [], "message": "No dead letters yet"}

    try:
        with open(dl_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        tail = lines[-limit:]
        items = []
        for line in tail:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                items.append({"raw": line})
        items.reverse()
        return {"success": True, "count": len(items), "dead_letters": items, "dead_letter_file": str(dl_path)}
    except Exception as e:
        logger.error(f"Failed to read dead letter: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": f"Failed to read dead letter: {e}"},
        )


# ---------------------------------------------------------------------------
# Health / Root (opsional tapi berguna)
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    tesseract_ok = is_tesseract_available()
    cmd = get_tesseract_cmd()

    # Callback health (tanpa bocorkan secret)
    callback_last_success = None
    callback_last_error = None
    dead_letter_count = 0
    callback_401_count = 0
    ocr_empty_count = 0
    ocr_failed_count = 0
    callback_log_exists = CALLBACK_LOG_FILE.exists()
    try:
        if CALLBACK_LOG_FILE.exists():
            # Baca 1 baris terakhir yang sukses
            with open(CALLBACK_LOG_FILE, "r", encoding="utf-8") as f:
                lines = [l.strip() for l in f if l.strip()]
                # Cari yang status_code 2xx terbaru dan hitung 401
                for line in reversed(lines[-100:]):
                    try:
                        entry = json.loads(line)
                        if entry.get("status_code") == 401:
                            callback_401_count += 1
                        if 200 <= entry.get("status_code", 0) < 300 and callback_last_success is None:
                            callback_last_success = entry.get("ts")
                        elif entry.get("error") and callback_last_error is None:
                            callback_last_error = entry.get("error") or entry.get("ts")
                    except Exception:
                        continue
                # Hitung total 401 di 100 terbaru (atau semua jika <100)
                # callback_401_count sudah di atas
                pass
        if CALLBACK_DEAD_LETTER_FILE.exists():
            with open(CALLBACK_DEAD_LETTER_FILE, "r", encoding="utf-8") as f:
                dead_letter_count = sum(1 for _ in f)
        # Hitung ocr empty/failed dari ocr_store (último 10 + disk)
        try:
            from app.ocr.store import ocr_store as _store
            all_results = _store.list_all()
            for r in all_results:
                if r.get("is_empty") or (not r.get("cleaned_text") and r.get("status") != "success"):
                    ocr_empty_count += 1
                if r.get("status") == "failed":
                    ocr_failed_count += 1
            # Juga cek disk jika store kosong
            if not all_results:
                import glob as _glob
                for jf in OCR_DIR.glob("*.json"):
                    try:
                        with open(jf, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            if data.get("is_empty") or (not data.get("cleaned_text") and data.get("status") != "success"):
                                ocr_empty_count += 1
                            if data.get("status") == "failed":
                                ocr_failed_count += 1
                    except Exception:
                        continue
        except Exception:
            pass
    except Exception:
        pass

    # Rate limit info untuk observability 429 ideal
    from app.config import GET_RATE_LIMIT_REQUESTS, GET_RATE_LIMIT_WINDOW_SECONDS, DELETE_RATE_LIMIT_REQUESTS, DELETE_RATE_LIMIT_WINDOW_SECONDS

    return {
        "status": "ok",
        "tesseract_available": tesseract_ok,
        "tesseract_cmd": cmd,
        "upload_dir": str(UPLOAD_DIR),
        "ocr_dir": str(OCR_DIR),
        "callback_enabled": CALLBACK_ENABLED,
        "callback_url": LARAVEL_CALLBACK_URL,
        "callback_log_exists": callback_log_exists,
        "callback_last_success": callback_last_success,
        "callback_last_error": callback_last_error,
        "dead_letter_count": dead_letter_count,
        "callback_401_count": callback_401_count,
        "ocr_empty_count": ocr_empty_count,
        "ocr_failed_count": ocr_failed_count,
        "preprocessing": "CLAHE+autocontrast+adaptiveThreshold 2-pass (Fix 2026-09-06)",
        "ocr_engine": OCR_ENGINE,
        "ocr_engine_available": (
            __import__("app.ocr.paddle_engine", fromlist=["is_paddle_available"]).is_paddle_available() if OCR_ENGINE == "paddle"
            else (__import__("app.ocr.vitstr_engine", fromlist=["is_vitstr_available"]).is_vitstr_available() if OCR_ENGINE == "vitstr"
            else (__import__("app.ocr.surya_engine", fromlist=["is_surya_available"]).is_surya_available() if OCR_ENGINE == "surya"
            else tesseract_ok))
        ),
        "ocr_none_contract": "cleaned_text='none' bila tak ada teks (text_present=false)",
        # OCR Tuning RapidOCR (2026-09-12 / PADDLEOCR_NATIVE_PLAN §6.6) — aditif.
        "ocr_engine_allowed": ["paddle", "vitstr", "surya", "paddle_native", "tesseract"],
        "ocr_native_available": __import__("app.ocr.paddle_native_engine", fromlist=["is_native_available"]).is_native_available(),
        "ocr_native_note": "paddle_native NONAKTIF: paddlepaddle tanpa wheel Python 3.14 (§6.0 RED). Akurasi via RAPIDOCR_REC_MODEL=medium.",
        "ocr_rapidocr_tuning": __import__("app.ocr.paddle_engine", fromlist=["active_model_label"]).active_model_label(),
        "ocr_optimization": {
            "warmup": __import__("app.config", fromlist=["OCR_WARMUP"]).OCR_WARMUP,
            "max_side_px": __import__("app.config", fromlist=["OCR_MAX_SIDE_PX"]).OCR_MAX_SIDE_PX,
            "det_limit_side": __import__("app.config", fromlist=["OCR_DET_LIMIT_SIDE"]).OCR_DET_LIMIT_SIDE,
            "workers": __import__("app.config", fromlist=["OCR_WORKERS"]).OCR_WORKERS,
            "queue_max": __import__("app.config", fromlist=["OCR_QUEUE_MAX"]).OCR_QUEUE_MAX,
            "queue": __import__("app.ocr.executor", fromlist=["queue_depth_approx"]).queue_depth_approx(),
            "ort_intra_threads": __import__("app.config", fromlist=["OCR_ORT_INTRA_THREADS"]).OCR_ORT_INTRA_THREADS,
            "longpoll_max_seconds": __import__("app.config", fromlist=["OCR_LONGPOLL_MAX_SECONDS"]).OCR_LONGPOLL_MAX_SECONDS,
            "callback_workers": __import__("app.config", fromlist=["CALLBACK_WORKERS"]).CALLBACK_WORKERS,
            "cascade_enabled": __import__("app.config", fromlist=["OCR_CASCADE_ENABLED"]).OCR_CASCADE_ENABLED,
            "use_cls": __import__("app.config", fromlist=["OCR_USE_CLS"]).OCR_USE_CLS,
            "timings_enabled": __import__("app.config", fromlist=["OCR_TIMINGS_ENABLED"]).OCR_TIMINGS_ENABLED,
            # FIX timeout Label 1/2 (2026-10-06): durable polling panjang.
            "max_memory_results": __import__("app.config", fromlist=["OCR_MAX_MEMORY_RESULTS"]).OCR_MAX_MEMORY_RESULTS,
            "pending_fresh_seconds": __import__("app.config", fromlist=["OCR_PENDING_FRESH_SECONDS"]).OCR_PENDING_FRESH_SECONDS,
            "pending_timeout_seconds": 180,
        },
        "rate_limit": {
            # FIX 429 dual-side (2026-09-15): baca dari config agar display selalu akurat.
            "post_status_per_minute": __import__("app.config", fromlist=["POST_RATE_LIMIT_REQUESTS"]).POST_RATE_LIMIT_REQUESTS,
            "post_window_seconds": __import__("app.config", fromlist=["POST_RATE_LIMIT_WINDOW_SECONDS"]).POST_RATE_LIMIT_WINDOW_SECONDS,
            "get_requests_per_window": GET_RATE_LIMIT_REQUESTS,
            "get_window_seconds": GET_RATE_LIMIT_WINDOW_SECONDS,
            "delete_requests_per_window": DELETE_RATE_LIMIT_REQUESTS,
            "delete_window_seconds": DELETE_RATE_LIMIT_WINDOW_SECONDS,
            "retry_after_header": "Retry-After (seconds) + X-RateLimit-* on 429",
            "ideal_polling": "GET /api/test/ocr/{filename} 1 req/poll, 3s×10=20/min, 2 parallel=40/min <60/min",
        },
        "filesystem_external": {
            "uploads": "uploads/label-*.png (POST input)",
            "ocr": "uploads/ocr/label-*.png.json",
            "latest_query": "GET /api/test/ocr/latest?filename=xxx (support original & saved)",
            "delete_ocr": "DELETE /api/test/ocr/{filename} (X-API-Key)",
            "delete_status": "DELETE /api/test/status?filename={fn} (X-API-Key)",
        },
        "check_batch": {
            "endpoint": "POST /api/check/captures (5 images batch, multipart/form-data)",
            "upload_root": str(__import__("app.config", fromlist=["CHECK_UPLOAD_ROOT"]).CHECK_UPLOAD_ROOT) if True else "uploads/check",
            "api_key_set": bool(__import__("app.config", fromlist=["CHECK_API_KEY"]).CHECK_API_KEY),
            "max_file_size": "5MB per file (25MB total)",
            "allowed_mimes": "image/png, image/jpeg",
            "idempotency": "duplicate batch_code returns 200 duplicate:true",
        },
    }


# ---------------------------------------------------------------------------
# Dynamic AI Model Settings (Dashboard Zero-Restart Switching)
# ---------------------------------------------------------------------------
class ModelSettingsPayload(BaseModel):
    ocr_profile: Optional[str] = None
    compare_profile: Optional[str] = None
    updated_by: Optional[str] = "operator"


@app.get("/api/settings/models")
async def get_model_settings():
    """Ambil model OCR & Similarity yang sedang aktif serta ketersediaan library sistem."""
    from app.runtime_config import runtime_config
    return {
        "success": True,
        "data": runtime_config.check_capabilities(),
    }


@app.post("/api/settings/models")
async def update_model_settings(
    payload: ModelSettingsPayload,
    request: Request,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
):
    """Update model OCR & Similarity aktif secara dinamis tanpa restart .env."""
    effective_key = x_api_key or request.headers.get("x-api-key") or request.query_params.get("api_key")
    if LARAVEL_API_KEY and effective_key:
        if effective_key != LARAVEL_API_KEY and effective_key != globals().get("CHECK_API_KEY", ""):
            return JSONResponse(status_code=401, content={"success": False, "message": "Invalid API Key"})

    from app.runtime_config import runtime_config
    caps = runtime_config.check_capabilities()

    if payload.ocr_profile:
        found_ocr = None
        for p in caps["ocr_profiles"]:
            if p["profile"] == payload.ocr_profile:
                found_ocr = p
                break
        if not found_ocr:
            return JSONResponse(status_code=422, content={"success": False, "message": f"Profil OCR '{payload.ocr_profile}' tidak dikenali."})
        if not found_ocr["is_installed"]:
            return JSONResponse(status_code=422, content={"success": False, "message": f"Model '{found_ocr['name']}' belum terpasang: {found_ocr['unavailable_reason']}"})

    if payload.compare_profile:
        found_comp = None
        for p in caps["compare_profiles"]:
            if p["profile"] == payload.compare_profile:
                found_comp = p
                break
        if not found_comp:
            return JSONResponse(status_code=422, content={"success": False, "message": f"Profil Compare '{payload.compare_profile}' tidak dikenali."})
        if not found_comp["is_installed"]:
            return JSONResponse(status_code=422, content={"success": False, "message": f"Engine Compare '{found_comp['name']}' belum siap: {found_comp['unavailable_reason']}"})

    try:
        new_cfg = runtime_config.set_config(
            ocr_profile=payload.ocr_profile,
            compare_profile=payload.compare_profile,
            updated_by=payload.updated_by or "operator",
        )
        return {
            "success": True,
            "message": "Konfigurasi model AI berhasil diperbarui secara instan.",
            "data": new_cfg,
        }
    except Exception as e:
        return JSONResponse(status_code=400, content={"success": False, "message": str(e)})


@app.get("/")
async def root():
    return {"message": "External Service is running"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=9000)
