"""
routers/references.py — Reference Library Label (Scan Sekali, Pilih Ulang).

POST /api/references              -> simpan baru (kompres + OCR background)
GET  /api/references              -> list paginasi + search (untuk Laravel dropdown)
GET  /api/references/stats        -> monitoring (SEBELUM /{ref_id})
GET  /api/references/health       -> DB ping (BARU, SEBELUM /{ref_id})
POST /api/references/from-capture -> promosikan file uploads/ lama (SEBELUM /{ref_id})
GET  /api/references/{ref_id}     -> detail + polling OCR
GET  /api/references/{ref_id}/image -> gambar asli (cache 1 hari)
GET  /api/references/{ref_id}/thumb -> thumbnail (cache 1 hari)
PATCH /api/references/{ref_id}    -> rename
DELETE /api/references/{ref_id}   -> hapus

Kontrak: plans/PLAN_REFERENCE_LIBRARY_FASTAPI.md §3 (TIDAK BERUBAH —
PLAN_MYSQL_BACKED_REFERENCE_STORAGE §4).
Backend: sqlite (file+SQLite, lama) | mysql (stateless DB utama, baru),
dipilih via REF_STORE di app/config.py. Logika handler SAMA, hanya sumber data beda.
"""

import functools
import json
import logging
import os
import re
import tempfile
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, File, Form, Header, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel

try:
    from app.config import (
        CHECK_API_KEY,
        LARAVEL_API_KEY,
        REF_DB_PATH,
        REF_JPG_QUALITY,
        REF_MAX_BYTES_MB,
        REF_MAX_COUNT,
        REF_MAX_IMAGE_DIM,
        REF_ROOT,
        REF_THUMB_DIM,
        REF_THUMB_QUALITY,
    )
except ImportError:
    import os as _os

    CHECK_API_KEY = _os.getenv("CHECK_API_KEY", "")
    LARAVEL_API_KEY = _os.getenv("LARAVEL_API_KEY", "")
    REF_ROOT = Path("uploads/references")
    REF_DB_PATH = REF_ROOT / "references.db"
    REF_MAX_IMAGE_DIM = 1280
    REF_THUMB_DIM = 320
    REF_JPG_QUALITY = 82
    REF_THUMB_QUALITY = 70
    REF_MAX_COUNT = 2000
    REF_MAX_BYTES_MB = 2048.0

try:
    from app.security import (
        check_delete_rate_limit,
        check_get_rate_limit,
        check_reference_rate_limit,
        is_safe_path,
        validate_image_file,
    )
except ImportError:
    def is_safe_path(base: Path, target: Path) -> bool:  # type: ignore
        try:
            return target.resolve().is_relative_to(base.resolve())  # type: ignore
        except AttributeError:
            return str(target.resolve()).startswith(str(base.resolve()))

    def validate_image_file(data: bytes, fn: str):  # type: ignore
        return True, ""

    def check_reference_rate_limit():  # type: ignore
        return True, 0.0, 999

    def check_get_rate_limit():  # type: ignore
        return True, 0.0, 999

    def check_delete_rate_limit():  # type: ignore
        return True, 0.0, 999

# --- Backend selection (PLAN_MYSQL_BACKED_REFERENCE_STORAGE F4) ---
# Dipilih 1x saat import via REF_STORE. Nama fungsi refdb.* SAMA di kedua
# backend (lihat app/db_mysql.py) sehingga logika handler tidak berubah.
try:
    from app.config import REF_STORE as _REF_STORE
except ImportError:
    _REF_STORE = "sqlite"
_USE_MYSQL = str(_REF_STORE or "sqlite").strip().lower() == "mysql"

if _USE_MYSQL:
    from app import db_mysql as refdb
    from app.db_mysql import DBUnavailableError, DuplicateRefError, ping as _db_ping

    count_and_bytes = refdb.count_and_bytes
    delete_ref = refdb.delete_ref
    find_by_sha256 = refdb.find_by_sha256
    get_ref = refdb.get_ref
    init_db = refdb.init_db
    insert_ref = refdb.insert_ref
    list_refs = refdb.list_refs
    touch_used = refdb.touch_used
    update_ref = refdb.update_ref
    get_blob = refdb.get_blob
else:
    from app.references import db as refdb

    count_and_bytes = refdb.count_and_bytes
    delete_ref = refdb.delete_ref
    find_by_sha256 = refdb.find_by_sha256
    get_ref = refdb.get_ref
    init_db = refdb.init_db
    insert_ref = refdb.insert_ref
    list_refs = refdb.list_refs
    touch_used = refdb.touch_used
    update_ref = refdb.update_ref
    get_blob = None  # type: ignore

    class DBUnavailableError(Exception):  # type: ignore
        """Dummy — mode sqlite tidak pernah raise ini."""

    class DuplicateRefError(Exception):  # type: ignore
        """Dummy — mode sqlite menangani race via fallback find_by_sha256."""

        def __init__(self, ref_id: str = ""):
            super().__init__(ref_id)
            self.ref_id = ref_id

    def _db_ping() -> float:  # type: ignore
        return 0.0

from app.references.service import (
    compress_and_save,
    compress_to_bytes,
    make_thumb,
    make_thumb_bytes,
    normalize_text,
    sha256_bytes,
    text_hash,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/references", tags=["references"])

JAKARTA_TZ = timezone(timedelta(hours=7))

REF_ID_RE = re.compile(r"^ref-[A-Za-z0-9\-_]+$")


def _db_503() -> JSONResponse:
    """Respons standar saat MySQL tidak bisa dihubungi (jangan bocorkan traceback)."""
    return JSONResponse(
        status_code=503,
        content={
            "success": False,
            "message": "Reference database unavailable, try again later",
            "error": "db_unavailable",
        },
        headers={"Cache-Control": "no-store", "Retry-After": "5"},
    )


def _handle_db_error(fn):
    """Decorator semua handler: DBUnavailableError -> 503 rapi (mode sqlite tak terpengaruh)."""

    @functools.wraps(fn)
    async def wrapper(*args, **kwargs):
        try:
            return await fn(*args, **kwargs)
        except DBUnavailableError as e:
            logger.warning(f"[REFERENCES] DB unavailable: {e}")
            return _db_503()

    return wrapper


def _now_iso() -> str:
    return datetime.now(JAKARTA_TZ).isoformat()


def _validate_api_key(x_api_key: Optional[str]) -> bool:
    """Validasi X-API-Key timing-safe. Jika key kosong (dev), izinkan."""
    valid_keys = [k for k in (CHECK_API_KEY, LARAVEL_API_KEY) if k]
    if not valid_keys:
        logger.warning("[REFERENCES] CHECK_API_KEY and LARAVEL_API_KEY empty — allowing all (dev mode)")
        return True
    if not x_api_key:
        return False
    import hmac

    for expected in valid_keys:
        if hmac.compare_digest(x_api_key, expected):
            return True
    return False


def _extract_api_key(request: Request, header_key: Optional[str] = None) -> Optional[str]:
    """Ambil API key dari header (case-insensitive) atau ?api_key=."""
    if header_key:
        return header_key
    for k in ("X-API-Key", "x-api-key", "X-Api-Key", "X-API-KEY"):
        v = request.headers.get(k)
        if v:
            return v
    qp = request.query_params.get("api_key")
    if qp:
        return qp
    return None


def _429_post(wait: float) -> JSONResponse:
    # Limiter khusus library: 10 req/menit (bukan 1/menit seperti capture).
    wait_int = int(wait) + 1
    return JSONResponse(
        status_code=429,
        content={
            "success": False,
            "message": f"Rate limit exceeded. Maximum 10 references per minute. Please wait {wait_int} seconds.",
            "error": "too_many_requests",
            "retry_after_seconds": wait_int,
        },
        headers={"Retry-After": str(wait_int)},
    )


def _429_get(wait: float) -> JSONResponse:
    wait_int = int(wait) + 1
    return JSONResponse(
        status_code=429,
        content={
            "success": False,
            "message": f"GET rate limit exceeded. Retry after {wait_int}s.",
            "error": "too_many_requests",
            "retry_after_seconds": wait_int,
        },
        headers={"Retry-After": str(wait_int)},
    )


def _is_empty_cleaned(cleaned_text: str, lines: list) -> bool:
    """Cek kosong dengan pengecualian kontrak 'none' = VALID (bukan kosong)."""
    if isinstance(cleaned_text, str) and cleaned_text.strip().lower() == "none":
        return False
    if not cleaned_text or not cleaned_text.strip():
        return True
    if not lines or not any(isinstance(ln, str) and ln.strip() for ln in lines):
        return True
    return False


def _generate_ref_id() -> str:
    ts = datetime.now(JAKARTA_TZ).strftime("%Y%m%d-%H%M%S")
    return f"ref-{ts}-{uuid.uuid4().hex[:8]}"


def _ocr_reference_background(ref_id: str) -> None:
    """
    Background task: OCR 1 reference lalu update DB.

    Tidak callback ke Laravel, tidak tulis ocr_store/metadata.json.
    Selalu selesai (success atau failed), tidak raise.
    """
    start = time.time()
    _tmp_img: Optional[str] = None  # tempfile khusus mode mysql (auto-delete di finally)
    try:
        row = get_ref(ref_id)
        if row is None:
            print(f"[REFERENCES OCR] abort {ref_id}: not in DB")
            return
        if _USE_MYSQL:
            # Stateless: tulis blob DB ke tempfile sekali pakai untuk engine OCR
            blob = get_blob(ref_id)
            if blob is None or not blob.get("image"):
                update_ref(ref_id, {"ocr_status": "failed", "is_empty": True, "error": "image blob not found"})
                print(f"[REFERENCES OCR] failed {ref_id}: blob missing")
                return
            tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            try:
                tmp.write(blob["image"])
            finally:
                tmp.close()
            _tmp_img = tmp.name
            image_path: object = _tmp_img
        else:
            stored = row.get("stored_filename", "")
            image_path = REF_ROOT / stored
            if not is_safe_path(REF_ROOT, image_path) or not image_path.exists():
                update_ref(ref_id, {"ocr_status": "failed", "is_empty": True, "error": "image file not found"})
                print(f"[REFERENCES OCR] failed {ref_id}: image missing")
                return

        from app.ocr.dispatch import extract_label_text

        try:
            ocr_out = extract_label_text(str(image_path), ref_id)
        except Exception as e_ocr:
            logger.error(f"[REFERENCES OCR] engine failed {ref_id}: {e_ocr}", exc_info=True)
            update_ref(
                ref_id,
                {"ocr_status": "failed", "is_empty": True, "error": str(e_ocr)[:500]},
            )
            return

        engine = ocr_out.get("engine", "tesseract")
        cleaned_text = ocr_out.get("cleaned_text", "") or ""
        lines = ocr_out.get("lines", []) or []
        text_present = ocr_out.get("text_present")

        if engine == "rapidocr" and text_present is False:
            ocr_status = "success"
            is_empty = False
            error_msg = ""
            empty_note = "no_text_detected"
            print(f"[REFERENCES OCR] {ref_id} -> no text ('none'), valid")
        else:
            is_empty = _is_empty_cleaned(cleaned_text, lines)
            if is_empty:
                ocr_status = "failed"
                error_msg = "empty cleaned_text — need retake with better lighting"
                empty_note = "no_text_detected"
                print(f"[REFERENCES OCR] {ref_id} -> failed (empty)")
            else:
                ocr_status = "success"
                error_msg = ""
                empty_note = ""
                print(f"[REFERENCES OCR] {ref_id} -> success {len(cleaned_text)} chars")
                print(f"[REFERENCES OCR RAW] {ref_id} cleaned_text: {repr(cleaned_text)}")

        elapsed = round(time.time() - start, 3)
        norm = normalize_text(cleaned_text)
        th = text_hash(norm)
        # Data update standar (selalu)
        update_data = {
            "cleaned_text": cleaned_text,
            "lines_json": json.dumps(lines, ensure_ascii=False),
            "engine": engine,
            "ocr_status": ocr_status,
            "is_empty": is_empty,
            "error": error_msg,
            "text_hash": th,
            "processing_time_seconds": elapsed,
        }

        # FIX_REFERENCE_OCR_ACUAN_PER_LABEL (2026-10-07):
        # Isi kolom per-sisi berdasarkan metadata side dari row ref.
        # side = "front" / "back" / "" (kosong = legacy single, isi front sebagai default)
        try:
            ref_side = ((row.get("side") or "").strip().lower())
        except Exception:
            ref_side = ""
        if ref_side == "back":
            update_data["cleaned_text_back"] = cleaned_text
            update_data["lines_json_back"] = json.dumps(lines, ensure_ascii=False)
            print(f"[REFERENCES OCR] {ref_id} -> side=back, cleaned_text_back={len(cleaned_text)} chars")
        else:
            # Default: front (termasuk legacy tanpa side)
            update_data["cleaned_text_front"] = cleaned_text
            update_data["lines_json_front"] = json.dumps(lines, ensure_ascii=False)
            print(f"[REFERENCES OCR] {ref_id} -> side={ref_side or 'front(default)'}, cleaned_text_front={len(cleaned_text)} chars")

        update_ref(ref_id, update_data)
        _ = empty_note
    except Exception as e:
        logger.error(f"[REFERENCES OCR] unexpected {ref_id}: {e}", exc_info=True)
        try:
            update_ref(ref_id, {"ocr_status": "failed", "is_empty": True, "error": str(e)[:500]})
        except Exception:
            pass
    finally:
        # Tempfile mode mysql wajib terhapus (sukses maupun gagal) → nol akumulasi disk
        if _tmp_img:
            try:
                os.unlink(_tmp_img)
            except Exception:
                pass


def _check_quota() -> Optional[JSONResponse]:
    """Cek soft quota. Return JSONResponse 413 jika penuh, else None."""
    try:
        count, total_bytes = count_and_bytes()
    except Exception as e:
        logger.warning(f"[REFERENCES] quota check failed: {e}")
        return None
    if count >= REF_MAX_COUNT:
        return JSONResponse(
            status_code=413,
            content={
                "success": False,
                "message": f"Reference quota full ({count}/{REF_MAX_COUNT}). Delete old references first.",
                "error": "quota_exceeded",
            },
        )
    total_mb = total_bytes / (1024 * 1024)
    if total_mb >= REF_MAX_BYTES_MB:
        return JSONResponse(
            status_code=413,
            content={
                "success": False,
                "message": f"Reference storage full ({total_mb:.1f}/{REF_MAX_BYTES_MB} MB). Delete old references first.",
                "error": "quota_exceeded",
            },
        )
    return None


# ---------------------------------------------------------------------------
# POST /api/references — simpan baru
# ---------------------------------------------------------------------------
@router.post("", include_in_schema=True)
@router.post("/", include_in_schema=False)
@_handle_db_error
async def create_reference(
    request: Request,
    background_tasks: BackgroundTasks,
    image: UploadFile = File(..., description="PNG/JPEG <=5MB"),
    name: str = Form(..., description="Nama tampil 3-80 char"),
    template_id: str = Form("", description="Opsional, max 32 char"),
    source: str = Form("", description="Opsional audit"),
    user_id: str = Form("", description="Opsional audit"),
    side: str = Form("", description="Sisi label: 'front' atau 'back'. Kosong = front (default)"),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
):
    effective_key = _extract_api_key(request, x_api_key)
    if not _validate_api_key(effective_key):
        return JSONResponse(status_code=401, content={"success": False, "message": "Invalid API key", "error": "unauthorized"})

    # Limiter khusus library (10/menit) — JANGAN pakai check_rate_limit global
    # 1/menit milik capture, agar push setelah forward tidak 429.
    allowed, wait, _remaining = check_reference_rate_limit()
    if not allowed:
        return _429_post(wait)

    # Baca bytes
    try:
        file_bytes = await image.read()
    except Exception:
        return JSONResponse(status_code=400, content={"success": False, "message": "Failed to read uploaded file", "error": "bad_request"})

    client_filename = image.filename or "upload.png"
    valid, err = validate_image_file(file_bytes, client_filename)
    if not valid:
        return JSONResponse(status_code=400, content={"success": False, "message": err, "error": "validation_failed"})

    clean_name = (name or "").strip()
    if len(clean_name) < 3 or len(clean_name) > 80:
        return JSONResponse(
            status_code=400,
            content={"success": False, "message": "name must be 3..80 characters", "error": "validation_failed"},
        )
    clean_template = (template_id or "").strip()
    if len(clean_template) > 32:
        return JSONResponse(
            status_code=400,
            content={"success": False, "message": "template_id max 32 characters", "error": "validation_failed"},
        )

    # FIX_REFERENCE_OCR_ACUAN_PER_LABEL (2026-10-07): validasi side
    clean_side = (side or "").strip().lower()
    if clean_side and clean_side not in ("front", "back"):
        return JSONResponse(
            status_code=400,
            content={"success": False, "message": "side must be 'front' or 'back' (or empty)", "error": "validation_failed"},
        )

    # Dedup lapis 1 (sebelum simpan — hemat disk + OCR)
    sha = sha256_bytes(file_bytes)
    try:
        existing = find_by_sha256(sha)
    except Exception as e:
        logger.warning(f"[REFERENCES] dedup lookup failed: {e}")
        existing = None
    if existing:
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "duplicate": True,
                "ref_id": existing.get("ref_id"),
                "message": f"Identical image already stored as {existing.get('ref_id')}",
                "existing": {"name": existing.get("name"), "created_at": existing.get("created_at")},
            },
        )

    # Quota
    quota_resp = _check_quota()
    if quota_resp is not None:
        return quota_resp

    # Simpan
    try:
        init_db()
    except Exception as e:
        logger.warning(f"[REFERENCES] init_db failed: {e}")

    ref_id = _generate_ref_id()

    image_payload: dict = {}
    if _USE_MYSQL:
        # Stateless: kompres di MEMORY, tanpa tulis disk sama sekali
        try:
            jpeg_bytes = compress_to_bytes(file_bytes, REF_MAX_IMAGE_DIM, REF_JPG_QUALITY)
            thumb_bytes = make_thumb_bytes(jpeg_bytes, REF_THUMB_DIM, REF_THUMB_QUALITY)
        except ValueError as e:
            return JSONResponse(status_code=400, content={"success": False, "message": str(e), "error": "validation_failed"})
        except Exception as e:
            logger.error(f"[REFERENCES] compress failed {ref_id}: {e}", exc_info=True)
            return JSONResponse(status_code=500, content={"success": False, "message": "Failed to process image", "error": "internal_error"})
        file_size = len(jpeg_bytes)
        image_payload = {"image_bytes": jpeg_bytes, "thumb_bytes": thumb_bytes}
    else:
        stored_filename = f"{ref_id}.jpg"
        thumb_filename = f"{ref_id}_thumb.jpg"
        dest = REF_ROOT / stored_filename
        thumb_dest = REF_ROOT / thumb_filename
        if not is_safe_path(REF_ROOT, dest) or not is_safe_path(REF_ROOT, thumb_dest):
            return JSONResponse(status_code=400, content={"success": False, "message": "Invalid generated path", "error": "internal_error"})

        try:
            file_size = compress_and_save(file_bytes, dest, REF_MAX_IMAGE_DIM, REF_JPG_QUALITY)
            make_thumb(dest, thumb_dest, REF_THUMB_DIM, REF_THUMB_QUALITY)
        except ValueError as e:
            return JSONResponse(status_code=400, content={"success": False, "message": str(e), "error": "validation_failed"})
        except Exception as e:
            logger.error(f"[REFERENCES] save failed {ref_id}: {e}", exc_info=True)
            return JSONResponse(status_code=500, content={"success": False, "message": "Failed to save file", "error": "internal_error"})
        image_payload = {"stored_filename": stored_filename, "thumb_filename": thumb_filename}

    now = _now_iso()
    try:
        insert_ref(
            {
                "ref_id": ref_id,
                "name": clean_name,
                "template_id": clean_template,
                "side": clean_side,
                "source": (source or "").strip(),
                "user_id": (user_id or "").strip(),
                **image_payload,
                "sha256": sha,
                "text_hash": "",
                "cleaned_text": "",
                "lines_json": "[]",
                "engine": "",
                "ocr_status": "pending",
                "is_empty": False,
                "error": "",
                "file_size_bytes": file_size,
                "processing_time_seconds": 0,
                "usage_count": 0,
                "created_at": now,
                "updated_at": now,
                "last_used_at": "",
            }
        )
    except DuplicateRefError as e:
        # Race duplikat (UNIQUE sha256, mode mysql) → return existing seperti dedup lapis 1
        dup = None
        try:
            dup = find_by_sha256(sha) if not e.ref_id else get_ref(e.ref_id)
        except Exception:
            pass
        if dup:
            return JSONResponse(
                status_code=200,
                content={
                    "success": True,
                    "duplicate": True,
                    "ref_id": dup.get("ref_id"),
                    "message": f"Identical image already stored as {dup.get('ref_id')}",
                    "existing": {"name": dup.get("name"), "created_at": dup.get("created_at")},
                },
            )
        return JSONResponse(status_code=500, content={"success": False, "message": "Failed to save reference", "error": "internal_error"})
    except Exception as e:
        # Bersihkan file jika DB gagal (mode sqlite; mode mysql tak punya file)
        logger.error(f"[REFERENCES] insert failed {ref_id}: {e}", exc_info=True)
        if not _USE_MYSQL:
            try:
                dest.unlink(missing_ok=True)
                thumb_dest.unlink(missing_ok=True)
            except Exception:
                pass
        # Jika race duplikat (UNIQUE sha256), coba return existing
        try:
            dup = find_by_sha256(sha)
            if dup:
                return JSONResponse(
                    status_code=200,
                    content={
                        "success": True,
                        "duplicate": True,
                        "ref_id": dup.get("ref_id"),
                        "message": f"Identical image already stored as {dup.get('ref_id')}",
                        "existing": {"name": dup.get("name"), "created_at": dup.get("created_at")},
                    },
                )
        except Exception:
            pass
        return JSONResponse(status_code=500, content={"success": False, "message": "Failed to save reference", "error": "internal_error"})

    print(f"[REFERENCES] saved {ref_id} name={clean_name!r} size={file_size}B (OCR queued)")
    background_tasks.add_task(_ocr_reference_background, ref_id)

    return JSONResponse(
        status_code=201,
        content={
            "success": True,
            "ref_id": ref_id,
            "name": clean_name,
            "side": clean_side,
            "ocr_status": "pending",
            "duplicate": False,
            "poll_url": f"/api/references/{ref_id}",
            "thumb_url": f"/api/references/{ref_id}/thumb",
            "image_url": f"/api/references/{ref_id}/image",
        },
        headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0", "Pragma": "no-cache", "Expires": "0"},
    )


# ---------------------------------------------------------------------------
# GET /api/references/stats — monitoring (SEBELUM /{ref_id})
# ---------------------------------------------------------------------------
@router.get("/stats")
@_handle_db_error
async def reference_stats(request: Request, x_api_key: Optional[str] = Header(None, alias="X-API-Key")):
    effective_key = _extract_api_key(request, x_api_key)
    if not _validate_api_key(effective_key):
        return JSONResponse(status_code=401, content={"success": False, "message": "Invalid API key", "error": "unauthorized"})
    rl = check_get_rate_limit()
    allowed, wait = rl[0], rl[1]
    if not allowed:
        return _429_get(wait)
    try:
        count, total_bytes = count_and_bytes()
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "message": f"Stats failed: {e}", "error": "internal_error"})
    total_mb = round(total_bytes / (1024 * 1024), 2)
    if _USE_MYSQL:
        db_rel = "mysql:label_inspection"
    else:
        try:
            db_rel = str(REF_DB_PATH.relative_to(Path.cwd())).replace("\\", "/")
        except ValueError:
            db_rel = "uploads/references/references.db"
    return JSONResponse(
        status_code=200,
        content={"success": True, "count": count, "total_bytes": total_bytes, "total_mb": total_mb, "db_path": db_rel},
        headers={"Cache-Control": "no-store"},
    )


# ---------------------------------------------------------------------------
# GET /api/references/health — DB ping (BARU, SEBELUM /{ref_id})
# ---------------------------------------------------------------------------
@router.get("/health")
async def reference_health():
    # Tanpa API key + tanpa rate limit: untuk monitoring/load-balancer.
    if not _USE_MYSQL:
        return JSONResponse(status_code=200, content={"success": True, "db": "sqlite", "store": "sqlite"})
    try:
        latency_ms = _db_ping()
    except DBUnavailableError as e:
        logger.warning(f"[REFERENCES] health DB down: {e}")
        return JSONResponse(
            status_code=503,
            content={"success": False, "db": "down", "error": "db_unavailable"},
            headers={"Cache-Control": "no-store"},
        )
    return JSONResponse(
        status_code=200,
        content={"success": True, "db": "ok", "store": "mysql", "latency_ms": latency_ms},
        headers={"Cache-Control": "no-store"},
    )


class FromCaptureBody(BaseModel):
    filename: str
    name: str
    template_id: str = ""
    source: str = ""
    user_id: str = ""
    side: str = ""


# ---------------------------------------------------------------------------
# POST /api/references/from-capture — promosikan file uploads/ lama
# ---------------------------------------------------------------------------
@router.post("/from-capture")
@_handle_db_error
async def create_from_capture(
    request: Request,
    body: FromCaptureBody,
    background_tasks: BackgroundTasks,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
):
    effective_key = _extract_api_key(request, x_api_key)
    if not _validate_api_key(effective_key):
        return JSONResponse(status_code=401, content={"success": False, "message": "Invalid API key", "error": "unauthorized"})

    # Limiter khusus library (10/menit) — sama seperti POST / (lihat komentar di atas).
    allowed, wait, _remaining = check_reference_rate_limit()
    if not allowed:
        return _429_post(wait)

    clean_name = (body.name or "").strip()
    if len(clean_name) < 3 or len(clean_name) > 80:
        return JSONResponse(
            status_code=400,
            content={"success": False, "message": "name must be 3..80 characters", "error": "validation_failed"},
        )
    clean_template = (body.template_id or "").strip()
    if len(clean_template) > 32:
        return JSONResponse(
            status_code=400,
            content={"success": False, "message": "template_id max 32 characters", "error": "validation_failed"},
        )
    # FIX_REFERENCE_OCR_ACUAN_PER_LABEL (2026-10-07): validasi side (aditif, default "")
    clean_side_fc = (getattr(body, "side", "") or "").strip().lower()
    if clean_side_fc and clean_side_fc not in ("front", "back"):
        return JSONResponse(
            status_code=400,
            content={"success": False, "message": "side must be 'front' or 'back' (or empty)", "error": "validation_failed"},
        )
    if not body.filename or not body.filename.strip():
        return JSONResponse(
            status_code=400, content={"success": False, "message": "filename is required", "error": "validation_failed"}
        )

    # Resolve uploads/ file (support original maupun saved)
    try:
        from app.config import OCR_DIR, UPLOAD_DIR
        from app.security import sanitize_filename_for_lookup
        from app.storage import resolve_filename
    except ImportError as e:
        return JSONResponse(status_code=500, content={"success": False, "message": f"Storage module missing: {e}", "error": "internal_error"})

    safe_req = sanitize_filename_for_lookup(body.filename.strip())
    resolved = resolve_filename(safe_req)
    lookup = resolved if resolved else safe_req
    src_path = UPLOAD_DIR / lookup
    if not is_safe_path(UPLOAD_DIR, src_path) or not src_path.exists() or not src_path.is_file():
        return JSONResponse(
            status_code=404,
            content={"success": False, "message": f"Capture file not found: {body.filename}", "error": "not_found"},
        )
    try:
        file_bytes = src_path.read_bytes()
    except Exception:
        return JSONResponse(status_code=500, content={"success": False, "message": "Failed to read capture file", "error": "internal_error"})

    valid, err = validate_image_file(file_bytes, lookup)
    if not valid:
        return JSONResponse(status_code=400, content={"success": False, "message": err, "error": "validation_failed"})

    sha = sha256_bytes(file_bytes)
    try:
        existing = find_by_sha256(sha)
    except Exception:
        existing = None
    if existing:
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "duplicate": True,
                "ref_id": existing.get("ref_id"),
                "message": f"Identical image already stored as {existing.get('ref_id')}",
                "existing": {"name": existing.get("name"), "created_at": existing.get("created_at")},
            },
        )

    quota_resp = _check_quota()
    if quota_resp is not None:
        return quota_resp

    try:
        init_db()
    except Exception as e:
        logger.warning(f"[REFERENCES] init_db failed: {e}")

    ref_id = _generate_ref_id()
    image_payload_fc: dict = {}
    if _USE_MYSQL:
        # Stateless: kompres di MEMORY (Fase 2: src dari tempfile, bukan uploads/)
        try:
            jpeg_fc = compress_to_bytes(file_bytes, REF_MAX_IMAGE_DIM, REF_JPG_QUALITY)
            thumb_fc = make_thumb_bytes(jpeg_fc, REF_THUMB_DIM, REF_THUMB_QUALITY)
        except ValueError as e:
            return JSONResponse(status_code=400, content={"success": False, "message": str(e), "error": "validation_failed"})
        except Exception as e:
            logger.error(f"[REFERENCES] from-capture compress failed {ref_id}: {e}", exc_info=True)
            return JSONResponse(status_code=500, content={"success": False, "message": "Failed to process image", "error": "internal_error"})
        file_size = len(jpeg_fc)
        image_payload_fc = {"image_bytes": jpeg_fc, "thumb_bytes": thumb_fc}
    else:
        stored_filename = f"{ref_id}.jpg"
        thumb_filename = f"{ref_id}_thumb.jpg"
        dest = REF_ROOT / stored_filename
        thumb_dest = REF_ROOT / thumb_filename
        try:
            file_size = compress_and_save(file_bytes, dest, REF_MAX_IMAGE_DIM, REF_JPG_QUALITY)
            make_thumb(dest, thumb_dest, REF_THUMB_DIM, REF_THUMB_QUALITY)
        except Exception as e:
            logger.error(f"[REFERENCES] from-capture save failed {ref_id}: {e}", exc_info=True)
            return JSONResponse(status_code=500, content={"success": False, "message": "Failed to save file", "error": "internal_error"})
        image_payload_fc = {"stored_filename": stored_filename, "thumb_filename": thumb_filename}

    # Optimasi: copy hasil OCR lama jika sudah success (hemat ~4 detik)
    prefill = {"ocr_status": "pending", "cleaned_text": "", "lines_json": "[]", "engine": "", "is_empty": False, "error": "", "text_hash": "", "processing_time_seconds": 0}
    need_background = True
    try:
        ocr_json = OCR_DIR / f"{lookup}.json"
        if ocr_json.exists() and is_safe_path(OCR_DIR, ocr_json):
            with open(ocr_json, "r", encoding="utf-8") as f:
                old = json.load(f)
            if old.get("status") == "success" and (old.get("cleaned_text") or "").strip():
                ct = old.get("cleaned_text", "")
                lines = old.get("lines", []) or []
                lines_json_str = json.dumps(lines, ensure_ascii=False)
                prefill = {
                    "ocr_status": "success",
                    "cleaned_text": ct,
                    "lines_json": lines_json_str,
                    "engine": old.get("engine", ""),
                    "is_empty": False,
                    "error": "",
                    "text_hash": text_hash(normalize_text(ct)),
                    "processing_time_seconds": float(old.get("processing_time_seconds", 0) or 0),
                }
                # FIX_REFERENCE_OCR_ACUAN_PER_LABEL (2026-10-07):
                # Prefill reuse juga isi kolom per-sisi agar GET langsung benar
                # tanpa menunggu background OCR.
                if clean_side_fc == "back":
                    prefill["cleaned_text_back"] = ct
                    prefill["lines_json_back"] = lines_json_str
                else:
                    prefill["cleaned_text_front"] = ct
                    prefill["lines_json_front"] = lines_json_str
                need_background = False
                print(f"[REFERENCES] from-capture {ref_id}: reused OCR from {lookup} ({len(lines)} lines)")
    except Exception as e:
        logger.warning(f"[REFERENCES] reuse OCR failed for {lookup}: {e}")

    now = _now_iso()
    try:
        insert_ref(
            {
                "ref_id": ref_id,
                "name": clean_name,
                "template_id": clean_template,
                "side": clean_side_fc,
                "source": (body.source or "").strip(),
                "user_id": str(body.user_id or "").strip(),
                **image_payload_fc,
                "sha256": sha,
                **prefill,
                "file_size_bytes": file_size,
                "usage_count": 0,
                "created_at": now,
                "updated_at": now,
                "last_used_at": "",
            }
        )
    except DuplicateRefError as e:
        dup_fc = None
        try:
            dup_fc = find_by_sha256(sha) if not e.ref_id else get_ref(e.ref_id)
        except Exception:
            pass
        if dup_fc:
            return JSONResponse(
                status_code=200,
                content={
                    "success": True,
                    "duplicate": True,
                    "ref_id": dup_fc.get("ref_id"),
                    "message": f"Identical image already stored as {dup_fc.get('ref_id')}",
                    "existing": {"name": dup_fc.get("name"), "created_at": dup_fc.get("created_at")},
                },
            )
        return JSONResponse(status_code=500, content={"success": False, "message": "Failed to save reference", "error": "internal_error"})
    except Exception as e:
        logger.error(f"[REFERENCES] insert failed {ref_id}: {e}", exc_info=True)
        if not _USE_MYSQL:
            try:
                dest.unlink(missing_ok=True)
                thumb_dest.unlink(missing_ok=True)
            except Exception:
                pass
        return JSONResponse(status_code=500, content={"success": False, "message": "Failed to save reference", "error": "internal_error"})

    if need_background:
        background_tasks.add_task(_ocr_reference_background, ref_id)
        ocr_status_out = "pending"
    else:
        ocr_status_out = "success"

    print(f"[REFERENCES] from-capture {ref_id} <- {lookup} ocr={ocr_status_out}")
    return JSONResponse(
        status_code=201,
        content={
            "success": True,
            "ref_id": ref_id,
            "name": clean_name,
            "side": clean_side_fc,
            "ocr_status": ocr_status_out,
            "duplicate": False,
            "source_capture": lookup,
            "poll_url": f"/api/references/{ref_id}",
            "thumb_url": f"/api/references/{ref_id}/thumb",
            "image_url": f"/api/references/{ref_id}/image",
        },
    )


# ---------------------------------------------------------------------------
# GET /api/references — list paginasi + search
# ---------------------------------------------------------------------------
@router.get("", include_in_schema=True)
@router.get("/", include_in_schema=False)
@_handle_db_error
async def list_references(
    request: Request,
    q: str = Query("", description="Search di name + cleaned_text"),
    template_id: str = Query("", description="Filter exact template"),
    ocr_status: str = Query("", description="pending|success|failed"),
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=200, description="Max 50 (di-clamp server)"),
    sort: str = Query("recent_used", description="recent_used|newest|name"),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
):
    effective_key = _extract_api_key(request, x_api_key)
    if not _validate_api_key(effective_key):
        return JSONResponse(status_code=401, content={"success": False, "message": "Invalid API key", "error": "unauthorized"})

    rl = check_get_rate_limit()
    if not rl[0]:
        return _429_get(rl[1])

    if sort not in ("recent_used", "newest", "name"):
        sort = "recent_used"
    if ocr_status not in ("", "pending", "success", "failed"):
        ocr_status = ""

    try:
        items, total = list_refs(q=q, template_id=template_id, ocr_status=ocr_status, sort=sort, page=page, per_page=per_page)
    except Exception as e:
        logger.error(f"[REFERENCES] list failed: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"success": False, "message": "List failed", "error": "internal_error"})

    eff_per_page = max(1, min(int(per_page or 20), 50))
    total_pages = (total + eff_per_page - 1) // eff_per_page if total > 0 else 0
    for it in items:
        it["thumb_url"] = f"/api/references/{it['ref_id']}/thumb"

    return JSONResponse(
        status_code=200,
        content={"success": True, "page": int(page or 1), "per_page": eff_per_page, "total": total, "total_pages": total_pages, "items": items},
        headers={"Cache-Control": "no-store"},
    )


# ---------------------------------------------------------------------------
# POST /api/references/detect-regions — Opsi C Auto-Detect & Deskew Label (Front & Back)
# ---------------------------------------------------------------------------
@router.post("/detect-regions")
@_handle_db_error
async def detect_reference_regions(
    request: Request,
    image: UploadFile = File(...),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
):
    effective_key = _extract_api_key(request, x_api_key)
    if not _validate_api_key(effective_key):
        return JSONResponse(status_code=401, content={"success": False, "message": "Invalid API key", "error": "unauthorized"})

    try:
        from app.vision.label_detector import detect_labels, image_to_base64_data_url
        import cv2
        import numpy as np

        contents = await image.read()
        if not contents:
            return JSONResponse(status_code=422, content={"success": False, "message": "File gambar kosong", "error": "empty_file"})

        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return JSONResponse(status_code=422, content={"success": False, "message": "Format gambar tidak valid / rusak", "error": "invalid_image"})

        res = detect_labels(img, fallback_on_miss=True)

        annotated_b64 = image_to_base64_data_url(res["annotated_image"], ext="png")

        slots_payload = []
        for s in res["slots"]:
            crop_b64 = image_to_base64_data_url(s["cropped_image"], ext="png")
            ch, cw = s["cropped_image"].shape[:2]
            slots_payload.append({
                "slot": s["slot"],
                "label_name": s["label_name"],
                "is_fallback": s.get("is_fallback", False),
                "crop_data_url": crop_b64,
                "w": cw,
                "h": ch,
                "rect": s["rect"],
                "box_points": s["box_points"],
            })

        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "detected_count": len(slots_payload),
                "annotated_preview": annotated_b64,
                "slots": slots_payload,
                "elapsed_ms": res["elapsed_ms"],
                "image_width": int(img.shape[1]),
                "image_height": int(img.shape[0]),
            }
        )
    except Exception as e:
        logger.error(f"[REFERENCES] detect-regions failed: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"success": False, "message": f"Deteksi gagal: {str(e)}", "error": "detection_failed"})


# ---------------------------------------------------------------------------
# GET /api/references/{ref_id} — detail
# ---------------------------------------------------------------------------
@router.get("/{ref_id}")
@_handle_db_error
async def get_reference_detail(
    ref_id: str,
    request: Request,
    track_use: Optional[int] = Query(None, description="1 = naikkan usage_count"),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
):
    effective_key = _extract_api_key(request, x_api_key)
    if not _validate_api_key(effective_key):
        return JSONResponse(status_code=401, content={"success": False, "message": "Invalid API key", "error": "unauthorized"})

    rl = check_get_rate_limit()
    if not rl[0]:
        return _429_get(rl[1])

    if not REF_ID_RE.match(ref_id or ""):
        return JSONResponse(
            status_code=422, content={"success": False, "message": "Invalid ref_id format", "error": "validation_failed"}
        )

    row = get_ref(ref_id)
    if row is None:
        return JSONResponse(status_code=404, content={"success": False, "message": "Reference not found", "error": "not_found"})

    # Track use (best-effort)
    want_track = (track_use == 1) or (request.headers.get("X-Track-Use") == "1")
    if want_track:
        touch_used(ref_id)
        row["usage_count"] = int(row.get("usage_count", 0) or 0) + 1

    if (row.get("ocr_status") or "pending") == "pending":
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "ref_id": row["ref_id"],
                "name": row["name"],
                "template_id": row.get("template_id", ""),
                "side": row.get("side", "") or "",
                "ocr_status": "pending",
                "message": "OCR still processing, retry in 2 seconds",
                "retry_after_seconds": 2,
                # FIX timeout Label 1/2 (2026-10-06): samakan kontrak retryable
                # agar frontend Reference lanjut polling (Cek Lagi) tanpa buat
                # reference baru. Aditif.
                "retryable": True,
                "poll_url": f"/api/references/{ref_id}",
                "pending_timeout_seconds": 180,
                "image_url": f"/api/references/{ref_id}/image",
                "thumb_url": f"/api/references/{ref_id}/thumb",
                "created_at": row.get("created_at", ""),
                "usage_count": int(row.get("usage_count", 0) or 0),
            },
            headers={"Cache-Control": "no-store", "Retry-After": "2"},
        )

    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "ref_id": row["ref_id"],
            "name": row["name"],
            "template_id": row.get("template_id", ""),
            "side": row.get("side", "") or "",
            "source": row.get("source", ""),
            "ocr_status": row.get("ocr_status", ""),
            "engine": row.get("engine", ""),
            "cleaned_text": row.get("cleaned_text", ""),
            "lines": row.get("lines", []),
            "cleaned_text_front": row.get("cleaned_text_front") or None,
            "cleaned_text_back": row.get("cleaned_text_back") or None,
            "lines_front": row.get("lines_front", []),
            "lines_back": row.get("lines_back", []),
            "has_back": bool(row.get("has_back", False)),
            "is_empty": bool(row.get("is_empty", False)),
            "error": row.get("error", "") or None,
            "image_url": f"/api/references/{ref_id}/image",
            "thumb_url": f"/api/references/{ref_id}/thumb",
            "file_size_bytes": int(row.get("file_size_bytes", 0) or 0),
            "processing_time_seconds": float(row.get("processing_time_seconds", 0) or 0),
            "usage_count": int(row.get("usage_count", 0) or 0),
            "created_at": row.get("created_at", ""),
            "updated_at": row.get("updated_at", ""),
            "last_used_at": row.get("last_used_at", ""),
        },
        headers={"Cache-Control": "no-store"},
    )


def _serve_file(ref_id: str, kind: str) -> Response | JSONResponse:
    """Helper serve image/thumb dengan cache header + ETag (file atau blob DB)."""
    row = get_ref(ref_id)
    if row is None:
        return JSONResponse(status_code=404, content={"success": False, "message": "Reference not found", "error": "not_found"})
    etag = (row.get("sha256") or "")[:12]
    headers = {"Cache-Control": "public, max-age=86400", "ETag": f'W/"{etag}"'}
    if _USE_MYSQL:
        # Stateless: stream bytes dari ref_blobs (1 SELECT by PK)
        blob = get_blob(ref_id)
        if blob is None or not blob.get(kind):
            return JSONResponse(status_code=404, content={"success": False, "message": "File not found", "error": "not_found"})
        return Response(content=blob[kind], media_type="image/jpeg", headers=headers)
    filename = row.get("stored_filename" if kind == "image" else "thumb_filename", "")
    if not filename:
        return JSONResponse(status_code=404, content={"success": False, "message": "File not found", "error": "not_found"})
    fpath = REF_ROOT / filename
    if not is_safe_path(REF_ROOT, fpath) or not fpath.exists() or not fpath.is_file():
        return JSONResponse(status_code=404, content={"success": False, "message": "File not found", "error": "not_found"})
    return FileResponse(
        path=str(fpath),
        media_type="image/jpeg",
        filename=filename,
        headers=headers,
    )


# ---------------------------------------------------------------------------
# GET image / thumb (public-read agar <img> Laravel bisa langsung pakai;
# list/detail tetap wajib API key. Rate limit tetap enforced.)
# ---------------------------------------------------------------------------
@router.get("/{ref_id}/image")
@_handle_db_error
async def get_reference_image(ref_id: str, request: Request):
    rl = check_get_rate_limit()
    if not rl[0]:
        return _429_get(rl[1])
    if not REF_ID_RE.match(ref_id or ""):
        return JSONResponse(status_code=422, content={"success": False, "message": "Invalid ref_id format", "error": "validation_failed"})
    return _serve_file(ref_id, "image")


@router.get("/{ref_id}/thumb")
@_handle_db_error
async def get_reference_thumb(ref_id: str, request: Request):
    rl = check_get_rate_limit()
    if not rl[0]:
        return _429_get(rl[1])
    if not REF_ID_RE.match(ref_id or ""):
        return JSONResponse(status_code=422, content={"success": False, "message": "Invalid ref_id format", "error": "validation_failed"})
    return _serve_file(ref_id, "thumb")


class PatchBody(BaseModel):
    name: Optional[str] = None
    template_id: Optional[str] = None


# ---------------------------------------------------------------------------
# PATCH /api/references/{ref_id} — rename
# ---------------------------------------------------------------------------
@router.patch("/{ref_id}")
@_handle_db_error
async def patch_reference(ref_id: str, body: PatchBody, request: Request, x_api_key: Optional[str] = Header(None, alias="X-API-Key")):
    effective_key = _extract_api_key(request, x_api_key)
    if not _validate_api_key(effective_key):
        return JSONResponse(status_code=401, content={"success": False, "message": "Invalid API key", "error": "unauthorized"})

    rl = check_delete_rate_limit()
    if not rl[0]:
        return _429_get(rl[1])

    if not REF_ID_RE.match(ref_id or ""):
        return JSONResponse(status_code=422, content={"success": False, "message": "Invalid ref_id format", "error": "validation_failed"})

    fields: dict = {}
    if body.name is not None:
        clean = body.name.strip()
        if len(clean) < 3 or len(clean) > 80:
            return JSONResponse(
                status_code=400, content={"success": False, "message": "name must be 3..80 characters", "error": "validation_failed"}
            )
        fields["name"] = clean
    if body.template_id is not None:
        clean_t = body.template_id.strip()
        if len(clean_t) > 32:
            return JSONResponse(
                status_code=400, content={"success": False, "message": "template_id max 32 characters", "error": "validation_failed"}
            )
        fields["template_id"] = clean_t
    if not fields:
        return JSONResponse(
            status_code=400, content={"success": False, "message": "Nothing to update (name/template_id required)", "error": "validation_failed"}
        )

    row = get_ref(ref_id)
    if row is None:
        return JSONResponse(status_code=404, content={"success": False, "message": "Reference not found", "error": "not_found"})

    ok = update_ref(ref_id, fields)
    if not ok:
        return JSONResponse(status_code=500, content={"success": False, "message": "Update failed", "error": "internal_error"})
    fresh = get_ref(ref_id) or {}
    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "ref_id": ref_id,
            "name": fresh.get("name", ""),
            "template_id": fresh.get("template_id", ""),
            "updated_at": fresh.get("updated_at", ""),
        },
    )


# ---------------------------------------------------------------------------
# DELETE /api/references/{ref_id} — hapus
# ---------------------------------------------------------------------------
@router.delete("/{ref_id}")
@_handle_db_error
async def delete_reference(ref_id: str, request: Request, x_api_key: Optional[str] = Header(None, alias="X-API-Key")):
    effective_key = _extract_api_key(request, x_api_key)
    if not _validate_api_key(effective_key):
        return JSONResponse(status_code=401, content={"success": False, "message": "Invalid API key", "error": "unauthorized"})

    rl = check_delete_rate_limit()
    if not rl[0]:
        return _429_get(rl[1])

    if not REF_ID_RE.match(ref_id or ""):
        return JSONResponse(status_code=422, content={"success": False, "message": "Invalid ref_id format", "error": "validation_failed"})

    old = delete_ref(ref_id)
    if old is None:
        return JSONResponse(status_code=404, content={"success": False, "message": "Reference not found", "error": "not_found"})

    if _USE_MYSQL:
        # Blob terhapus via ON DELETE CASCADE → 2 artefak logis (image + thumb)
        deleted_files = 2
    else:
        deleted_files = 0
        for key in ("stored_filename", "thumb_filename"):
            fn = old.get(key, "")
            if not fn:
                continue
            p = REF_ROOT / fn
            try:
                if is_safe_path(REF_ROOT, p) and p.exists() and p.is_file():
                    p.unlink()
                    deleted_files += 1
            except Exception as e:
                logger.warning(f"[REFERENCES] delete file failed {p}: {e}")

    print(f"[REFERENCES] deleted {ref_id} files={deleted_files}")
    return JSONResponse(status_code=200, content={"success": True, "ref_id": ref_id, "deleted_files": deleted_files})
