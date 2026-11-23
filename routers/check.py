"""
routers/check.py — Endpoint FastAPI untuk menerima 4-5 gambar Check Batch dari Laravel.

POST /api/check/captures
- Validasi X-API-Key == CHECK_API_KEY (401 jika salah)
- Validasi batch_code regex ^[A-Za-z0-9\\-_]+$ (422 jika salah)
- Validasi len(images) in (4,5) — sinkron Laravel between:4,5 (422)
- Validasi slot_numbers unique 1..5, count == len(images) (422)
- Validasi content_type image/png/jpeg + ext whitelist + magic bytes + size 5MB
- Sanitasi batch_code basename + anti ../
- Buat folder uploads/check/{batch_code}/ (mkdir -p)
- Idempotency: jika folder sudah ada N file valid, return 200 duplicate (bukan error)
- Simpan per slot: uploads/check/{batch_code}/slot{slot}-...png
- Return JSON {success:true, batch_code, saved_count, paths, message}

Kontrak: Harus sinkron dengan Laravel .env CHECK_EXTERNAL_URL, OCR_CALLBACK_API_KEY
Laravel: CheckCaptureController (between:4,5), CheckExternalClient + CheckOcrClient (in [4,5])
Spec lengkap: plans/PLAN_CHECK_5_IMAGES_FASTAPI_EXTERNAL_SERVICE.md
"""

from fastapi import APIRouter, UploadFile, File, Form, Header, Request, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from typing import List, Optional
import concurrent.futures
import os
import re
import logging
import time
import json
import shutil
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Import config — fallback jika belum ada
try:
    from app.config import (
        CHECK_UPLOAD_ROOT,
        CHECK_API_KEY,
        LARAVEL_API_KEY,
        CHECK_MAX_FILE_SIZE,
        CHECK_ALLOWED_MIMES,
        CHECK_OCR_ENABLED,
        CHECK_OCR_MAX_WORKERS,
        CHECK_OCR_TIMEOUT_SECONDS,
        CHECK_PROCESS_RATE_LIMIT_REQUESTS,
        CHECK_PROCESS_RATE_LIMIT_WINDOW_SECONDS,
        CHECK_OCR_CALLBACK_ENABLED,
        CHECK_OCR_CALLBACK_URL,
        CHECK_OCR_PROGRESSIVE_CALLBACK,
    )
except ImportError:
    CHECK_UPLOAD_ROOT = Path("uploads/check")
    CHECK_API_KEY = os.getenv("CHECK_API_KEY", "")
    LARAVEL_API_KEY = os.getenv("LARAVEL_API_KEY", "")
    CHECK_MAX_FILE_SIZE = 5 * 1024 * 1024
    CHECK_ALLOWED_MIMES = {"image/png", "image/jpeg"}
    CHECK_OCR_ENABLED = os.getenv("CHECK_OCR_ENABLED", "true").lower() == "true"
    CHECK_OCR_MAX_WORKERS = int(os.getenv("CHECK_OCR_MAX_WORKERS", "1"))
    CHECK_OCR_TIMEOUT_SECONDS = float(os.getenv("CHECK_OCR_TIMEOUT_SECONDS", "15.0"))
    CHECK_PROCESS_RATE_LIMIT_REQUESTS = int(os.getenv("CHECK_PROCESS_RATE_LIMIT_REQUESTS", "10"))
    CHECK_PROCESS_RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("CHECK_PROCESS_RATE_LIMIT_WINDOW_SECONDS", "60"))
    CHECK_OCR_CALLBACK_ENABLED = os.getenv("CHECK_OCR_CALLBACK_ENABLED", "false").lower() == "true"
    CHECK_OCR_CALLBACK_URL = os.getenv("CHECK_OCR_CALLBACK_URL", "http://127.0.0.1:8000/api/check/ocr/callback")
    CHECK_OCR_PROGRESSIVE_CALLBACK = os.getenv("CHECK_OCR_PROGRESSIVE_CALLBACK", "true").lower() == "true"

# Timezone Jakarta (untuk processed_at)
JAKARTA_TZ = timezone(timedelta(hours=7))

try:
    from app.security import is_safe_path, sanitize_filename_for_lookup
except ImportError:
    def is_safe_path(base: Path, target: Path) -> bool:  # type: ignore
        try:
            return target.resolve().is_relative_to(base.resolve())  # type: ignore
        except AttributeError:
            return str(target.resolve()).startswith(str(base.resolve()))

    def sanitize_filename_for_lookup(fn: str) -> str:  # type: ignore
        return re.sub(r"[^a-zA-Z0-9._-]", "", os.path.basename(fn))


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/check", tags=["check"])

# ---------------------------------------------------------------------------
# PROGRESSIVE CALLBACK — background ThreadPool (PLAN_PROGRESSIVE_COMPARE_AND_ASYNC_CALLBACK_PER_LABEL)
# GOLDEN RULE 1: pengiriman HTTP callback TIDAK BOLEH sinkron di loop OCR.
# Wajib via _CALLBACK_EXECUTOR.submit(). Max 4 worker agar tidak berebut CPU
# dengan RapidOCR/Tesseract.
# ---------------------------------------------------------------------------
def _callback_workers() -> int:
    try:
        from app.config import CALLBACK_WORKERS as _w
        return max(1, min(int(_w or 2), 4))
    except Exception:
        try:
            return max(1, min(int(os.getenv("CALLBACK_WORKERS", "2") or 2), 4))
        except Exception:
            return 2


_CALLBACK_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=_callback_workers(),
    thread_name_prefix="check_callback_worker_",
)


def _is_progressive_enabled() -> bool:
    """Baca flag progressive saat runtime (mendukung rollback via .env + restart)."""
    try:
        from app.config import CHECK_OCR_PROGRESSIVE_CALLBACK as _flag
        return bool(_flag)
    except Exception:
        return bool(globals().get("CHECK_OCR_PROGRESSIVE_CALLBACK", True))

# Regex batch_code — sama dengan Laravel
BATCH_CODE_RE = re.compile(r"^[A-Za-z0-9\-_]+$")
ALLOWED_EXTS = {".png", ".jpg", ".jpeg"}
MAGIC_PNG = b"\x89PNG\r\n\x1a\n"
MAGIC_JPEG = b"\xff\xd8\xff"

# Jumlah gambar per batch yang diterima — sinkron Laravel between:4,5
# (CheckCaptureController, CheckExternalClient, CheckOcrClient: in [4,5])
CHECK_ALLOWED_COUNTS = (4, 5)


def _validate_api_key(x_api_key: Optional[str]) -> bool:
    """Validasi X-API-Key timing-safe. Jika key kosong (dev), izinkan."""
    valid_keys = [k for k in (CHECK_API_KEY, LARAVEL_API_KEY) if k]
    if not valid_keys:
        logger.warning("[CHECK] CHECK_API_KEY and LARAVEL_API_KEY empty — allowing all (dev mode)")
        return True
    if not x_api_key:
        return False
    import hmac

    for expected in valid_keys:
        if hmac.compare_digest(x_api_key, expected):
            return True
    return False


def _validate_magic(file_bytes: bytes, ext: str) -> bool:
    if ext == ".png":
        return len(file_bytes) >= 8 and file_bytes[:8].startswith(MAGIC_PNG)
    if ext in (".jpg", ".jpeg"):
        return len(file_bytes) >= 3 and file_bytes[:3].startswith(MAGIC_JPEG)
    return False


def _is_valid_slot_file(name: str) -> bool:
    return bool(re.match(r"^slot[1-5]-.+\.(png|jpg|jpeg)$", name, re.IGNORECASE))


def _slot_from_filename(name: str) -> int:
    m = re.match(r"^slot([1-5])-.*", name, re.IGNORECASE)
    return int(m.group(1)) if m else 99


def _build_slot_filename(slot: int, safe_batch: str, safe_orig: str, ext: str) -> str:
    """
    Bangun nama file tersimpan untuk 1 slot check.

    FIX_ACUAN_PER_LABEL (2026-10-08): basename yang dipotong [:32] bisa
    memenggal penanda sisi ("...-slot1-front-1" → "...-slot1-f") sehingga
    detect_side() = "unknown" dan semua slot jatuh ke acuan tunggal
    (kasus: semua slot dibandingankan ke teks label 1 FRONT). Karena itu
    suffix sisi eksplisit (-front/-back, dibaca dari nama asli SEBELUM
    dipotong) selalu ditambahkan bila terdeteksi, agar routing front/back
    di _process_batch_ocr selalu benar.

    Aturan lama dipertahankan: nama generik → slot{N}-{batch}{ext};
    nama panjang → potong head 32 char, total maks 100 char.
    TIDAK PERNAH raise (gagal deteksi sisi → suffix "").
    """
    low = (safe_orig or "").lower()
    # Anggap generic jika nama hanya slot1.png, image.png, blob, atau kosong
    is_generic = (
        (not safe_orig)
        or (low in ("image.png", "blob", "file", "image.jpg", "image.jpeg"))
        or (low in (f"slot{slot}.png", f"slot{slot}.jpg", f"slot{slot}.jpeg"))
        or bool(re.match(r"^slot[1-5]$", os.path.splitext(safe_orig)[0], re.IGNORECASE))
    )
    # Sisi dari nama asli (pola sama dengan side_detect). "slot1 tanpa ext
    # sudah handle, tapi safe_orig selalu ada ext" — komentar lama dipertahankan.
    side_suffix = ""
    try:
        _m = re.search(r"-(front|back)(?:-\d+)?$", os.path.splitext(safe_orig)[0], re.IGNORECASE)
        if _m:
            side_suffix = f"-{_m.group(1).lower()}"
    except Exception:
        side_suffix = ""
    if is_generic:
        return f"slot{slot}-{safe_batch}{side_suffix}{ext}"
    base_no_ext = os.path.splitext(safe_orig)[0]
    base_no_ext = base_no_ext[:32]
    safe_name = f"slot{slot}-{base_no_ext}{side_suffix}{ext}"
    if len(safe_name) > 100:
        safe_name = f"slot{slot}-{safe_batch}{side_suffix}{ext}"
    return safe_name


def _discover_batch_slots(batch_code: str) -> List[int]:
    """
    Scan folder batch dan kembalikan sorted slot numbers yang punya file valid.

    Dipakai pipeline OCR agar dinamis mengikuti jumlah aktual (4 atau 5),
    bukan hardcoded range(1, 6). Return [] jika folder tidak ada.
    """
    bdir = _batch_dir(batch_code)
    if not bdir.exists() or not bdir.is_dir():
        return []
    slots: List[int] = []
    for slot in range(1, 6):
        p = _slot_png_path(batch_code, slot)
        if p and p.exists():
            slots.append(slot)
    return sorted(slots)


@router.post("/captures")
async def save_check_batch(
    request: Request,
    batch_code: str = Form(..., description="CHK-YYYYMMDD-His-xxxx, regex ^[A-Za-z0-9\\-_]+$"),
    reference_id: int = Form(..., description="ID reference di Laravel"),
    template_id: str = Form(..., description="XL/L/M/S"),
    slot_numbers: Optional[List[str]] = Form(None, description="List 4-5 int 1..5, kirim sebagai slot_numbers=1 repeated atau slot_numbers[0]=1"),
    filenames: Optional[List[str]] = Form(None, description="Opsional original filenames (filenames[0] atau filenames repeated)"),
    images: Optional[List[UploadFile]] = File(None, description="4-5 files PNG/JPEG (images atau images[0..n])"),
    user_id: Optional[str] = Form(None, description="Opsional audit user_id dari Laravel"),
    timestamp: Optional[str] = Form(None, description="Opsional ISO8601 timestamp dari Laravel"),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    x_api_key_lower: Optional[str] = Header(None, alias="x-api-key"),
):
    # 0. Fallback cari api key di raw headers jika Header alias gagal
    effective_key = x_api_key or x_api_key_lower
    if not effective_key:
        for k in ("X-API-Key", "x-api-key", "X-Api-Key"):
            v = request.headers.get(k)
            if v:
                effective_key = v
                break
        if not effective_key:
            effective_key = request.query_params.get("api_key")

    # 1. Validasi X-API-Key
    if not _validate_api_key(effective_key):
        logger.warning(
            f"[CHECK] 401 Invalid API key for batch {batch_code} from {request.client.host if request.client else 'unknown'}"
        )
        return JSONResponse(
            status_code=401,
            content={"success": False, "message": "Invalid API key", "error": "unauthorized"},
        )

    # 1b. Fallback parsing untuk bracket style (Laravel Http::attach images[0], slot_numbers[0])
    # Jika client mengirim sebagai images[0]..[4] atau slot_numbers[0]..[4], FastAPI List[UploadFile] tidak otomatis ter-map.
    # Maka baca raw form dan kumpulkan manual.
    form = None
    try:
        form = await request.form()
    except Exception:
        form = None

    # Normalize slot_numbers dari bracket jika slot_numbers None atau count di luar (4,5)
    if form is not None:
        # slot_numbers fallback
        if not slot_numbers or len(slot_numbers) not in CHECK_ALLOWED_COUNTS:
            collected_slots = []
            # Coba getlist untuk key exact
            try:
                # Starlette FormData.getlist
                if hasattr(form, "getlist"):
                    gl = form.getlist("slot_numbers")
                    if gl:
                        collected_slots = list(gl)
            except Exception:
                pass
            if not collected_slots:
                # Scan keys like slot_numbers[0], slot_numbers[1], slot_numbers[]
                temp_dict = {}
                for k, v in form.multi_items() if hasattr(form, "multi_items") else form.items():
                    if isinstance(v, str) or not hasattr(v, "filename"):
                        # Hanya field string (bukan file)
                        if k == "slot_numbers" or k.startswith("slot_numbers["):
                            # Extract index jika ada: slot_numbers[0] -> 0, slot_numbers -> append
                            if k == "slot_numbers" or k == "slot_numbers[]":
                                collected_slots.append(v)
                            elif k.startswith("slot_numbers["):
                                # Parse index
                                try:
                                    idx_str = k[len("slot_numbers[") :].rstrip("]")
                                    idx = int(idx_str) if idx_str.isdigit() else len(collected_slots)
                                    # Pastikan list cukup panjang
                                    while len(collected_slots) <= idx:
                                        collected_slots.append(None)  # type: ignore
                                    collected_slots[idx] = v
                                except Exception:
                                    collected_slots.append(v)
                # Filter None
                collected_slots = [x for x in collected_slots if x is not None]
            else:
                collected_slots = [x for x in collected_slots if x is not None]
            if collected_slots:
                slot_numbers = collected_slots  # type: ignore

        # filenames fallback (optional, count fleksibel 4-5 — dicocokkan dgn images nanti)
        if form is not None and (not filenames or len(filenames) not in CHECK_ALLOWED_COUNTS):
            collected_fnames = []
            try:
                if hasattr(form, "getlist"):
                    gl = form.getlist("filenames")
                    if gl:
                        collected_fnames = list(gl)
            except Exception:
                pass
            if not collected_fnames:
                temp = {}
                for k, v in form.multi_items() if hasattr(form, "multi_items") else form.items():
                    if isinstance(v, str):
                        if k == "filenames" or k.startswith("filenames["):
                            if k == "filenames" or k == "filenames[]":
                                collected_fnames.append(v)
                            elif k.startswith("filenames["):
                                try:
                                    idx_str = k[len("filenames[") :].rstrip("]")
                                    idx = int(idx_str) if idx_str.isdigit() else len(collected_fnames)
                                    while len(collected_fnames) <= idx:
                                        collected_fnames.append(None)  # type: ignore
                                    collected_fnames[idx] = v
                                except Exception:
                                    collected_fnames.append(v)
                collected_fnames = [x for x in collected_fnames if x is not None]
            else:
                collected_fnames = [x for x in collected_fnames if x is not None]
            if collected_fnames:
                filenames = collected_fnames

        # images fallback
        if not images or len(images) not in CHECK_ALLOWED_COUNTS:
            collected_images: List[UploadFile] = []
            # Cek getlist dulu
            try:
                if hasattr(form, "getlist"):
                    gl = form.getlist("images")
                    # Filter hanya UploadFile
                    gl_files = [x for x in gl if hasattr(x, "filename")]
                    if gl_files:
                        collected_images = gl_files  # type: ignore
            except Exception:
                pass
            if not collected_images:
                # Scan multi_items untuk keys images, images[0], images[]
                temp_map = {}
                for k, v in form.multi_items() if hasattr(form, "multi_items") else form.items():
                    if hasattr(v, "filename"):  # UploadFile
                        if k == "images" or k.startswith("images[") or k == "images[]":
                            if k == "images" or k == "images[]":
                                collected_images.append(v)  # type: ignore
                            elif k.startswith("images["):
                                try:
                                    idx_str = k[len("images[") :].rstrip("]")
                                    idx = int(idx_str) if idx_str.isdigit() else len(collected_images)
                                    # Expand list
                                    while len(collected_images) <= idx:
                                        collected_images.append(None)  # type: ignore
                                    collected_images[idx] = v  # type: ignore
                                except Exception:
                                    collected_images.append(v)  # type: ignore
                # Filter None dan sort by index completeness
                collected_images = [x for x in collected_images if x is not None]
            else:
                collected_images = [x for x in collected_images if x is not None]
            if collected_images:
                images = collected_images  # type: ignore

    # 2. Validasi batch_code
    batch_code = batch_code.strip()
    if not batch_code or len(batch_code) > 64 or not BATCH_CODE_RE.match(batch_code):
        return JSONResponse(
            status_code=422,
            content={
                "success": False,
                "message": "Invalid batch_code format. Must match ^[A-Za-z0-9\\-_]+$ and 1..64 chars",
                "error": "validation_failed",
            },
        )
    safe_batch = sanitize_filename_for_lookup(batch_code)
    if safe_batch != batch_code:
        return JSONResponse(
            status_code=422,
            content={"success": False, "message": "Invalid batch_code (contains unsafe characters)", "error": "validation_failed"},
        )

    # 3. Validasi reference_id & template_id
    if reference_id <= 0:
        return JSONResponse(
            status_code=422, content={"success": False, "message": "reference_id must be >0", "error": "validation_failed"}
        )
    if not template_id or len(template_id) > 32:
        return JSONResponse(
            status_code=422, content={"success": False, "message": "Invalid template_id", "error": "validation_failed"}
        )

    # 4. Validasi images count (setelah fallback bracket) — sinkron Laravel between:4,5
    expected_count = len(images) if images else 0
    if expected_count not in CHECK_ALLOWED_COUNTS:
        return JSONResponse(
            status_code=422,
            content={
                "success": False,
                "message": f"images must contain 4 or 5 items (got {expected_count})",
                "error": "validation_failed",
                "expected_counts": list(CHECK_ALLOWED_COUNTS),
            },
        )

    # 5. Validasi slot_numbers (setelah fallback) — unique 1..5, count == len(images)
    if not slot_numbers or len(slot_numbers) != expected_count:
        return JSONResponse(
            status_code=422,
            content={
                "success": False,
                "message": f"slot_numbers must contain exactly {expected_count} items (match images count)",
                "error": "validation_failed",
            },
        )
    try:
        slot_ints = [int(s) for s in slot_numbers]  # type: ignore
    except Exception:
        return JSONResponse(
            status_code=422, content={"success": False, "message": "slot_numbers must be integers 1..5", "error": "validation_failed"}
        )
    if len(set(slot_ints)) != expected_count or any(s < 1 or s > 5 for s in slot_ints):
        return JSONResponse(
            status_code=422,
            content={"success": False, "message": f"slot_numbers must be unique values 1..5 (got {sorted(slot_ints)})", "error": "validation_failed"},
        )

    # 6. Buat folder batch
    batch_dir = CHECK_UPLOAD_ROOT / safe_batch
    if not is_safe_path(CHECK_UPLOAD_ROOT, batch_dir):
        return JSONResponse(
            status_code=422, content={"success": False, "message": "Invalid batch_code path", "error": "validation_failed"}
        )
    batch_dir.mkdir(parents=True, exist_ok=True)

    # 7. Idempotency: jika sudah ada N file valid, return duplicate
    existing_files = list(batch_dir.glob("slot*.png")) + list(batch_dir.glob("slot*.jpg")) + list(batch_dir.glob("slot*.jpeg"))
    existing_valid = [p for p in existing_files if p.is_file() and _is_valid_slot_file(p.name)]
    if len(existing_valid) >= expected_count:
        existing_valid.sort(key=lambda p: _slot_from_filename(p.name))
        rel_paths = []
        for p in existing_valid:
            try:
                rel = p.relative_to(Path.cwd())
                rel_paths.append(str(rel).replace("\\", "/"))
            except ValueError:
                rel_paths.append(f"uploads/check/{safe_batch}/{p.name}")
        logger.info(f"[CHECK] duplicate batch {safe_batch} already has {len(existing_valid)} files, returning existing")
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "batch_code": safe_batch,
                "reference_id": reference_id,
                "template_id": template_id,
                "saved_count": len(existing_valid[:expected_count]),
                "paths": rel_paths[:expected_count],
                "message": "Batch already exists, returning existing paths",
                "duplicate": True,
            },
        )

    # 8. Loop simpan N files
    saved_paths: List[str] = []

    effective_filenames = (
        filenames if filenames and len(filenames) == expected_count else [img.filename or f"slot{slot_ints[i]}.png" for i, img in enumerate(images)]
    )

    for i, img in enumerate(images):
        slot = slot_ints[i]
        orig_name = effective_filenames[i] if i < len(effective_filenames) else (img.filename or f"slot{slot}.png")

        # a. Validasi content_type
        ctype = (img.content_type or "").lower()
        # Toleransi: jika ctype kosong, cek dari ext saja (beberapa client tidak kirim mime)
        if ctype and ctype not in CHECK_ALLOWED_MIMES and ctype not in ("image/png", "image/jpeg", "image/jpg"):
            return JSONResponse(
                status_code=422,
                content={
                    "success": False,
                    "message": f"File slot {slot} invalid content_type: {ctype}. Only PNG/JPEG allowed",
                    "error": "invalid_file_type",
                },
            )

        # b. Validasi ekstensi
        ext = os.path.splitext(orig_name)[1].lower() or os.path.splitext(img.filename or "")[1].lower() or ".png"
        if ext not in ALLOWED_EXTS:
            if ctype == "image/png":
                ext = ".png"
            elif ctype in ("image/jpeg", "image/jpg"):
                ext = ".jpg"
            else:
                return JSONResponse(
                    status_code=422,
                    content={"success": False, "message": f"File slot {slot} invalid extension: {ext}", "error": "invalid_file_type"},
                )

        # c. Baca bytes + cek size + magic
        try:
            file_bytes = await img.read()
        except Exception as e:
            logger.error(f"[CHECK] failed read slot {slot}: {e}", exc_info=True)
            return JSONResponse(
                status_code=500, content={"success": False, "message": f"Failed to read file slot {slot}", "error": "internal_error"}
            )

        if len(file_bytes) > CHECK_MAX_FILE_SIZE:
            return JSONResponse(
                status_code=413,
                content={
                    "success": False,
                    "message": f"File slot {slot} exceeds 5MB limit ({len(file_bytes)} bytes)",
                    "error": "file_too_large",
                },
            )
        if len(file_bytes) == 0:
            return JSONResponse(
                status_code=422, content={"success": False, "message": f"File slot {slot} is empty", "error": "validation_failed"}
            )

        if not _validate_magic(file_bytes, ext):
            return JSONResponse(
                status_code=422,
                content={"success": False, "message": f"File slot {slot} content does not match {ext}", "error": "invalid_file_type"},
            )

        # d. Sanitasi filename & buat dest
        safe_orig = sanitize_filename_for_lookup(os.path.basename(orig_name))
        # FIX_ACUAN_PER_LABEL (2026-10-08): penamaan via helper agar penanda
        # sisi (-front/-back) tidak hilang terpotong (lihat _build_slot_filename).
        safe_name = _build_slot_filename(slot, safe_batch, safe_orig, ext)

        dest = batch_dir / safe_name
        if not is_safe_path(CHECK_UPLOAD_ROOT, dest):
            return JSONResponse(
                status_code=422, content={"success": False, "message": f"Unsafe path for slot {slot}", "error": "validation_failed"}
            )

        # e. Simpan ke disk
        try:
            with open(dest, "wb") as out:
                out.write(file_bytes)
        except Exception as e:
            logger.error(f"[CHECK] failed save slot {slot} to {dest}: {e}", exc_info=True)
            return JSONResponse(
                status_code=500, content={"success": False, "message": f"Failed to save file slot {slot}", "error": "internal_error"}
            )

        # f. Catat path relative
        try:
            rel = dest.relative_to(Path.cwd())
            rel_str = str(rel).replace("\\", "/")
        except ValueError:
            rel_str = f"uploads/check/{safe_batch}/{safe_name}"
        saved_paths.append(rel_str)
        logger.info(f"[CHECK] saved slot {slot} -> {rel_str} ({len(file_bytes)} bytes)")

    # 9. Log sukses (sertakan user_id/timestamp jika ada untuk audit)
    if user_id or timestamp:
        logger.info(
            f"[CHECK] saved batch {safe_batch} reference_id={reference_id} template={template_id} user_id={user_id} timestamp={timestamp} {len(saved_paths)} files to {batch_dir}"
        )
    else:
        logger.info(f"[CHECK] saved batch {safe_batch} reference_id={reference_id} template={template_id} {len(saved_paths)} files to {batch_dir}")
    print(f"[CHECK] saved batch {safe_batch} {len(saved_paths)} files (user_id={user_id})")

    # 9b. Efisiensi: cleanup old batches (retain 1 terbaru) — NEW
    try:
        _cleanup_old_batches(safe_batch)
    except Exception as e:
        logger.warning(f"[CHECK] cleanup_old_batches failed for {safe_batch}: {e}", exc_info=True)
        print(f"[CHECK CLEANUP ERROR] {safe_batch}: {e}")
        # Jangan gagalkan response 200 — storage baru sudah sukses, cleanup best-effort

    # 9c. Tulis latest_batch.txt (sudah di _cleanup_old_batches, tapi double ensure)
    try:
        from app.config import CHECK_LATEST_BATCH_FILE

        latest_tmp = CHECK_LATEST_BATCH_FILE.with_suffix(".tmp")
        latest_tmp.write_text(safe_batch, encoding="utf-8")
        latest_tmp.replace(CHECK_LATEST_BATCH_FILE)
    except Exception:
        pass

    # 10. Return sukses — sort paths by slot number agar urut
    slot_path_map = {slot_ints[i]: saved_paths[i] for i in range(expected_count)}
    sorted_paths = [slot_path_map[s] for s in sorted(slot_path_map.keys())]

    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "batch_code": safe_batch,
            "reference_id": reference_id,
            "template_id": template_id,
            "saved_count": expected_count,
            "paths": sorted_paths,
            "message": f"{expected_count} files saved to uploads/check/{safe_batch}",
            "duplicate": False,
        },
    )


# ---------------------------------------------------------------------------
# OCR Batch Helpers — Process N Labels (PLAN_CHECK_OCR_5_LABEL_PROCESS, compat 4-5)
# ---------------------------------------------------------------------------

def _check_process_429(wait_seconds: float) -> JSONResponse:
    """Return 429 untuk rate limit Process."""
    wait_int = int(wait_seconds) + 1
    return JSONResponse(
        status_code=429,
        content={
            "success": False,
            "message": f"Process rate limit exceeded. Retry after {wait_int}s",
            "error": "too_many_requests",
            "retry_after_seconds": wait_int,
        },
        headers={"Retry-After": str(wait_int), "X-RateLimit-Remaining": "0"},
    )


def _batch_dir(batch_code: str) -> Path:
    safe = sanitize_filename_for_lookup(batch_code)
    return CHECK_UPLOAD_ROOT / safe


def _batch_ocr_dir(batch_code: str) -> Path:
    return _batch_dir(batch_code) / "ocr"


def _slot_png_path(batch_code: str, slot: int) -> Optional[Path]:
    bdir = _batch_dir(batch_code)
    if not bdir.exists():
        return None
    for ext in (".png", ".jpg", ".jpeg"):
        candidates = list(bdir.glob(f"slot{slot}-*{ext}"))
        if not candidates:
            candidates = list(bdir.glob(f"slot{slot}{ext}"))
        for p in candidates:
            if p.is_file():
                return p
    return None


def _is_empty_result(cleaned: dict) -> bool:
    if not cleaned:
        return True
    # OCR Engine Upgrade (2026-09-11): 'none' = hasil VALID, bukan kosong.
    try:
        from app.ocr.parser import is_none_text

        if is_none_text(cleaned.get("cleaned_text", "")):
            return False
    except ImportError:
        if isinstance(cleaned.get("cleaned_text", ""), str) and cleaned.get("cleaned_text", "").strip().lower() == "none":
            return False
    ct = cleaned.get("cleaned_text", "")
    lines = cleaned.get("lines", [])
    if not ct or not ct.strip():
        return True
    if not lines or not any(ln.strip() for ln in lines if isinstance(ln, str)):
        return True
    return False


def _read_batch_json(batch_code: str) -> Optional[dict]:
    bjson = _batch_ocr_dir(batch_code) / "batch.json"
    if bjson.exists():
        try:
            with open(bjson, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def _write_batch_json(batch_code: str, data: dict) -> None:
    odir = _batch_ocr_dir(batch_code)
    odir.mkdir(parents=True, exist_ok=True)
    bjson = odir / "batch.json"
    tmp = bjson.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    tmp.replace(bjson)


def _cleanup_old_batches(keep_batch_code: str) -> dict:
    """
    Hapus semua folder batch lama di uploads/check/ kecuali keep_batch_code.
    Retain N terbaru (CHECK_RETAIN_BATCHES), default 1 = hanya terbaru.
    Dipanggil setelah POST /captures sukses simpan N files.
    NOTE: uploads/references/ sengaja diabaikan — library permanen
    (PLAN_REFERENCE_LIBRARY_FASTAPI). Fungsi ini hanya iterasi CHECK_UPLOAD_ROOT.
    Returns: {"deleted_batches": int, "deleted_files": int, "kept": str, "errors": []}
    """
    try:
        from app.config import CHECK_RETAIN_BATCHES, CHECK_CLEANUP_ON_CAPTURE, CHECK_LATEST_BATCH_FILE
    except ImportError:
        CHECK_RETAIN_BATCHES = 1
        CHECK_CLEANUP_ON_CAPTURE = True
        CHECK_LATEST_BATCH_FILE = CHECK_UPLOAD_ROOT / "latest_batch.txt"

    if not CHECK_CLEANUP_ON_CAPTURE:
        print(f"[CHECK CLEANUP] skipped (CHECK_CLEANUP_ON_CAPTURE=false) keep={keep_batch_code}")
        return {"deleted_batches": 0, "deleted_files": 0, "skipped": True, "kept": keep_batch_code}

    keep_safe = sanitize_filename_for_lookup(keep_batch_code)
    if not keep_safe or keep_safe != keep_batch_code:
        return {"deleted_batches": 0, "deleted_files": 0, "errors": [f"invalid keep_batch_code: {keep_batch_code}"], "kept": keep_batch_code}

    stats = {"deleted_batches": 0, "deleted_files": 0, "errors": [], "kept": keep_safe}
    try:
        candidates: list[Path] = []
        for p in CHECK_UPLOAD_ROOT.iterdir():
            if not p.is_dir():
                continue
            if p.name in (".git", "__pycache__", ".venv", "venv"):
                continue
            if not BATCH_CODE_RE.match(p.name):
                continue
            if p.name == keep_safe:
                continue
            if not is_safe_path(CHECK_UPLOAD_ROOT, p):
                stats["errors"].append(f"skip unsafe {p}")
                continue
            candidates.append(p)

        # Handle RETAIN >1: keep N-1 terbaru besides keep
        if candidates and CHECK_RETAIN_BATCHES > 1:
            candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            keep_n = CHECK_RETAIN_BATCHES - 1
            to_keep = set(p.name for p in candidates[:keep_n])
            to_delete = [p for p in candidates if p.name not in to_keep]
        else:
            to_delete = candidates

        def _rmtree_force(path: Path) -> None:
            """Windows-safe rmtree: chmod + retry on PermissionError."""
            import stat as _stat

            def _onerror(func, p, exc_info):
                # exc_info = (type, value, traceback)
                try:
                    os.chmod(p, _stat.S_IWRITE | _stat.S_IREAD)
                    func(p)
                except Exception:
                    # second try: walk and chmod all
                    try:
                        for root, dirs, files in os.walk(path):
                            for fname in files:
                                fp = os.path.join(root, fname)
                                try:
                                    os.chmod(fp, _stat.S_IWRITE | _stat.S_IREAD)
                                except Exception:
                                    pass
                            for dname in dirs:
                                dp = os.path.join(root, dname)
                                try:
                                    os.chmod(dp, _stat.S_IWRITE | _stat.S_IREAD)
                                except Exception:
                                    pass
                        shutil.rmtree(path, ignore_errors=False)
                    except Exception as e2:
                        raise e2

            # First try chmod all files before rmtree (prevent lock)
            try:
                for fname in path.rglob("*"):
                    try:
                        if fname.is_file() or fname.is_dir():
                            os.chmod(fname, _stat.S_IWRITE | _stat.S_IREAD)
                    except Exception:
                        pass
                try:
                    os.chmod(path, _stat.S_IWRITE | _stat.S_IREAD)
                except Exception:
                    pass
            except Exception:
                pass
            shutil.rmtree(path, onerror=_onerror)

        for p in to_delete:
            try:
                n_files = sum(1 for _ in p.rglob("*") if _.is_file())
                _rmtree_force(p)
                stats["deleted_batches"] += 1
                stats["deleted_files"] += n_files
                print(f"[CHECK CLEANUP] Deleted old batch {p.name} ({n_files} files)")
            except PermissionError as e:
                stats["errors"].append(f"{p.name}: locked ({e})")
                print(f"[CHECK CLEANUP WARN] Locked {p.name}: {e}")
            except Exception as e:
                stats["errors"].append(f"{p.name}: {e}")
                print(f"[CHECK CLEANUP ERROR] {p}: {e}")

        # Tulis latest_batch.txt pointer (atomic)
        try:
            latest_file = CHECK_LATEST_BATCH_FILE
            tmp = latest_file.with_suffix(".tmp")
            tmp.write_text(keep_safe, encoding="utf-8")
            tmp.replace(latest_file)
        except Exception as e:
            stats["errors"].append(f"latest_batch.txt: {e}")

        print(f"[CHECK CLEANUP] Done keep={keep_safe}: deleted {stats['deleted_batches']} batches, {stats['deleted_files']} files")
    except Exception as e:
        stats["errors"].append(f"cleanup failed: {e}")
        print(f"[CHECK CLEANUP ERROR] {e}")
    return stats


def _process_single_slot(batch_code: str, slot: int, image_path: Path, reference_text: Optional[str] = None) -> dict:
    """Proses 1 slot via dispatcher OCR (RapidOCR default, Tesseract fallback).

    COMPARE_TEXT_SIMILARITY_PLAN (2026-09-12): bila reference_text diberikan dan
    SIMILARITY_ENABLED=true, hitung similarity ke blok "similarity" + excerpt.
    """
    start = time.time()
    orig_name = image_path.name

    # low-res warning
    try:
        from PIL import Image as _PIL

        _tmp = _PIL.open(image_path)
        _w, _h = _tmp.size
        if _w < 400 or _h < 400:
            print(f"[CHECK OCR WARN] Low-res slot{slot} {batch_code} {_w}x{_h}")
    except Exception:
        pass

    # OCR Engine Upgrade (2026-09-11): 1 pintu via dispatcher (input original).
    try:
        from app.ocr.dispatch import extract_label_text
    except ImportError as e:
        raise RuntimeError(f"OCR modules not available: {e}") from e

    ocr_out = extract_label_text(str(image_path), orig_name)
    engine = ocr_out.get("engine", "tesseract")
    text_present = ocr_out.get("text_present")
    if engine == "rapidocr":
        # Tuning 2026-09-12: rec_model small (cepat) | medium (akurat, fix H->K).
        rec_model = ocr_out.get("rec_model") or "PP-OCRv6_rec_small"
        det_model = ocr_out.get("det_model") or "PP-OCRv6_det_small"
        steps = [f"rapidocr_det_rec({det_model}+{rec_model})", "original_no_upscale"]
        attempts = [f"rapidocr_det_rec({rec_model})"]
        used_pass = 1
        language = "en"
    else:
        steps = ocr_out.get("preprocessing_steps", [])
        attempts = ocr_out.get("preprocessing_attempts", [])
        used_pass = ocr_out.get("preprocessed_pass", 1)
        language = "eng"

    cleaned = {
        "raw_text": ocr_out.get("raw_text", ""),
        "cleaned_text": ocr_out.get("cleaned_text", ""),
        "lines": ocr_out.get("lines", []),
    }

    elapsed = time.time() - start
    if engine == "rapidocr" and text_present is False:
        # Kontrak 'none' (§6 plan): jawaban VALID, bukan error.
        status = "success"
        is_empty = False
        error_msg = None
        empty_reason = "no_text_detected"
        print(f"[CHECK OCR] slot{slot} {batch_code} -> no text ('none') {elapsed:.2f}s")
    else:
        is_empty = _is_empty_result(cleaned)
        status = "failed" if is_empty else "success"
        error_msg = "empty cleaned_text after 2-pass — dark low-contrast, need retake or better lighting" if is_empty else None
        empty_reason = "no_text_detected_after_2pass" if is_empty else None
    result = {
        "batch_code": batch_code,
        "slot": slot,
        "filename": image_path.name,
        "original_filename": orig_name,
        "stored_path": str(image_path).replace("\\", "/"),
        "engine": engine,
        "language": language,
        "text_present": text_present,
        "det_threshold": ocr_out.get("det_threshold"),
        "rec_threshold": ocr_out.get("rec_threshold"),
        # Tuning 2026-09-12 (aditif).
        "det_model": ocr_out.get("det_model"),
        "rec_model": ocr_out.get("rec_model"),
        "preprocessed": True,
        "preprocessing_steps": steps,
        "preprocessing_attempts": attempts,
        "preprocessed_pass": used_pass,
        "raw_text": cleaned["raw_text"] if cleaned else "",
        "cleaned_text": cleaned["cleaned_text"] if cleaned else "",
        "lines": cleaned["lines"] if cleaned else [],
        "is_empty": is_empty,
        "empty_reason": empty_reason,
        "processed_at": datetime.now(JAKARTA_TZ).isoformat(),
        "status": status,
        "error": error_msg,
        "file_size_bytes": image_path.stat().st_size if image_path.exists() else 0,
        "processing_time_seconds": round(elapsed, 3),
    }

    # --- COMPARE_TEXT_SIMILARITY_PLAN (2026-09-12): reference vs slot ---
    # Dihitung SEBELUM tulis slotN.json agar otomatis ikut callback yang ada.
    # Kill switch SIMILARITY_ENABLED=false → tanpa blok (perilaku lama).
    try:
        from app.config import SIMILARITY_ENABLED, SIMILARITY_PASS_THRESHOLD
    except ImportError:
        SIMILARITY_ENABLED, SIMILARITY_PASS_THRESHOLD = True, 0.85
    if SIMILARITY_ENABLED and reference_text is not None:
        try:
            from app.compare_text.similarity import compare_slot

            sim = compare_slot(
                reference_text,
                result.get("cleaned_text", ""),
                result.get("lines", []),
                SIMILARITY_PASS_THRESHOLD,
            )
            result["similarity"] = sim
            result["reference_excerpt"] = str(reference_text or "")[:120]

            # ★ BARU: Print diff_details ke console FastAPI server
            _diff_details = sim.get("diff_details", [])
            _verdict = sim.get("verdict", "?")
            _score = sim.get("score", "?")
            print(
                f"[COMPARE] batch={batch_code} slot={slot} score={_score} "
                f"verdict={_verdict} (ref_lines={sim.get('ref_lines')} "
                f"slot_lines={sim.get('slot_lines')} matched={sim.get('matched_lines')})"
            )
            if _diff_details:
                print(f"[COMPARE DIFF] batch={batch_code} slot={slot} verdict={_verdict} — {len(_diff_details)} perbedaan:")
                for dd in _diff_details[:10]:  # Limit 10 di console agar tidak terlalu panjang
                    _type = dd.get("type", "?")
                    _rln = dd.get("ref_line_num", "?")
                    _cln = dd.get("check_line_num", "?")
                    _rtxt = dd.get("ref_text", "")
                    _ctxt = dd.get("check_text", "")
                    if _type == "changed":
                        print(f'[COMPARE DIFF]   changed L{_rln}→L{_cln}: Acuan="{_rtxt}" → Check="{_ctxt}"')
                    elif _type == "missing":
                        print(f'[COMPARE DIFF]   missing L{_rln}: Acuan="{_rtxt}" (tidak ada di Check)')
                    elif _type == "added":
                        print(f'[COMPARE DIFF]   added L{_cln}: Check="{_ctxt}" (tidak ada di Acuan)')
                if len(_diff_details) > 10:
                    print(f"[COMPARE DIFF]   ... dan {len(_diff_details) - 10} perbedaan lainnya")
            elif _verdict == "PASS":
                print(f"[COMPARE DIFF] batch={batch_code} slot={slot} PASS — semua baris cocok")
        except Exception as e:
            logger.warning(f"[COMPARE] slot{slot} {batch_code} exception: {e}")
            result["similarity"] = {
                "version": "v1", "verdict": "SKIP", "score": None,
                "threshold": SIMILARITY_PASS_THRESHOLD, "ref_lines": 0,
                "slot_lines": 0, "matched_lines": 0,
                "reason": f"exception:{type(e).__name__}"[:120],
                "diff_details": [],
            }
            result["reference_excerpt"] = str(reference_text or "")[:120]

    odir = _batch_ocr_dir(batch_code)
    odir.mkdir(parents=True, exist_ok=True)
    slot_json = odir / f"slot{slot}.json"
    tmp = slot_json.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    tmp.replace(slot_json)
    print(f"[CHECK OCR] slot{slot} {batch_code} -> {status} is_empty={is_empty} {elapsed:.2f}s")
    # [OCR-REENABLE 2026-09-11 / PLAN_REVERT_TO_OCR] Console log raw teks per slot (sementara)
    # SINGLE-LINE FORMAT (2026-10-07): cleaned 1 baris — loop per-baris dihapus.
    print("=" * 60)
    print(f"[CHECK OCR RAW] batch={batch_code} slot={slot} file={image_path.name} status={status}")
    print(f"[CHECK OCR RAW] cleaned_text: {repr(result.get('cleaned_text', ''))}")
    print("=" * 60)
    return result


def _select_slot_ref(
    slot: int,
    side: str,
    ref_front: Optional[str],
    ref_back: Optional[str],
    ref_single: Optional[str],
    ref_map: Optional[dict] = None,
) -> tuple:
    """
    Pilih teks acuan untuk 1 slot + laporkan sumbernya.

    FIX_WRONG_REFERENCE (2026-10-07): 1 batch bisa berisi N label beda produk,
    sehingga acuan per slot (map) mengalahkan acuan sisi/tunggal.

    Prioritas: per-slot map > sisi (front/back) > tunggal.
    (Fallback file diurus pemanggil — helper ini hanya menerima string jadi.)

    Args:
        slot: nomor slot 1..5.
        side: "front" | "back" | "unknown" (dari detect_side).
        ref_front/ref_back/ref_single: teks acuan (sudah strip) atau None.
        ref_map: dict {slot_int: teks} hasil normalisasi (atau None).

    Returns:
        (teks_atau_None, sumber: str) dengan sumber salah satu dari
        "slot" | "side:front" | "side:back" | "single" | "missing".

    TIDAK PERNAH raise.
    """
    try:
        if ref_map:
            t = (ref_map.get(int(slot)) or "").strip()
            if t:
                return t, "slot"
        s = (side or "unknown").lower()
        if s == "front" and ref_front:
            return ref_front, "side:front"
        if s == "back" and ref_back:
            return ref_back, "side:back"
        if ref_single:
            return ref_single, "single"
        return None, "missing"
    except Exception:
        return None, "missing"


def _normalize_ref_map(raw_map) -> dict:
    """
    Normalisasi `reference_texts` dari compare_request.json menjadi {slot: teks}.

    FIX_WRONG_REFERENCE (2026-10-07). Bentuk diterima:
      dict {"1": teks}, list [teks1, ...] (index 0 = slot 1),
      string JSON (via form/query). Nilai kosong/non-string diabaikan.
    Tiap teks truncate 20000 char. TIDAK PERNAH raise (gagal -> {}).
    """
    try:
        if isinstance(raw_map, str) and raw_map.strip():
            try:
                raw_map = json.loads(raw_map.strip())
            except Exception:
                return {}
        out: dict = {}
        if isinstance(raw_map, dict):
            for k, v in raw_map.items():
                try:
                    slot_n = int(str(k).strip())
                except Exception:
                    continue
                if 1 <= slot_n <= 5 and isinstance(v, str) and v.strip():
                    out[slot_n] = v.strip()[:20000]
        elif isinstance(raw_map, list):
            for idx, v in enumerate(raw_map[:5], start=1):
                if isinstance(v, str) and v.strip():
                    out[idx] = v.strip()[:20000]
        return out
    except Exception:
        return {}


def _send_slot_callback_task(batch_code: str, slot_payload: dict, progress_str: str = "") -> bool:
    """Worker background-thread: kirim 1 callback HTTP per slot ke Laravel.

    PROGRESSIVE CALLBACK (PLAN_PROGRESSIVE_COMPARE_AND_ASYNC_CALLBACK_PER_LABEL).
    Dijalankan via ``_CALLBACK_EXECUTOR.submit()`` — TIDAK BOLEH dipanggil
    sinkron di loop OCR (GOLDEN RULE 1).

    Backward-compatible (GOLDEN RULE 2): struktur ``slot_payload`` existing
    dipertahankan utuh; hanya field aditif ``event`` / ``progress`` /
    ``is_final`` yang disisipkan bila belum ada.

    Signing + retry + log + dead-letter = salinan persis logika borongan lama
    (``_post_one_slot``) agar verifikasi HMAC Laravel tidak berubah.
    TIDAK PERNAH raise.
    """
    try:
        try:
            from app.config import (
                CHECK_OCR_CALLBACK_ENABLED as _cb_enabled,
                CHECK_OCR_CALLBACK_URL as _cb_url,
                LARAVEL_API_KEY as _api_key,
                LARAVEL_HMAC_SECRET as _hmac_secret,
                CALLBACK_TIMEOUT_SECONDS,
                CALLBACK_RETRIES,
                CALLBACK_RETRY_BACKOFF_BASE,
                CALLBACK_ALLOWED_HOSTS,
                CALLBACK_LOG_FILE,
                CALLBACK_DEAD_LETTER_FILE,
                JAKARTA_TZ as _JTZ,
            )
            _tz = _JTZ
        except ImportError:
            from app.config import CHECK_OCR_CALLBACK_ENABLED as _cb_enabled  # type: ignore
            from app.config import CHECK_OCR_CALLBACK_URL as _cb_url  # type: ignore
            from app.config import LARAVEL_API_KEY as _api_key  # type: ignore
            try:
                from app.config import LARAVEL_HMAC_SECRET as _hmac_secret  # type: ignore
            except ImportError:
                _hmac_secret = ""
            try:
                from app.config import (  # type: ignore
                    CALLBACK_TIMEOUT_SECONDS,
                    CALLBACK_RETRIES,
                    CALLBACK_RETRY_BACKOFF_BASE,
                    CALLBACK_ALLOWED_HOSTS,
                    CALLBACK_LOG_FILE,
                    CALLBACK_DEAD_LETTER_FILE,
                )
            except ImportError:
                CALLBACK_TIMEOUT_SECONDS = 5.0
                CALLBACK_RETRIES = 3
                CALLBACK_RETRY_BACKOFF_BASE = 2.0
                CALLBACK_ALLOWED_HOSTS = {"127.0.0.1", "localhost"}
                from app.config import UPLOAD_DIR as _ud  # type: ignore
                CALLBACK_LOG_FILE = _ud / "callback_log.jsonl"
                CALLBACK_DEAD_LETTER_FILE = _ud / "callback_dead_letter.jsonl"
            _tz = JAKARTA_TZ
    except Exception as e:
        logger.warning(f"[CHECK CALLBACK] Config import error: {e}")
        return False

    if not _cb_enabled or not _cb_url or not _api_key or not _hmac_secret:
        return False

    # Helper sign — salinan persis perilaku lama (body_json string -> sign_payload).
    try:
        from app.callback.signer import sign_payload  # type: ignore

        def _sign(body: str, ts: str, secret: str) -> str:
            return sign_payload(body, ts, secret)
    except ImportError:
        import hmac as _hmac
        import hashlib as _hash

        def _sign(body: str, ts: str, secret: str) -> str:
            msg = (body + ts).encode("utf-8")
            return _hmac.new(secret.encode("utf-8"), msg, _hash.sha256).hexdigest()

    # Sisipkan metadata aditif tanpa merusak struktur utama.
    try:
        payload_to_send = dict(slot_payload or {})
    except Exception:
        return False
    is_summary = payload_to_send.get("event") == "batch_completed"
    if "event" not in payload_to_send:
        payload_to_send["event"] = "slot_completed"
    if progress_str and "progress" not in payload_to_send:
        payload_to_send["progress"] = progress_str
    if "is_final" not in payload_to_send:
        payload_to_send["is_final"] = bool(is_summary)
    if "processed_at" not in payload_to_send and "completed_at" not in payload_to_send:
        try:
            from datetime import datetime as _dt
            payload_to_send["processed_at" if not is_summary else "completed_at"] = _dt.now(_tz).isoformat()
        except Exception:
            pass

    import httpx
    from urllib.parse import urlparse

    try:
        parsed = urlparse(_cb_url)
        if CALLBACK_ALLOWED_HOSTS and parsed.hostname and parsed.hostname.lower() not in CALLBACK_ALLOWED_HOSTS:
            print(f"[CHECK CALLBACK PROGRESSIVE] blocked SSRF host {parsed.hostname} not in allowlist")
            return False
    except Exception:
        pass

    def _log_attempt(payload: dict, status_code: int, resp_text: str, attempt: int, success: bool):
        try:
            entry = {
                "at": datetime.now(_tz).isoformat(),
                "batch_code": batch_code,
                "slot": payload.get("slot"),
                "event": payload.get("event"),
                "attempt": attempt,
                "status_code": status_code,
                "success": success,
                "response": resp_text[:500] if resp_text else "",
            }
            with open(CALLBACK_LOG_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            pass

    body_json = json.dumps(payload_to_send, ensure_ascii=False, separators=(",", ":"))
    ts = datetime.now(_tz).isoformat()
    sig = _sign(body_json, ts, _hmac_secret)
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": _api_key,
        "X-Signature": sig,
        "X-Timestamp": ts,
    }

    slot_no = payload_to_send.get("slot")
    tag = f"slot{slot_no}" if not is_summary else "summary"
    for attempt in range(1, CALLBACK_RETRIES + 1):
        try:
            with httpx.Client(timeout=httpx.Timeout(CALLBACK_TIMEOUT_SECONDS, connect=2.0)) as client:
                resp = client.post(_cb_url, content=body_json.encode("utf-8"), headers=headers)
            if 200 <= resp.status_code < 300:
                print(f"[CHECK CALLBACK PROGRESSIVE] {tag} OK {resp.status_code} attempt {attempt}")
                _log_attempt(payload_to_send, resp.status_code, resp.text, attempt, True)
                return True
            if resp.status_code in (401, 422):
                print(f"[CHECK CALLBACK PROGRESSIVE] {tag} {resp.status_code} no-retry: {resp.text[:200]}")
                _log_attempt(payload_to_send, resp.status_code, resp.text, attempt, False)
                return False
            print(f"[CHECK CALLBACK PROGRESSIVE] {tag} {resp.status_code} retry {attempt}/{CALLBACK_RETRIES}")
            _log_attempt(payload_to_send, resp.status_code, resp.text, attempt, False)
            if attempt < CALLBACK_RETRIES:
                time.sleep(CALLBACK_RETRY_BACKOFF_BASE ** attempt + (0.3 if attempt > 1 else 0))
        except Exception as e:
            print(f"[CHECK CALLBACK PROGRESSIVE] {tag} exception attempt {attempt}: {e}")
            _log_attempt(payload_to_send, 0, str(e)[:500], attempt, False)
            if attempt < CALLBACK_RETRIES:
                time.sleep(CALLBACK_RETRY_BACKOFF_BASE ** attempt)

    try:
        with open(CALLBACK_DEAD_LETTER_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps({"batch_code": batch_code, "slot": slot_no, "payload": payload_to_send, "at": datetime.now(_tz).isoformat()}, ensure_ascii=False) + "\n")
        print(f"[CHECK CALLBACK PROGRESSIVE] dead letter {tag} {batch_code}")
    except Exception:
        pass
    return False


def _dispatch_slot_callback(batch_code: str, slot_payload: dict, progress_str: str = "") -> None:
    """Submit non-blocking ke pool. TIDAK PERNAH raise (aman dipanggil dari loop OCR)."""
    try:
        snapshot = dict(slot_payload or {})
    except Exception:
        return
    try:
        _CALLBACK_EXECUTOR.submit(_send_slot_callback_task, batch_code, snapshot, progress_str)
    except Exception as e:
        logger.warning(f"[CHECK CALLBACK] submit failed slot={snapshot.get('slot')}: {e}")


def _process_batch_ocr(batch_code: str):
    """Background task: proses N slot serial, update batch.json tiap slot."""
    # Guard: jika batch folder sudah hilang (ter-replace oleh batch baru), abort
    bdir = _batch_dir(batch_code)
    if not bdir.exists():
        print(f"[CHECK OCR] abort {batch_code}: batch folder gone (replaced by newer batch)")
        return

    print(f"[CHECK OCR] START batch {batch_code}")
    odir = _batch_ocr_dir(batch_code)
    try:
        odir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"[CHECK OCR] failed mkdir ocr for {batch_code}: {e}")
        return

    # Double-check folder masih ada setelah mkdir
    if not bdir.exists():
        print(f"[CHECK OCR] abort {batch_code}: batch folder missing after mkdir check")
        return

    # Slot yang diproses = file aktual di disk (dinamis 4-5, sinkron Laravel)
    expected_slots = _discover_batch_slots(batch_code)
    if not expected_slots:
        print(f"[CHECK OCR] abort {batch_code}: no valid slot files found")
        _write_batch_json(batch_code, {
            "batch_code": batch_code,
            "overall_status": "failed",
            "progress": "0/0",
            "completed_count": 0,
            "failed_count": 0,
            "pending_count": 0,
            "started_at": datetime.now(JAKARTA_TZ).isoformat(),
            "updated_at": datetime.now(JAKARTA_TZ).isoformat(),
            "error": "no valid slot files found",
            "slots": [],
        })
        return
    total = len(expected_slots)

    now = datetime.now(JAKARTA_TZ).isoformat()

    # --- COMPARE_TEXT_SIMILARITY_PLAN (2026-09-12) + REVIEW_FLOW_COMPARE_FRONT_BACK_PLAN (2026-10-05): muat reference teks ---
    # Sumber: uploads/check/{batch}/compare_request.json (ditulis process_check_batch).
    # reference_text_front + _back > reference_text (fallback single) > reference_filename.
    # Tidak ada sama sekali → slot TANPA blok similarity (perilaku lama).
    ref_text_front: Optional[str] = None
    ref_text_back: Optional[str] = None
    ref_text_fallback: Optional[str] = None  # backward compat (lama tanpa sisi)
    ref_map: dict = {}  # FIX_WRONG_REFERENCE (2026-10-07): map slot -> teks acuan
    ref_source = "missing"
    try:
        cmp_req_path = _batch_dir(batch_code) / "compare_request.json"
        if cmp_req_path.exists():
            with open(cmp_req_path, "r", encoding="utf-8") as f:
                cmp_req = json.load(f)
            # Baru: per-sisi
            ref_text_front = (cmp_req.get("reference_text_front") or "").strip() or None
            ref_text_back = (cmp_req.get("reference_text_back") or "").strip() or None
            # Fallback: field lama (1 teks tanpa sisi)
            ref_text_fallback = (cmp_req.get("reference_text") or "").strip() or None
            # BARU (FIX_WRONG_REFERENCE 2026-10-07): map acuan per slot.
            # Bentuk: dict {"1": teks} / list [teks1, ...] / string JSON.
            ref_map = _normalize_ref_map(cmp_req.get("reference_texts"))
            if ref_text_front or ref_text_back:
                ref_source = "body_dual"
            elif ref_text_fallback:
                ref_source = "body_single"
            # Fallback file (existing logic, dipertahankan)
            ref_fn = cmp_req.get("reference_filename")
            if ref_text_front is None and ref_text_back is None and ref_text_fallback is None and ref_fn:
                from app.config import OCR_DIR as _OCR_DIR
                _ref_json = _OCR_DIR / f"{sanitize_filename_for_lookup(ref_fn)}.json"
                if _ref_json.exists():
                    try:
                        with open(_ref_json, "r", encoding="utf-8") as f:
                            ref_data = json.load(f)
                        ref_text_fallback = (ref_data.get("cleaned_text") or "").strip() or None
                        ref_source = "file" if ref_text_fallback else "missing"
                    except Exception as e:
                        logger.warning(f"[COMPARE] read reference json failed: {e}")
            # Log
            _rf = len([ln for ln in (ref_text_front or "").splitlines() if ln.strip()]) if ref_text_front else 0
            _rb = len([ln for ln in (ref_text_back or "").splitlines() if ln.strip()]) if ref_text_back else 0
            _rs = len([ln for ln in (ref_text_fallback or "").splitlines() if ln.strip()]) if ref_text_fallback else 0
            print(f"[COMPARE] ref_front_lines={_rf} ref_back_lines={_rb} ref_single_lines={_rs} slotmap={len(ref_map)} source={ref_source}")
        else:
            print(f"[COMPARE] reference missing (source={ref_source}) — similarity skipped, callback tetap jalan")
    except Exception as e:
        logger.warning(f"[COMPARE] load compare_request failed: {e}")
    # Alias backward compat: ref_text = fallback tunggal (dipakai summary guard lama)
    ref_text: Optional[str] = ref_text_fallback

    batch_data = {
        "batch_code": batch_code,
        "overall_status": "processing",
        "progress": f"0/{total}",
        "completed_count": 0,
        "failed_count": 0,
        "pending_count": total,
        "started_at": now,
        "updated_at": now,
        "slots": [{"slot": s, "status": "pending"} for s in expected_slots],
    }
    _write_batch_json(batch_code, batch_data)
    total_start = time.time()

    def _worker(slot: int) -> dict:
        img_path = _slot_png_path(batch_code, slot)
        if not img_path or not img_path.exists():
            res = {
                "slot": slot,
                "status": "failed",
                "is_empty": True,
                "error": "image file not found",
                "filename": f"slot{slot}.png",
                "batch_code": batch_code,
                "side": "unknown",
            }
            odir.mkdir(parents=True, exist_ok=True)
            with open(odir / f"slot{slot}.json", "w", encoding="utf-8") as f:
                json.dump(res, f, indent=2, ensure_ascii=False)
            return res

        try:
            # Deteksi sisi dari filename (REVIEW_FLOW_COMPARE_FRONT_BACK_PLAN)
            from app.compare_text.side_detect import detect_side
            slot_side = detect_side(img_path.name)

            # FIX_WRONG_REFERENCE (2026-10-07): pilih acuan via helper + catat sumber.
            slot_ref_text, ref_src = _select_slot_ref(
                slot, slot_side, ref_text_front, ref_text_back, ref_text_fallback, ref_map,
            )
            try:
                _rx = (slot_ref_text or "")[:60].replace("\n", " ")
                print(f"[COMPARE] batch={batch_code} slot={slot} side={slot_side} "
                      f"ref_source={ref_src} ref_len={len(slot_ref_text or '')} ref_head={_rx!r}")
                if ref_src == "single" and (ref_text_front or ref_text_back or ref_map):
                    print(f"[COMPARE] WARNING batch={batch_code} slot={slot} jatuh ke acuan "
                          f"tunggal padahal acuan sisi/slot tersedia — risiko salah pasangan!")
            except Exception:
                pass

            res = _process_single_slot(batch_code, slot, img_path, reference_text=slot_ref_text)
            res["side"] = slot_side  # TAMBAH field side ke result
            res["ref_source"] = ref_src  # FIX_WRONG_REFERENCE: sumber acuan (diagnostik, aditif)
            # Persist side ke slot JSON (overwrite yang ditulis _process_single_slot)
            try:
                with open(odir / f"slot{slot}.json", "w", encoding="utf-8") as f:
                    json.dump(res, f, indent=2, ensure_ascii=False)
            except Exception:
                pass
            return res
        except Exception as e:
            logger.error(f"[CHECK OCR] slot{slot} {batch_code} exception: {e}", exc_info=True)
            print(f"[CHECK OCR] slot{slot} {batch_code} exception: {e}")
            res = {
                "slot": slot,
                "status": "failed",
                "is_empty": True,
                "error": str(e)[:500],
                "filename": img_path.name if img_path else f"slot{slot}.png",
                "batch_code": batch_code,
                "side": "unknown",
            }
            with open(odir / f"slot{slot}.json", "w", encoding="utf-8") as f:
                json.dump(res, f, indent=2, ensure_ascii=False)
            return res

    import concurrent.futures
    from app.config import CHECK_OCR_MAX_WORKERS

    workers = max(1, min(CHECK_OCR_MAX_WORKERS, len(expected_slots)))
    print(f"[CHECK OCR] Processing {len(expected_slots)} slots with {workers} workers (parallel)")

    results_map: dict[int, dict] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_slot = {executor.submit(_worker, slot): slot for slot in expected_slots}
        for future in concurrent.futures.as_completed(future_to_slot):
            slot = future_to_slot[future]
            try:
                res = future.result()
            except Exception as e:
                res = {
                    "slot": slot,
                    "status": "failed",
                    "is_empty": True,
                    "error": str(e)[:500],
                    "batch_code": batch_code,
                    "side": "unknown",
                }
            results_map[slot] = res

            # ★ PROGRESSIVE CALLBACK: dispatch hasil slot SEGERA via thread
            # terpisah (Non-Blocking). Loop OCR langsung lanjut slot berikut.
            if CHECK_OCR_CALLBACK_ENABLED and _is_progressive_enabled():
                print(f"[CHECK CALLBACK] Dispatching slot {slot} callback in background...")
                _dispatch_slot_callback(batch_code, res, f"{len(results_map)}/{total}")

            # update batch.json progress tiap slot selesai
            completed = sum(1 for r in results_map.values() if r.get("status") == "success")
            failed = sum(1 for r in results_map.values() if r.get("status") == "failed")
            pending = total - len(results_map)
            slots_progress = [results_map.get(s, {"slot": s, "status": "pending"}) for s in expected_slots]
            batch_data.update(
                {
                    "updated_at": datetime.now(JAKARTA_TZ).isoformat(),
                    "completed_count": completed,
                    "failed_count": failed,
                    "pending_count": pending,
                    "progress": f"{len(results_map)}/{total}",
                    "overall_status": "processing" if pending > 0 else ("completed" if failed == 0 else ("partial_failed" if failed < total else "failed")),
                    "slots": slots_progress,
                }
            )
            _write_batch_json(batch_code, batch_data)

    total_elapsed = round(time.time() - total_start, 3)
    final_slots: List[dict] = []
    for slot in expected_slots:
        sj = odir / f"slot{slot}.json"
        if sj.exists():
            try:
                with open(sj, "r", encoding="utf-8") as f:
                    final_slots.append(json.load(f))
            except Exception:
                final_slots.append({"slot": slot, "status": "failed", "batch_code": batch_code})
        else:
            final_slots.append({"slot": slot, "status": "failed", "batch_code": batch_code})
    completed = sum(1 for r in final_slots if r.get("status") == "success")
    failed = sum(1 for r in final_slots if r.get("status") == "failed")
    overall = "completed" if failed == 0 else ("partial_failed" if failed < total else "failed")
    batch_final = {
        "batch_code": batch_code,
        "overall_status": overall,
        "progress": f"{total}/{total}",
        "completed_count": completed,
        "failed_count": failed,
        "pending_count": 0,
        "started_at": batch_data.get("started_at"),
        "updated_at": datetime.now(JAKARTA_TZ).isoformat(),
        "completed_at": datetime.now(JAKARTA_TZ).isoformat(),
        "processing_time_total_seconds": total_elapsed,
        "slots": final_slots,
    }

    # --- Ringkasan batch per sisi (REVIEW_FLOW_COMPARE_FRONT_BACK_PLAN 2026-10-05) ---
    # Backward compat: bila hanya ada ref tunggal lama, compare_batch lama tetap valid,
    # tapi kita selalu pakai compare_batch_by_side (v2) bila ada reference apapun.
    try:
        from app.config import SIMILARITY_ENABLED as _SIM_EN, SIMILARITY_PASS_THRESHOLD as _SIM_THR
    except ImportError:
        _SIM_EN, _SIM_THR = True, 0.85
    if _SIM_EN and (ref_text_front or ref_text_back or ref_text_fallback):
        try:
            from app.compare_text.similarity import compare_batch_by_side
            _summary = compare_batch_by_side(final_slots, _SIM_THR)
            batch_final["compare_summary"] = _summary
            _front = _summary.get("front", {})
            _back = _summary.get("back", {})
            print(
                f"[COMPARE] DONE batch={batch_code} "
                f"front(pass={_front.get('pass_count')}, fail={_front.get('fail_count')}) "
                f"back(pass={_back.get('pass_count')}, fail={_back.get('fail_count')}) "
                f"combined={_summary.get('combined_overall')} threshold={_SIM_THR}"
            )
        except Exception as e:
            logger.warning(f"[COMPARE] summary failed for {batch_code}: {e}")
    # Cek lagi folder masih ada sebelum tulis final (race dengan cleanup batch baru)
    if not _batch_dir(batch_code).exists():
        print(f"[CHECK OCR] abort finalize {batch_code}: batch folder gone before final write (replaced)")
        return
    try:
        _write_batch_json(batch_code, batch_final)
    except Exception as e:
        print(f"[CHECK OCR] failed write batch.json final for {batch_code}: {e}")
        logger.warning(f"[CHECK OCR] final write failed {batch_code}: {e}")
        return
    print(f"[CHECK OCR] DONE batch {batch_code} overall={overall} {total_elapsed:.2f}s success={completed} failed={failed}")

    # --- Progressive summary callback (event batch_completed, non-blocking) ---
    # Dikirim setelah compare_batch_by_side selesai. Payload aditif, tidak
    # mengubah struktur slot. Di-skip bila progressive mode nonaktif.
    try:
        if CHECK_OCR_CALLBACK_ENABLED and _is_progressive_enabled():
            summary_payload = {
                "event": "batch_completed",
                "batch_code": batch_code,
                "overall_status": overall,
                "progress": f"{total}/{total}",
                "completed_count": completed,
                "failed_count": failed,
                "processing_time_total_seconds": total_elapsed,
                "is_final": True,
                "compare_summary": batch_final.get("compare_summary", {}),
                "completed_at": datetime.now(JAKARTA_TZ).isoformat(),
            }
            print(f"[CHECK CALLBACK] Dispatching batch summary in background...")
            _dispatch_slot_callback(batch_code, summary_payload, f"{total}/{total}")
    except Exception as e:
        logger.warning(f"[CHECK CALLBACK] summary dispatch failed for {batch_code}: {e}")

    # --- Callback per label ke Laravel (PUSH JSON per slot, N POST) ---
    try:
        # Cek enabled
        try:
            from app.config import CHECK_OCR_CALLBACK_ENABLED as _cb_enabled, CHECK_OCR_CALLBACK_URL as _cb_url, LARAVEL_API_KEY as _api_key, LARAVEL_HMAC_SECRET as _hmac_secret
        except ImportError:
            _cb_enabled = CHECK_OCR_CALLBACK_ENABLED
            _cb_url = CHECK_OCR_CALLBACK_URL
            from app.config import LARAVEL_API_KEY as _api_key
            try:
                from app.config import LARAVEL_HMAC_SECRET as _hmac_secret
            except ImportError:
                _hmac_secret = ""
        if not _cb_enabled or not _cb_url or not _api_key or not _hmac_secret:
            print(f"[CHECK CALLBACK] skipped for {batch_code}: enabled={_cb_enabled} url={bool(_cb_url)} api_key={bool(_api_key)} hmac={bool(_hmac_secret)}")
        else:
            # Validasi allowlist & prepare retry config
            try:
                from app.config import CALLBACK_TIMEOUT_SECONDS, CALLBACK_RETRIES, CALLBACK_RETRY_BACKOFF_BASE, CALLBACK_ALLOWED_HOSTS, CALLBACK_LOG_FILE, CALLBACK_DEAD_LETTER_FILE
            except ImportError:
                CALLBACK_TIMEOUT_SECONDS = 5.0
                CALLBACK_RETRIES = 3
                CALLBACK_RETRY_BACKOFF_BASE = 2.0
                CALLBACK_ALLOWED_HOSTS = {"127.0.0.1", "localhost"}
                from app.config import UPLOAD_DIR as _ud
                CALLBACK_LOG_FILE = _ud / "callback_log.jsonl"
                CALLBACK_DEAD_LETTER_FILE = _ud / "callback_dead_letter.jsonl"

            # Helper sign (reuse app/callback/signer jika ada)
            try:
                from app.callback.signer import sign_payload  # type: ignore

                def _sign(body: str, ts: str, secret: str) -> str:
                    return sign_payload(body, ts, secret)
            except ImportError:
                import hmac as _hmac
                import hashlib as _hash

                def _sign(body: str, ts: str, secret: str) -> str:
                    # Fallback: HMAC SHA256 of body+timestamp
                    msg = (body + ts).encode("utf-8")
                    return _hmac.new(secret.encode("utf-8"), msg, _hash.sha256).hexdigest()

            import httpx
            from urllib.parse import urlparse

            parsed = urlparse(_cb_url)
            if CALLBACK_ALLOWED_HOSTS and parsed.hostname and parsed.hostname.lower() not in CALLBACK_ALLOWED_HOSTS:
                print(f"[CHECK CALLBACK] blocked SSRF host {parsed.hostname} not in allowlist {CALLBACK_ALLOWED_HOSTS}")
            else:
                def _log_attempt(payload: dict, status_code: int, resp_text: str, attempt: int, success: bool):
                    try:
                        entry = {
                            "at": datetime.now(JAKARTA_TZ).isoformat(),
                            "batch_code": batch_code,
                            "slot": payload.get("slot"),
                            "attempt": attempt,
                            "status_code": status_code,
                            "success": success,
                            "response": resp_text[:500] if resp_text else "",
                        }
                        with open(CALLBACK_LOG_FILE, "a", encoding="utf-8") as f:
                            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    except Exception:
                        pass

                def _post_one_slot(slot_payload: dict) -> bool:
                    body_json = json.dumps(slot_payload, ensure_ascii=False, separators=(",", ":"))
                    ts = datetime.now(JAKARTA_TZ).isoformat()
                    sig = _sign(body_json, ts, _hmac_secret)
                    # Support both with and without sha256= prefix — Laravel accepts both
                    headers = {
                        "Content-Type": "application/json",
                        "X-API-Key": _api_key,
                        "X-Signature": sig,
                        "X-Timestamp": ts,
                    }
                    for attempt in range(1, CALLBACK_RETRIES + 1):
                        try:
                            with httpx.Client(timeout=httpx.Timeout(CALLBACK_TIMEOUT_SECONDS, connect=2.0)) as client:
                                resp = client.post(_cb_url, content=body_json.encode("utf-8"), headers=headers)
                            ok = 200 <= resp.status_code < 300
                            # No-retry untuk 401/422
                            if resp.status_code in (401, 422):
                                print(f"[CHECK CALLBACK] slot{slot_payload.get('slot')} {resp.status_code} no-retry: {resp.text[:200]}")
                                _log_attempt(slot_payload, resp.status_code, resp.text, attempt, False)
                                return False
                            if ok:
                                print(f"[CHECK CALLBACK] slot{slot_payload.get('slot')} OK {resp.status_code} attempt {attempt}")
                                _log_attempt(slot_payload, resp.status_code, resp.text, attempt, True)
                                return True
                            print(f"[CHECK CALLBACK] slot{slot_payload.get('slot')} {resp.status_code} retry {attempt}/{CALLBACK_RETRIES}: {resp.text[:200]}")
                            _log_attempt(slot_payload, resp.status_code, resp.text, attempt, False)
                            if attempt < CALLBACK_RETRIES:
                                time.sleep(CALLBACK_RETRY_BACKOFF_BASE ** attempt + (0.3 if attempt > 1 else 0))
                                continue
                            return False
                        except Exception as e:
                            print(f"[CHECK CALLBACK] slot{slot_payload.get('slot')} exception attempt {attempt}: {e}")
                            _log_attempt(slot_payload, 0, str(e)[:500], attempt, False)
                            if attempt < CALLBACK_RETRIES:
                                time.sleep(CALLBACK_RETRY_BACKOFF_BASE ** attempt)
                                continue
                            return False
                    return False

                # Kirim N slot sequential (hemat, tidak burst) — MODE LAMA.
                # Di-guard: dilewati bila progressive mode aktif agar tidak duplikat
                # (slot sudah dikirim per-slot via _dispatch_slot_callback di loop).
                if CHECK_OCR_CALLBACK_ENABLED and _is_progressive_enabled():
                    print(f"[CHECK CALLBACK] progressive mode — skip bulk loop for {batch_code} ({len(final_slots)} slots already dispatched)")
                else:
                    for slot_data in final_slots:
                        ok = _post_one_slot(slot_data)
                        if not ok:
                            try:
                                with open(CALLBACK_DEAD_LETTER_FILE, "a", encoding="utf-8") as f:
                                    f.write(json.dumps({"batch_code": batch_code, "slot": slot_data.get("slot"), "payload": slot_data, "error": "callback failed after retries", "at": datetime.now(JAKARTA_TZ).isoformat()}, ensure_ascii=False) + "\n")
                                print(f"[CHECK CALLBACK] dead letter slot{slot_data.get('slot')} {batch_code}")
                            except Exception:
                                pass
                        time.sleep(0.1)
                    print(f"[CHECK CALLBACK] Done {len(final_slots)} POST to {_cb_url} for {batch_code}")
    except Exception as e:
        print(f"[CHECK CALLBACK] skip/error for {batch_code}: {e}")
        logger.warning(f"[CHECK CALLBACK] error for {batch_code}: {e}", exc_info=True)


# ---------------------------------------------------------------------------
# Visual Compare Mode — worker + endpoint (PLAN_VISUAL_COMPARE_MODE)
# Pengganti OCR batch selama mode komparasi visual aktif.
# Reference = gambar acuan di uploads\ (label-* terbaru / eksplisit).
# Check    = slot files di uploads\check\{batch}\ (4 atau 5).
# ---------------------------------------------------------------------------

def _resolve_reference_image(reference_filename: Optional[str] = None) -> Path:
    """
    Resolve gambar acuan (reference) di folder uploads\\.

    - Jika reference_filename diisi: pakai file itu (hanya basename, anti traversal).
    - Jika None: pakai file label-* TERBARU di uploads\\ (by mtime).
    - Validasi: ada, file, ekstensi PNG/JPEG, di dalam UPLOAD_DIR.

    Raises:
        FileNotFoundError: jika tidak ada gambar acuan yang cocok.
        ValueError: jika ekstensi tidak didukung.
    """
    try:
        from app.config import UPLOAD_DIR
    except ImportError:
        UPLOAD_DIR = (Path(__file__).resolve().parent.parent / "uploads").resolve()

    if reference_filename:
        safe_name = sanitize_filename_for_lookup(os.path.basename(reference_filename))
        if not safe_name or safe_name != os.path.basename(reference_filename):
            raise ValueError(f"Nama file reference tidak valid: {reference_filename}")
        ref_path = UPLOAD_DIR / safe_name
        if not is_safe_path(UPLOAD_DIR, ref_path):
            raise ValueError(f"Path reference tidak valid: {reference_filename}")
        if ref_path.suffix.lower() not in ALLOWED_EXTS:
            raise ValueError(f"Format reference tidak didukung (hanya PNG/JPEG): {safe_name}")
        if not ref_path.exists() or not ref_path.is_file():
            raise FileNotFoundError(f"Gambar acuan tidak ditemukan di uploads\\: {safe_name}")
        return ref_path

    # Otomatis: label-* terbaru
    candidates: List[Path] = []
    for ext in (".png", ".jpg", ".jpeg"):
        candidates.extend([p for p in UPLOAD_DIR.glob(f"label-*{ext}") if p.is_file()])
    if not candidates:
        raise FileNotFoundError(
            "Tidak ada gambar acuan (label-*) di folder uploads\\. "
            "Upload dulu via POST /api/test/status atau isi reference_filename."
        )
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _process_batch_compare(batch_code: str, reference_filename: Optional[str] = None) -> dict:
    """
    Background task Visual Compare Mode: 1 reference vs N check images.

    1. Resolve check images (urut slot) + reference (eksplisit / label-* terbaru).
    2. compare_images_with_reference() → simpan compare_result.json di folder batch.
    3. Print ringkasan ke console (sementara).
    4. Callback hasil JSON ke Laravel (best-effort via app.compare.callback).

    Returns: dict hasil compare (status success/failed). Tidak raise.
    """
    print("=" * 60)
    print(f"  [COMPARE] START batch={batch_code} ref={reference_filename or '(auto label-* terbaru)'}")
    print("=" * 60)
    try:
        expected_slots = _discover_batch_slots(batch_code)
        if len(expected_slots) not in CHECK_ALLOWED_COUNTS:
            raise FileNotFoundError(
                f"Batch {batch_code}: ditemukan {len(expected_slots)} slot valid "
                f"(butuh 4 atau 5). Pastikan gambar sudah masuk via POST /api/check/captures."
            )
        check_paths = []
        for slot in expected_slots:
            p = _slot_png_path(batch_code, slot)
            if not p or not p.exists():
                raise FileNotFoundError(f"File slot {slot} hilang di batch {batch_code}")
            check_paths.append(str(p))

        ref_path = _resolve_reference_image(reference_filename)

        try:
            from app.config import GEMINI_API_KEY, GEMINI_MODEL, GEMINI_COMPARE_TIMEOUT
        except ImportError:
            GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
            GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
            GEMINI_COMPARE_TIMEOUT = float(os.getenv("GEMINI_COMPARE_TIMEOUT", "30"))

        from app.compare.service import compare_images_with_reference

        result = compare_images_with_reference(
            reference_image_path=str(ref_path),
            check_image_paths=check_paths,
            api_key=GEMINI_API_KEY,
            model=GEMINI_MODEL,
            timeout=GEMINI_COMPARE_TIMEOUT,
            batch_code=batch_code,
            save_result=True,
            save_dir=str(_batch_dir(batch_code)),
        )

        # --- Console log sementara: ringkasan JAVA-friendly ---
        total = len(result.get("results", []))
        matches = sum(1 for r in result.get("results", []) if r.get("is_match"))
        print("=" * 60)
        print(f"  [COMPARE] batch={batch_code} ref={ref_path.name}")
        print(f"  Model: {result.get('model')} | Slots: {total} | Time: {result.get('processing_time_seconds')}s")
        print(f"  SUMMARY: {matches}/{total} MATCH, {total - matches}/{total} MISMATCH")
        for r in result.get("results", []):
            flag = "MATCH" if r.get("is_match") else "MISMATCH"
            print(f"    Slot {r.get('slot')}: {flag} ({r.get('confidence_score')}) — {(r.get('notes') or '')[:100]}")
        print("=" * 60)

        # --- Callback ke Laravel (best-effort) ---
        try:
            from app.compare.callback import send_compare_callback

            cb = send_compare_callback(result)
            result["callback"] = cb
        except Exception as e:
            logger.warning(f"[COMPARE] callback sender error for {batch_code}: {e}", exc_info=True)
            print(f"[COMPARE CALLBACK] sender error: {e} (compare tetap sukses)")
            result["callback"] = {"success": False, "error": str(e)[:300]}

        return result

    except Exception as e:
        logger.error(f"[COMPARE] worker failed for {batch_code}: {e}", exc_info=True)
        print(f"[COMPARE] FAILED batch={batch_code}: {e}")
        return {"batch_code": batch_code, "status": "failed", "error": str(e), "results": []}


def _extract_compare_params(request: Request, body: Optional[dict] = None) -> tuple:
    """Ambil (batch_code, reference_filename) dari JSON body / form / query."""
    batch_code = None
    reference_filename = None
    if isinstance(body, dict):
        batch_code = body.get("batch_code") or body.get("batchCode")
        reference_filename = body.get("reference_filename") or body.get("referenceFilename")
    if not batch_code:
        batch_code = request.query_params.get("batch_code") or request.query_params.get("batchCode")
    if not reference_filename:
        reference_filename = request.query_params.get("reference_filename") or request.query_params.get("referenceFilename")
    return batch_code, reference_filename


# --- ENDPOINT: POST /api/check/compare — trigger langsung Visual Compare ---
@router.post("/compare")
async def compare_check_batch(
    request: Request,
    background_tasks: BackgroundTasks,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    x_api_key_lower: Optional[str] = Header(None, alias="x-api-key"),
):
    """
    Trigger Visual Compare untuk 1 batch: reference (uploads\\) vs N check images.

    Body JSON / form / query: {batch_code (wajib), reference_filename (opsional)}.
    Response 202 langsung; hasil menyusul di compare_result.json + callback + console.
    Lihat hasil: GET /api/check/compare/{batch_code}.
    """
    # [OCR-REENABLE 2026-09-11 / PLAN_REVERT_TO_OCR] — Gemini OFF sementara, OCR ON.
    # Sisa fungsi dibiarkan utuh agar mudah diaktifkan lagi (lihat §10 plan revert).
    try:
        from app.config import GEMINI_COMPARE_ENABLED as _COMPARE_ENABLED
    except ImportError:
        import os as _os
        _COMPARE_ENABLED = _os.getenv("GEMINI_COMPARE_ENABLED", "false").lower() == "true"
    if not _COMPARE_ENABLED:
        return JSONResponse(
            status_code=503,
            content={
                "success": False,
                "message": "Visual compare disabled (OCR mode active). Gunakan POST /api/check/process untuk OCR.",
                "error": "compare_disabled",
            },
        )
    # 1. JSON body (opsional)
    body = None
    try:
        ctype = request.headers.get("content-type", "") or ""
        if "application/json" in ctype.lower():
            body = await request.json()
    except Exception:
        body = None
    # 1b. Form fallback
    if not isinstance(body, dict):
        try:
            form = await request.form()
            if form and (form.get("batch_code") or form.get("reference_filename")):
                body = {"batch_code": form.get("batch_code"), "reference_filename": form.get("reference_filename")}
        except Exception:
            pass
    batch_code, reference_filename = _extract_compare_params(request, body)

    # 2. API key (validasi sama seperti /captures)
    effective_key = x_api_key or x_api_key_lower
    if not effective_key:
        for k in ("X-API-Key", "x-api-key", "X-Api-Key"):
            v = request.headers.get(k)
            if v:
                effective_key = v
                break
        if not effective_key:
            effective_key = request.query_params.get("api_key")
    if not _validate_api_key(effective_key):
        return JSONResponse(status_code=401, content={"success": False, "message": "Invalid API key", "error": "unauthorized"})

    # 3. Validasi batch_code
    if not batch_code or not isinstance(batch_code, str):
        return JSONResponse(status_code=422, content={"success": False, "message": "batch_code is required", "error": "validation_failed"})
    batch_code = batch_code.strip()
    if not batch_code or len(batch_code) > 64 or not BATCH_CODE_RE.match(batch_code):
        return JSONResponse(status_code=422, content={"success": False, "message": "Invalid batch_code format. Must match ^[A-Za-z0-9\\-_]+$ and 1..64 chars", "error": "validation_failed"})
    safe_batch = sanitize_filename_for_lookup(batch_code)
    if safe_batch != batch_code:
        return JSONResponse(status_code=422, content={"success": False, "message": "Invalid batch_code (contains unsafe characters)", "error": "validation_failed"})
    bdir = _batch_dir(batch_code)
    if not is_safe_path(CHECK_UPLOAD_ROOT, bdir):
        return JSONResponse(status_code=422, content={"success": False, "message": "Invalid batch_code path", "error": "validation_failed"})
    if not bdir.exists() or not bdir.is_dir():
        return JSONResponse(status_code=404, content={"success": False, "message": f"Batch not found: {batch_code}", "error": "batch_not_found"})

    # 4. Validasi slot 4-5
    expected_slots = _discover_batch_slots(batch_code)
    if len(expected_slots) not in CHECK_ALLOWED_COUNTS:
        return JSONResponse(
            status_code=422,
            content={"success": False, "message": f"Batch incomplete: found {len(expected_slots)} valid slot files (expected 4 or 5)", "error": "batch_incomplete", "found": len(expected_slots), "expected_counts": list(CHECK_ALLOWED_COUNTS), "batch_code": batch_code},
        )

    # 5. Validasi reference (gagal cepat sebelum queue)
    try:
        ref_path = _resolve_reference_image(reference_filename if isinstance(reference_filename, str) else None)
    except FileNotFoundError as e:
        return JSONResponse(status_code=404, content={"success": False, "message": str(e), "error": "reference_not_found"})
    except ValueError as e:
        return JSONResponse(status_code=422, content={"success": False, "message": str(e), "error": "validation_failed"})

    # 6. Queue background worker
    background_tasks.add_task(_process_batch_compare, batch_code, reference_filename if isinstance(reference_filename, str) else None)
    print(f"[COMPARE] triggered for {batch_code} ref={ref_path.name} ({len(expected_slots)} slots)")
    return JSONResponse(
        status_code=202,
        content={
            "success": True,
            "batch_code": batch_code,
            "reference_image": ref_path.name,
            "total_slots": len(expected_slots),
            "slots": expected_slots,
            "status": "processing",
            "message": f"Visual compare started: 1 reference vs {len(expected_slots)} images",
            "poll_url": f"/api/check/compare/{batch_code}",
            "retry_after_seconds": 2,
        },
        headers={"Cache-Control": "no-store", "Pragma": "no-cache", "Expires": "0"},
    )


# --- ENDPOINT: GET /api/check/compare/{batch_code} — viewer sementara (web) ---
@router.get("/compare/{batch_code}")
async def get_compare_result(batch_code: str, request: Request, x_api_key: Optional[str] = Header(None, alias="X-API-Key")):
    """
    Viewer sementara hasil Visual Compare (web): baca compare_result.json.

    200 = isi compare_result.json. 404 = batch / hasil belum ada (compare dulu).
    """
    try:
        from app.security import check_get_rate_limit

        allowed, wait, remaining = check_get_rate_limit()
        if not allowed:
            wait_int = int(wait) + 1
            return JSONResponse(
                status_code=429,
                content={"success": False, "message": f"GET rate limit exceeded. Retry after {wait_int}s", "error": "too_many_requests", "retry_after_seconds": wait_int},
                headers={"Retry-After": str(wait_int)},
            )
    except Exception:
        pass
    effective_key = x_api_key or request.headers.get("X-API-Key") or request.headers.get("x-api-key") or request.query_params.get("api_key")
    if CHECK_API_KEY and effective_key:
        if not _validate_api_key(effective_key):
            return JSONResponse(status_code=401, content={"success": False, "message": "Invalid API key", "error": "unauthorized"})
    if not batch_code or len(batch_code) > 64 or not BATCH_CODE_RE.match(batch_code):
        return JSONResponse(status_code=422, content={"success": False, "message": "Invalid batch_code", "error": "validation_failed"})
    safe_batch = sanitize_filename_for_lookup(batch_code)
    if safe_batch != batch_code:
        return JSONResponse(status_code=422, content={"success": False, "message": "Invalid batch_code", "error": "validation_failed"})
    bdir = _batch_dir(batch_code)
    if not is_safe_path(CHECK_UPLOAD_ROOT, bdir):
        return JSONResponse(status_code=422, content={"success": False, "message": "Invalid batch_code path", "error": "validation_failed"})
    if not bdir.exists():
        return JSONResponse(status_code=404, content={"success": False, "message": f"Batch not found: {batch_code}", "error": "batch_not_found"}, headers={"Cache-Control": "no-store"})
    result_path = bdir / "compare_result.json"
    if not result_path.exists() or not result_path.is_file():
        return JSONResponse(
            status_code=404,
            content={"success": False, "batch_code": batch_code, "message": f"Compare result not found for batch {batch_code}. Trigger POST /api/check/compare or /api/check/process first.", "error": "not_found", "poll_url": f"/api/check/compare/{batch_code}"},
            headers={"Cache-Control": "no-store"},
        )
    try:
        with open(result_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return JSONResponse(status_code=200, content={"success": True, **data}, headers={"Cache-Control": "no-store", "Pragma": "no-cache", "Expires": "0"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "message": str(e), "error": "internal_error"})


# --- ENDPOINT: POST /api/check/process (alias /ocr/process for backward compat) ---
@router.post("/ocr/process")
@router.post("/process")
async def process_check_batch(
    request: Request,
    background_tasks: BackgroundTasks,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    x_api_key_lower: Optional[str] = Header(None, alias="x-api-key"),
):
    # support JSON body, Form, query param — manual extract (no Form param to allow JSON)
    batch_code: Optional[str] = None
    reference_text: Optional[str] = None
    reference_text_front: Optional[str] = None
    reference_text_back: Optional[str] = None
    reference_filename: Optional[str] = None
    # FIX_WRONG_REFERENCE (2026-10-07): map acuan per slot (dict/list/str-JSON).
    reference_texts = None
    # 1. Try JSON body (for Laravel CheckOcrClient which sends JSON)
    try:
        ctype = request.headers.get("content-type", "") or ""
        if "application/json" in ctype.lower():
            body = await request.json()
            if isinstance(body, dict):
                batch_code = body.get("batch_code") or body.get("batchCode")
                # COMPARE_TEXT_SIMILARITY_PLAN (2026-09-12): reference teks (opsional).
                reference_text = body.get("reference_text") or body.get("referenceText")
                # REVIEW_FLOW_COMPARE_FRONT_BACK_PLAN (2026-10-05): reference per sisi.
                reference_text_front = body.get("reference_text_front") or body.get("referenceTextFront")
                reference_text_back = body.get("reference_text_back") or body.get("referenceTextBack")
                reference_filename = body.get("reference_filename") or body.get("referenceFilename")
                # FIX_WRONG_REFERENCE (2026-10-07): map acuan per slot.
                reference_texts = body.get("reference_texts") or body.get("referenceTexts")
    except Exception:
        pass
    # 2. Try query param
    if not batch_code:
        batch_code = request.query_params.get("batch_code") or request.query_params.get("batchCode")
    if not reference_text:
        reference_text = request.query_params.get("reference_text") or request.query_params.get("referenceText")
    if not reference_text_front:
        reference_text_front = request.query_params.get("reference_text_front")
    if not reference_text_back:
        reference_text_back = request.query_params.get("reference_text_back")
    if not reference_filename:
        reference_filename = request.query_params.get("reference_filename") or request.query_params.get("referenceFilename")
    # 3. Try form data (multipart/form-data or x-www-form-urlencoded)
    if not batch_code:
        try:
            form = await request.form()
            # form may contain batch_code or batchCode
            batch_code = form.get("batch_code") or form.get("batchCode")
        except Exception:
            pass
    if not reference_text or not reference_filename or not reference_text_front or not reference_text_back or reference_texts is None:
        try:
            form = await request.form()
            if not reference_text:
                reference_text = form.get("reference_text") or form.get("referenceText")
            if not reference_text_front:
                reference_text_front = form.get("reference_text_front")
            if not reference_text_back:
                reference_text_back = form.get("reference_text_back")
            if not reference_filename:
                reference_filename = form.get("reference_filename") or form.get("referenceFilename")
            # FIX_WRONG_REFERENCE (2026-10-07): map acuan per slot via form.
            if reference_texts is None:
                reference_texts = form.get("reference_texts")
        except Exception:
            pass
    # API key
    effective_key = x_api_key or x_api_key_lower
    if not effective_key:
        for k in ("X-API-Key", "x-api-key", "X-Api-Key"):
            v = request.headers.get(k)
            if v:
                effective_key = v
                break
        if not effective_key:
            effective_key = request.query_params.get("api_key")
    if not _validate_api_key(effective_key):
        return JSONResponse(status_code=401, content={"success": False, "message": "Invalid API key", "error": "unauthorized"})
    # Validasi batch_code
    if not batch_code or not isinstance(batch_code, str):
        return JSONResponse(status_code=422, content={"success": False, "message": "batch_code is required", "error": "validation_failed"})
    batch_code = batch_code.strip()
    if not batch_code or len(batch_code) > 64 or not BATCH_CODE_RE.match(batch_code):
        return JSONResponse(status_code=422, content={"success": False, "message": "Invalid batch_code format. Must match ^[A-Za-z0-9\\-_]+$ and 1..64 chars", "error": "validation_failed"})
    safe_batch = sanitize_filename_for_lookup(batch_code)
    if safe_batch != batch_code:
        return JSONResponse(status_code=422, content={"success": False, "message": "Invalid batch_code (contains unsafe characters)", "error": "validation_failed"})
    bdir = _batch_dir(batch_code)
    if not is_safe_path(CHECK_UPLOAD_ROOT, bdir):
        return JSONResponse(status_code=422, content={"success": False, "message": "Invalid batch_code path", "error": "validation_failed"})
    if not bdir.exists() or not bdir.is_dir():
        return JSONResponse(status_code=404, content={"success": False, "message": f"Batch not found: {batch_code}", "error": "batch_not_found"})
    # cek N files lengkap (dinamis 4-5, sinkron Laravel)
    expected_slots = _discover_batch_slots(batch_code)
    valid_files: List[Path] = []
    for slot in expected_slots:
        p = _slot_png_path(batch_code, slot)
        if p and p.exists():
            valid_files.append(p)
    if len(valid_files) not in CHECK_ALLOWED_COUNTS:
        return JSONResponse(
            status_code=422,
            content={"success": False, "message": f"Batch incomplete: found {len(valid_files)} valid slot files (expected 4 or 5)", "error": "batch_incomplete", "found": len(valid_files), "expected_counts": list(CHECK_ALLOWED_COUNTS), "batch_code": batch_code},
        )
    # rate limit Process (opsional, fallback jika belum ada)
    try:
        from app.security import check_process_rate_limit

        allowed, wait, remaining = check_process_rate_limit()
        if not allowed:
            return _check_process_429(wait)
    except ImportError:
        pass
    except Exception:
        pass
    # cek apakah sudah processing/completed (idempotency)
    force = request.query_params.get("force") == "true"
    existing = _read_batch_json(batch_code)
    if existing and not force:
        status = existing.get("overall_status")
        started = existing.get("started_at")
        if status == "processing":
            try:
                st = datetime.fromisoformat(started) if started else None
                if st:
                    # handle naive vs aware
                    if st.tzinfo is None:
                        st = st.replace(tzinfo=JAKARTA_TZ)
                    age = (datetime.now(JAKARTA_TZ) - st).total_seconds()
                    if age < 60:
                        return JSONResponse(
                            status_code=202,
                            content={
                                "success": True,
                                "batch_code": batch_code,
                                "status": "processing",
                                "overall_status": "processing",
                                "already_processing": True,
                                "message": "OCR already in progress, please poll /api/check/ocr/{batch_code}",
                                "poll_url": f"/api/check/ocr/{batch_code}",
                                "retry_after_seconds": 2,
                                "started_at": started,
                            },
                            headers={"Cache-Control": "no-store"},
                        )
            except Exception:
                pass
        if status in ("completed", "partial_failed", "failed"):
            slots = existing.get("slots", [])
            if not slots or len(slots) != len(expected_slots):
                slots = []
                for slot in expected_slots:
                    sj = _batch_ocr_dir(batch_code) / f"slot{slot}.json"
                    if sj.exists():
                        try:
                            with open(sj, "r", encoding="utf-8") as f:
                                slots.append(json.load(f))
                        except Exception:
                            slots.append({"slot": slot, "status": "failed", "batch_code": batch_code})
                    else:
                        slots.append({"slot": slot, "status": "pending", "batch_code": batch_code})
            return JSONResponse(
                status_code=200,
                content={
                    "success": True,
                    "batch_code": batch_code,
                    "status": status,
                    "overall_status": status,
                    "cached": True,
                    "message": "OCR already completed, returning cached result",
                    "slots": slots,
                    "poll_url": f"/api/check/ocr/{batch_code}",
                    "completed_at": existing.get("completed_at"),
                    "processing_time_total_seconds": existing.get("processing_time_total_seconds"),
                },
                headers={"Cache-Control": "no-store"},
            )
    # tulis early processing jika belum ada atau force
    if not existing or force or existing.get("overall_status") in ("pending", "failed", "partial_failed"):
        now = datetime.now(JAKARTA_TZ).isoformat()
        early = {
            "batch_code": batch_code,
            "overall_status": "processing",
            "progress": f"0/{len(expected_slots)}",
            "completed_count": 0,
            "failed_count": 0,
            "pending_count": len(expected_slots),
            "started_at": now,
            "updated_at": now,
            "slots": [{"slot": s, "status": "pending"} for s in expected_slots],
        }
        _write_batch_json(batch_code, early)
    # [OCR-REENABLE 2026-09-11 / PLAN_REVERT_TO_OCR] — compare Gemini OFF, OCR ON.
    #     Marker lama [OCR-DISABLED 2026-09-10] dibiarkan sebagai jejak di bawah.
    # [OCR-DISABLED 2026-09-10 / PLAN_VISUAL_COMPARE_MODE]
    #     Sistem berjalan dalam mode Komparasi Visual Label (Gemini).
    #     CARA AKTIFKAN KEMBALI (mode OCR): uncomment baris _process_batch_ocr di
    #     bawah dan comment/hapus baris _process_batch_compare. Fungsi
    #     _process_batch_ocr tetap utuh di file ini (tidak dihapus).

    # --- COMPARE_TEXT_SIMILARITY_PLAN (2026-09-12) + REVIEW_FLOW_COMPARE_FRONT_BACK_PLAN (2026-10-05): simpan permintaan compare ---
    # Dibaca worker _process_batch_ocr SEBELUM OCR mulai (anti race).
    # Tanpa reference → worker jalan tanpa similarity (backward compatible).
    if reference_text or reference_text_front or reference_text_back or reference_filename or reference_texts:
        try:
            cmp_req = {}
            if reference_text:
                cmp_req["reference_text"] = str(reference_text)[:20000]
            if reference_text_front:
                cmp_req["reference_text_front"] = str(reference_text_front)[:20000]
            if reference_text_back:
                cmp_req["reference_text_back"] = str(reference_text_back)[:20000]
            if reference_filename:
                cmp_req["reference_filename"] = sanitize_filename_for_lookup(reference_filename)
            # FIX_WRONG_REFERENCE (2026-10-07): map acuan per slot (dict/list/str-JSON mentah,
            # dinormalisasi saat dibaca worker). str() hanya bila string; dict/list apa adanya.
            if reference_texts:
                if isinstance(reference_texts, str):
                    cmp_req["reference_texts"] = reference_texts[:20000]
                else:
                    try:
                        cmp_req["reference_texts"] = json.loads(json.dumps(reference_texts))
                    except Exception:
                        cmp_req["reference_texts"] = str(reference_texts)[:20000]
            cmp_req["created_at"] = datetime.now(JAKARTA_TZ).isoformat()
            tmp = bdir / "compare_request.json.tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(cmp_req, f, ensure_ascii=False, indent=2)
            tmp.replace(bdir / "compare_request.json")
            print(f"[COMPARE] compare_request.json saved for {batch_code} "
                  f"(front={bool(reference_text_front)}, back={bool(reference_text_back)}, "
                  f"single={bool(reference_text)}, file={bool(reference_filename)}, "
                  f"texts={bool(reference_texts)})")
        except Exception as e:
            logger.warning(f"[COMPARE] save compare_request.json failed for {batch_code}: {e}")

    background_tasks.add_task(_process_batch_ocr, batch_code)
    # background_tasks.add_task(_process_batch_compare, batch_code)  # DISABLED 2026-09-11 — Gemini off
    print(f"[CHECK] Process triggered for {batch_code} -> background OCR {len(expected_slots)} slots")
    slots_pending = []
    for slot in expected_slots:
        p = _slot_png_path(batch_code, slot)
        slots_pending.append({"slot": slot, "status": "pending", "filename": p.name if p else f"slot{slot}.png"})
    return JSONResponse(
        status_code=202,
        content={
            "success": True,
            "batch_code": batch_code,
            "status": "processing",
            "overall_status": "processing",
            "message": f"OCR started for {len(expected_slots)} images",
            "slots": slots_pending,
            "poll_url": f"/api/check/ocr/{batch_code}",
            "retry_after_seconds": 2,
            "already_processing": False,
        },
        headers={"Cache-Control": "no-store", "Pragma": "no-cache", "Expires": "0"},
    )


# --- ENDPOINT: GET /api/check/ocr/{batch_code} ---
@router.get("/ocr/{batch_code}")
async def get_check_ocr_batch(batch_code: str, request: Request, x_api_key: Optional[str] = Header(None, alias="X-API-Key")):
    # rate limit GET (reuse)
    try:
        from app.security import check_get_rate_limit

        allowed, wait, remaining = check_get_rate_limit()
        if not allowed:
            # fallback build 429 jika main helper tidak ada
            wait_int = int(wait) + 1
            return JSONResponse(
                status_code=429,
                content={"success": False, "message": f"GET rate limit exceeded. Retry after {wait_int}s", "error": "too_many_requests", "retry_after_seconds": wait_int},
                headers={"Retry-After": str(wait_int), "X-RateLimit-Limit": "60", "X-RateLimit-Remaining": "0"},
            )
    except Exception:
        pass
    # optional auth: jika CHECK_API_KEY set dan client kirim key, validasi; jika tidak kirim, tetap izinkan (untuk polling internal)
    effective_key = x_api_key or request.headers.get("X-API-Key") or request.headers.get("x-api-key") or request.query_params.get("api_key")
    if CHECK_API_KEY and effective_key:
        if not _validate_api_key(effective_key):
            return JSONResponse(status_code=401, content={"success": False, "message": "Invalid API key", "error": "unauthorized"})
    # sanitasi batch_code
    if not batch_code or len(batch_code) > 64 or not BATCH_CODE_RE.match(batch_code):
        return JSONResponse(status_code=422, content={"success": False, "message": "Invalid batch_code", "error": "validation_failed"})
    safe_batch = sanitize_filename_for_lookup(batch_code)
    if safe_batch != batch_code:
        return JSONResponse(status_code=422, content={"success": False, "message": "Invalid batch_code", "error": "validation_failed"})
    bdir = _batch_dir(batch_code)
    if not is_safe_path(CHECK_UPLOAD_ROOT, bdir):
        return JSONResponse(status_code=422, content={"success": False, "message": "Invalid batch_code path", "error": "validation_failed"})
    if not bdir.exists():
        return JSONResponse(status_code=404, content={"success": False, "message": f"Batch not found: {batch_code}", "error": "batch_not_found"}, headers={"Cache-Control": "no-store"})
    batch = _read_batch_json(batch_code)
    if not batch:
        known_slots = _discover_batch_slots(batch_code)
        if not known_slots:
            known_slots = [1, 2, 3, 4, 5]
        valid = len([s for s in known_slots if _slot_png_path(batch_code, s) and _slot_png_path(batch_code, s).exists()])
        if valid not in CHECK_ALLOWED_COUNTS:
            return JSONResponse(
                status_code=200,
                content={
                    "success": False,
                    "batch_code": batch_code,
                    "overall_status": "pending",
                    "message": f"OCR not started yet for batch {batch_code}. Found {valid} files (expected 4 or 5). Call POST /api/check/process first.",
                    "found_files": valid,
                    "expected_counts": list(CHECK_ALLOWED_COUNTS),
                    "slots": [{"slot": i, "status": "pending", "filename": (_slot_png_path(batch_code, i).name if _slot_png_path(batch_code, i) else f"slot{i}.png")} for i in known_slots],
                },
                headers={"Cache-Control": "no-store", "Pragma": "no-cache", "Expires": "0"},
            )
        return JSONResponse(
            status_code=200,
            content={
                "success": False,
                "batch_code": batch_code,
                "overall_status": "pending",
                "message": f"OCR not started yet for batch {batch_code}. Call POST /api/check/process first.",
                "slots": [{"slot": i, "status": "pending", "filename": (_slot_png_path(batch_code, i).name if _slot_png_path(batch_code, i) else f"slot{i}.png")} for i in known_slots],
                "poll_url": f"/api/check/ocr/{batch_code}",
            },
            headers={"Cache-Control": "no-store", "Pragma": "no-cache", "Expires": "0"},
        )
    # enrich slots jika masih ringkas
    slots = batch.get("slots", [])
    if batch.get("overall_status") == "processing":
        # Slot acuan = batch.json bila ada, fallback ke file aktual di disk
        tracked = [s.get("slot") for s in slots if isinstance(s, dict) and isinstance(s.get("slot"), int)] or _discover_batch_slots(batch_code) or [1, 2, 3, 4, 5]
        real_slots: List[dict] = []
        for slot in tracked:
            sj = _batch_ocr_dir(batch_code) / f"slot{slot}.json"
            if sj.exists():
                try:
                    with open(sj, "r", encoding="utf-8") as f:
                        real_slots.append(json.load(f))
                except Exception:
                    real_slots.append({"slot": slot, "status": "pending", "batch_code": batch_code})
            else:
                found = next((s for s in slots if s.get("slot") == slot), {"slot": slot, "status": "pending", "batch_code": batch_code})
                real_slots.append(found)
        batch["slots"] = real_slots
        batch["completed_count"] = sum(1 for s in real_slots if s.get("status") == "success")
        batch["failed_count"] = sum(1 for s in real_slots if s.get("status") == "failed")
        batch["pending_count"] = sum(1 for s in real_slots if s.get("status") == "pending")
        batch["progress"] = f"{batch['completed_count']+batch['failed_count']}/{len(real_slots)}"
    headers = {
        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
        "Pragma": "no-cache",
        "Expires": "0",
        "ETag": f'W/"{batch_code}-{batch.get("updated_at","")}"',
    }
    if batch.get("updated_at"):
        headers["Last-Modified"] = batch.get("updated_at")
    body = {
        "success": True,
        "batch_code": batch_code,
        "overall_status": batch.get("overall_status"),
        "progress": batch.get("progress"),
        "completed_count": batch.get("completed_count"),
        "pending_count": batch.get("pending_count"),
        "failed_count": batch.get("failed_count"),
        "started_at": batch.get("started_at"),
        "updated_at": batch.get("updated_at"),
        "completed_at": batch.get("completed_at"),
        "processing_time_total_seconds": batch.get("processing_time_total_seconds"),
        "slots": batch.get("slots", []),
        # COMPARE_TEXT_SIMILARITY_PLAN (2026-09-12): ringkasan compare bila ada.
        "compare_summary": batch.get("compare_summary"),
    }
    if batch.get("overall_status") in ("partial_failed", "failed"):
        body["warning"] = f"{batch.get('failed_count')} slots failed (empty) — suggest retake with better lighting"
        body["empty_hint"] = "No text detected — try retake failed slots"
    return JSONResponse(status_code=200, content=body, headers=headers)


# --- ENDPOINT: GET /api/check/ocr/{batch_code}/slot/{slot_number} ---
@router.get("/ocr/{batch_code}/slot/{slot_number}")
async def get_check_ocr_slot(batch_code: str, slot_number: int, request: Request, x_api_key: Optional[str] = Header(None, alias="X-API-Key")):
    try:
        from app.security import check_get_rate_limit

        allowed, wait, remaining = check_get_rate_limit()
        if not allowed:
            wait_int = int(wait) + 1
            return JSONResponse(
                status_code=429,
                content={"success": False, "message": f"GET rate limit exceeded. Retry after {wait_int}s", "error": "too_many_requests", "retry_after_seconds": wait_int},
                headers={"Retry-After": str(wait_int)},
            )
    except Exception:
        pass
    if slot_number < 1 or slot_number > 5:
        return JSONResponse(status_code=422, content={"success": False, "message": "slot_number must be 1..5", "error": "validation_failed"})
    if not BATCH_CODE_RE.match(batch_code) or len(batch_code) > 64:
        return JSONResponse(status_code=422, content={"success": False, "message": "Invalid batch_code", "error": "validation_failed"})
    safe_batch = sanitize_filename_for_lookup(batch_code)
    if safe_batch != batch_code:
        return JSONResponse(status_code=422, content={"success": False, "message": "Invalid batch_code", "error": "validation_failed"})
    bdir = _batch_dir(batch_code)
    if not bdir.exists():
        return JSONResponse(status_code=404, content={"success": False, "message": f"Batch not found: {batch_code}", "error": "batch_not_found"})
    sj = _batch_ocr_dir(batch_code) / f"slot{slot_number}.json"
    if not sj.exists():
        batch = _read_batch_json(batch_code)
        if batch and batch.get("overall_status") == "processing":
            return JSONResponse(
                status_code=200,
                content={"success": False, "batch_code": batch_code, "slot": slot_number, "status": "pending", "message": f"Slot {slot_number} still processing, retry in 2s", "retry_after_seconds": 2},
                headers={"Cache-Control": "no-store", "Retry-After": "2"},
            )
        return JSONResponse(status_code=404, content={"success": False, "message": f"OCR result not found for slot {slot_number} in batch {batch_code}", "error": "not_found"}, headers={"Cache-Control": "no-store"})
    try:
        with open(sj, "r", encoding="utf-8") as f:
            data = json.load(f)
        return JSONResponse(status_code=200, content={"success": True, **data}, headers={"Cache-Control": "no-store", "Pragma": "no-cache", "Expires": "0"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "message": str(e), "error": "internal_error"})


# --- ENDPOINT: GET /api/check/latest (helper untuk debug retain 1) ---
@router.get("/latest")
async def get_latest_batch(request: Request, x_api_key: Optional[str] = Header(None, alias="X-API-Key")):
    # optional auth
    effective_key = x_api_key or request.headers.get("X-API-Key") or request.headers.get("x-api-key") or request.query_params.get("api_key")
    if CHECK_API_KEY and effective_key:
        if not _validate_api_key(effective_key):
            return JSONResponse(status_code=401, content={"success": False, "message": "Invalid API key", "error": "unauthorized"})
    try:
        from app.config import CHECK_LATEST_BATCH_FILE

        if CHECK_LATEST_BATCH_FILE.exists():
            batch = CHECK_LATEST_BATCH_FILE.read_text(encoding="utf-8").strip()
            if batch:
                # cek folder masih ada
                bdir = _batch_dir(batch)
                if bdir.exists():
                    return JSONResponse(
                        status_code=200,
                        content={"success": True, "latest_batch": batch, "poll_url": f"/api/check/ocr/{batch}", "source": "latest_batch.txt"},
                        headers={"Cache-Control": "no-store"},
                    )
    except Exception:
        pass
    # fallback: cari folder terbaru by mtime
    try:
        batches = [p for p in CHECK_UPLOAD_ROOT.iterdir() if p.is_dir() and BATCH_CODE_RE.match(p.name)]
        if not batches:
            return JSONResponse(status_code=404, content={"success": False, "message": "No batch found", "error": "not_found"}, headers={"Cache-Control": "no-store"})
        batches.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        latest = batches[0].name
        return JSONResponse(
            status_code=200,
            content={"success": True, "latest_batch": latest, "poll_url": f"/api/check/ocr/{latest}", "source": "mtime"},
            headers={"Cache-Control": "no-store"},
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "message": str(e), "error": "internal_error"})


# --- ENDPOINT: DELETE /api/check/cleanup (Retensi & Pembersihan Batch FastAPI) ---
@router.delete("/cleanup")
async def cleanup_check_uploads(
    request: Request,
    days: int = Query(7, ge=0, description="Hapus folder batch & file OCR yang usianya >= N hari (default: 7 hari, 0 = semua)"),
    dry_run: bool = Query(False, description="Simulasi tanpa menghapus file fisik"),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
):
    """
    Membersihkan subdirektori batch di uploads/check/ dan file JSON lama di uploads/ocr/
    yang lebih tua dari N hari (default: 7 hari).
    uploads/references/ (library global permanen) tidak pernah disentuh.
    """
    effective_key = x_api_key or request.headers.get("X-API-Key") or request.headers.get("x-api-key") or request.query_params.get("api_key")
    if CHECK_API_KEY and effective_key:
        if not _validate_api_key(effective_key):
            return JSONResponse(status_code=401, content={"success": False, "message": "Invalid API key", "error": "unauthorized"})

    try:
        now_ts = time.time()
        cutoff_seconds = days * 86400

        deleted_batches = 0
        deleted_files = 0
        freed_bytes = 0
        skipped_batches = 0
        errors = []

        # 1. Bersihkan subdirektori batch di uploads/check/
        if CHECK_UPLOAD_ROOT.exists():
            for p in CHECK_UPLOAD_ROOT.iterdir():
                if not p.is_dir():
                    continue
                if not BATCH_CODE_RE.match(p.name):
                    continue
                if not is_safe_path(CHECK_UPLOAD_ROOT, p):
                    continue

                try:
                    mtime = p.stat().st_mtime
                    age_seconds = now_ts - mtime

                    # Cek threshold hari
                    if days > 0 and age_seconds < cutoff_seconds:
                        skipped_batches += 1
                        continue

                    # Hitung ukuran folder dan jumlah file
                    batch_bytes = 0
                    batch_files = 0
                    for f in p.rglob("*"):
                        if f.is_file():
                            batch_files += 1
                            try:
                                batch_bytes += f.stat().st_size
                            except Exception:
                                pass

                    if not dry_run:
                        shutil.rmtree(p, ignore_errors=True)

                    deleted_batches += 1
                    deleted_files += batch_files
                    freed_bytes += batch_bytes

                except Exception as ex:
                    errors.append(f"Gagal memproses batch {p.name}: {str(ex)}")

        # 2. Bersihkan file JSON OCR lama di uploads/ocr/ (tanpa menyentuh uploads/references)
        try:
            from app.config import OCR_DIR
            if OCR_DIR.exists() and is_safe_path(Path(__file__).resolve().parent.parent / "uploads", OCR_DIR):
                for jf in OCR_DIR.glob("*.json"):
                    if not jf.is_file():
                        continue
                    try:
                        mtime = jf.stat().st_mtime
                        age_seconds = now_ts - mtime
                        if days > 0 and age_seconds < cutoff_seconds:
                            continue
                        fsize = 0
                        try:
                            fsize = jf.stat().st_size
                        except Exception:
                            pass

                        if not dry_run:
                            jf.unlink(missing_ok=True)

                        deleted_files += 1
                        freed_bytes += fsize
                    except Exception as ocr_ex:
                        errors.append(f"Gagal hapus OCR file {jf.name}: {str(ocr_ex)}")
        except Exception as ocr_err:
            errors.append(f"Gagal scan folder OCR: {str(ocr_err)}")

        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "dry_run": dry_run,
                "days_threshold": days,
                "deleted_batches": deleted_batches,
                "deleted_files": deleted_files,
                "freed_bytes": freed_bytes,
                "skipped_batches": skipped_batches,
                "errors": errors,
                "message": f"{'Simulasi: ' if dry_run else ''}Berhasil membersihkan {deleted_batches} batch ({deleted_files} file, {freed_bytes} bytes).",
            },
        )
    except Exception as e:
        logger.error(f"[CHECK CLEANUP] endpoint failed: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"success": False, "message": str(e), "error": "internal_error"})
