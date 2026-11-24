"""
app/security.py — Validasi file upload, sanitasi nama file, dan rate limiting.

Fokus: keamanan sistem, performa (cek cepat tanpa load berat), tidak mengeksekusi input user.
Semua fungsi pure / thread-safe; rate limit menggunakan threading.Lock.
"""

import os
import re
import time
import uuid
import threading
from collections import deque
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

from app.config import (
    ALLOWED_EXTENSIONS,
    MAGIC_BYTES,
    MAX_UPLOAD_SIZE,
    RATE_LIMIT_SECONDS,
    UPLOAD_DIR,
)

JAKARTA_TZ = timezone(timedelta(hours=7))

# Rate-limit state (in-memory, process-local)
_last_upload_time: float = 0.0
_rate_limit_lock = threading.Lock()

# --- GET & DELETE sliding-window rate limit (Ideal 429 handling §11) ---
# GET: 60 req / 60s (default) cukup untuk polling 10×3s =20/min + parallel retake
# DELETE: 30 req / 60s (best-effort cleanup, cegah abuse)
_get_timestamps: deque = deque()
_get_lock = threading.Lock()
_delete_timestamps: deque = deque()
_delete_lock = threading.Lock()


def validate_image_file(file_bytes: bytes, filename: str) -> tuple[bool, str]:
    """
    Validasi ukuran, ekstensi, dan magic bytes.

    Args:
        file_bytes: Bytes dari file yang diupload.
        filename: Nama file asli dari client (hanya untuk cek ekstensi).

    Returns:
        (valid, error_message) — error_message kosong jika valid.
    """
    # 1. Cek ukuran (cepat, tanpa baca disk)
    if len(file_bytes) > MAX_UPLOAD_SIZE:
        return False, "File size exceeds maximum limit of 5 MB"

    # 2. Cek ekstensi (case-insensitive)
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return False, "Only PNG and JPEG formats are supported"

    # 3. Cek magic bytes (pastikan isi file cocok dengan ekstensi)
    if ext == ".png":
        if len(file_bytes) < 8 or not file_bytes[:8].startswith(MAGIC_BYTES["png"]):
            return False, "File content does not match expected image format"
    elif ext in (".jpg", ".jpeg"):
        if len(file_bytes) < 3 or not file_bytes[:3].startswith(MAGIC_BYTES["jpeg"]):
            return False, "File content does not match expected image format"

    return True, ""


def generate_safe_filename(original_filename: str) -> str:
    """
    Generate nama file internal yang aman (tidak percaya input client).

    Format: label-{YYYYMMDD-HHMMSS}-{uuid8}[-suffix].{ext}
    Contoh: label-20260828-093924-a1b2c3d4.png
    Contoh split: label-20260930-093419-bb86b5df-front-1.png

    Suffix OPSIONAL dipertahankan dari basename original agar alur
    split dual-side (front/back + Label 1/2) bisa dikorelasi di log
    dan tidak saling menghapus saat retensi (FIX false-empty Label 1,
    2026-09-30): hanya pola -(front|back)(-[12])? di ujung basename
    yang dipertahankan (lowercase, whitelist). Selain itu perilaku
    sama seperti sebelumnya. Keunikan tetap dijamin timestamp+uuid8.

    Jika ekstensi tidak dikenali, fallback ke .png.

    Args:
        original_filename: Nama file dari client (hanya untuk ambil ekstensi + suffix).

    Returns:
        Nama file aman (hanya basename, tidak ada path traversal).
    """
    ext = os.path.splitext(original_filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        ext = ".png"

    timestamp = datetime.now(JAKARTA_TZ).strftime("%Y%m%d-%H%M%S")
    short_uuid = uuid.uuid4().hex[:8]

    # Pertahankan penanda sisi/label dari original (mis. -front-1, -back-2, -back).
    # Diambil via regex di ujung basename agar aman dari traversal/injeksi.
    suffix = ""
    try:
        base = os.path.splitext(os.path.basename(original_filename or ""))[0]
        m = re.search(r"-(front|back)(-[12])?$", base, re.IGNORECASE)
        if m:
            suffix = "-" + m.group(0).lstrip("-").lower()
            suffix = re.sub(r"[^a-z0-9-]", "", suffix)
    except Exception:
        suffix = ""

    # Hanya izinkan karakter aman untuk timestamp dan uuid (sudah aman),
    # tapi tetap pastikan tidak ada path separator
    safe_name = f"label-{timestamp}-{short_uuid}{suffix}{ext}"
    # Defense-in-depth: hapus karakter selain alfanumerik, dash, underscore, dot
    safe_name = re.sub(r"[^a-zA-Z0-9._-]", "", safe_name)
    return safe_name


def is_safe_path(base: Path, target: Path) -> bool:
    """
    Cek apakah target path berada di dalam base path (cegah path traversal).

    Args:
        base: Folder yang diizinkan (UPLOAD_DIR).
        target: Path yang akan dicek.

    Returns:
        True jika aman (di dalam base), False jika di luar base.
    """
    try:
        base_resolved = base.resolve()
        target_resolved = target.resolve()
        # Python 3.9+: Path.is_relative_to ; fallback manual untuk kompatibilitas
        try:
            return target_resolved.is_relative_to(base_resolved)
        except AttributeError:
            # Fallback untuk Python versi lama (tidak dipakai di sini, tapi aman)
            return str(target_resolved).startswith(str(base_resolved))
    except Exception:
        return False


# --- POST sliding-window burst (FIX 429 dual-side front+back, 2026-09-15) ---
# Menggantikan single-slot _last_upload_time untuk capture.
# check_rate_limit() lama dipertahankan sebagai wrapper agar main.py/test lama tidak rusak.
_post_timestamps: deque = deque()
_post_lock = threading.Lock()


def check_post_rate_limit() -> tuple[bool, float, int]:
    """
    Rate limit POST capture burst: default 5 req / 60s (env POST_RATE_LIMIT_*).
    Returns (allowed, wait_seconds, remaining).
    """
    try:
        from app.config import POST_RATE_LIMIT_REQUESTS, POST_RATE_LIMIT_WINDOW_SECONDS
    except ImportError:
        POST_RATE_LIMIT_REQUESTS, POST_RATE_LIMIT_WINDOW_SECONDS = 60, 60
    return _check_sliding_window(_post_timestamps, _post_lock, POST_RATE_LIMIT_REQUESTS, POST_RATE_LIMIT_WINDOW_SECONDS)


def check_rate_limit() -> tuple[bool, float]:
    """
    Backward-compat: delegasi ke check_post_rate_limit (abaikan remaining).

    Dulu: single-slot 1 gambar/menit via _last_upload_time + RATE_LIMIT_SECONDS.
    Sekarang: burst sliding-window agar front+back <60 detik lolos.
    _last_upload_time dipertahankan (tidak dipakai) agar import lama tidak rusak.
    """
    allowed, wait, _ = check_post_rate_limit()
    return allowed, wait


def reset_rate_limit_for_testing() -> None:
    """
    Reset rate limit (hanya untuk testing).
    Jangan pakai di production flow normal.
    """
    global _last_upload_time
    with _rate_limit_lock:
        _last_upload_time = 0.0
    with _post_lock:
        try:
            _post_timestamps.clear()
        except NameError:
            pass
    # Juga reset GET/DELETE/Process windows untuk test isolation
    with _get_lock:
        _get_timestamps.clear()
    with _delete_lock:
        _delete_timestamps.clear()
    with _process_lock:
        try:
            _process_timestamps.clear()
        except NameError:
            pass
    with _reference_lock:
        try:
            _reference_timestamps.clear()
        except NameError:
            pass


def _check_sliding_window(
    timestamps: deque,
    lock: threading.Lock,
    limit: int,
    window_seconds: int,
) -> tuple[bool, float, int]:
    """
    Generic sliding window check.
    Returns (allowed, wait_seconds, remaining).
    Thread-safe, bersihkan timestamp kadaluarsa > window.
    """
    now = time.time()
    with lock:
        # Hapus yang sudah lewat window
        while timestamps and (now - timestamps[0]) >= window_seconds:
            timestamps.popleft()
        if len(timestamps) >= limit:
            # Hitung wait = window - (now - oldest)
            oldest = timestamps[0]
            wait = window_seconds - (now - oldest)
            remaining = 0
            return False, max(0.0, wait), remaining
        # Izinkan: catat timestamp
        timestamps.append(now)
        remaining = limit - len(timestamps)
        return True, 0.0, remaining


def check_get_rate_limit() -> tuple[bool, float, int]:
    """
    Rate limit untuk endpoint GET polling (./ocr/*, ./status, ./files, ./image/*).
    Default: 60 req / 60s (config GET_RATE_LIMIT_REQUESTS/WINDOW).
    Returns (allowed, wait_seconds, remaining).
    """
    try:
        from app.config import GET_RATE_LIMIT_REQUESTS, GET_RATE_LIMIT_WINDOW_SECONDS
    except ImportError:
        GET_RATE_LIMIT_REQUESTS, GET_RATE_LIMIT_WINDOW_SECONDS = 60, 60
    return _check_sliding_window(_get_timestamps, _get_lock, GET_RATE_LIMIT_REQUESTS, GET_RATE_LIMIT_WINDOW_SECONDS)


def check_delete_rate_limit() -> tuple[bool, float, int]:
    """
    Rate limit untuk DELETE cleanup (Laravel best-effort).
    Default: 30 req / 60s.
    Returns (allowed, wait_seconds, remaining).
    """
    try:
        from app.config import DELETE_RATE_LIMIT_REQUESTS, DELETE_RATE_LIMIT_WINDOW_SECONDS
    except ImportError:
        DELETE_RATE_LIMIT_REQUESTS, DELETE_RATE_LIMIT_WINDOW_SECONDS = 30, 60
    return _check_sliding_window(_delete_timestamps, _delete_lock, DELETE_RATE_LIMIT_REQUESTS, DELETE_RATE_LIMIT_WINDOW_SECONDS)


# --- Process rate limit (PLAN_CHECK_OCR_5_LABEL_PROCESS) ---
_process_timestamps: deque = deque()
_process_lock = threading.Lock()


def check_process_rate_limit() -> tuple[bool, float, int]:
    """
    Rate limit untuk POST /api/check/process (trigger OCR batch).
    Default: 10 req / 60s (config CHECK_PROCESS_RATE_LIMIT_REQUESTS/WINDOW).
    Returns (allowed, wait_seconds, remaining).
    """
    try:
        from app.config import CHECK_PROCESS_RATE_LIMIT_REQUESTS, CHECK_PROCESS_RATE_LIMIT_WINDOW_SECONDS
    except ImportError:
        CHECK_PROCESS_RATE_LIMIT_REQUESTS, CHECK_PROCESS_RATE_LIMIT_WINDOW_SECONDS = 10, 60
    return _check_sliding_window(_process_timestamps, _process_lock, CHECK_PROCESS_RATE_LIMIT_REQUESTS, CHECK_PROCESS_RATE_LIMIT_WINDOW_SECONDS)


def reset_process_rate_limit_for_testing() -> None:
    """Reset process rate limit (testing)."""
    with _process_lock:
        _process_timestamps.clear()


# --- Reference library rate limit (PLAN_REFERENCE_LIBRARY_FASTAPI, fix 2026-09-15) ---
# POST /api/references & POST /from-capture punya limiter SENDIRI (10 req/menit),
# terpisah dari limiter global capture 1/menit. Alasan: push library dari Laravel
# selalu datang <1 detik setelah forward POST /api/test/status — kalau berbagi
# slot global, push SELALU kena 429 dan library tidak pernah terisi (Jalur A gagal).
_reference_timestamps: deque = deque()
_reference_lock = threading.Lock()


def check_reference_rate_limit() -> tuple[bool, float, int]:
    """
    Rate limit khusus create reference library.
    Default: 10 req / 60s (longgar untuk push best-effort, tetap anti-spam).
    Returns (allowed, wait_seconds, remaining).
    """
    return _check_sliding_window(_reference_timestamps, _reference_lock, 10, 60)


def get_rate_limit_headers(limit: int, remaining: int, reset_seconds: float) -> dict:
    """
    Bangun headers standar untuk observability 429 ideal.
    - X-RateLimit-Limit, X-RateLimit-Remaining, X-RateLimit-Reset
    - Retry-After (jika remaining==0)
    """
    headers = {
        "X-RateLimit-Limit": str(limit),
        "X-RateLimit-Remaining": str(max(0, remaining)),
        "X-RateLimit-Reset": str(int(reset_seconds)) if reset_seconds > 0 else "0",
    }
    if remaining == 0 and reset_seconds > 0:
        headers["Retry-After"] = str(int(reset_seconds) + 1)
    return headers


def verify_delete_api_key(provided: Optional[str]) -> tuple[bool, str]:
    """
    Verifikasi header X-API-Key untuk DELETE (Laravel retakeCleanup).
    Jika LARAVEL_API_KEY kosong di server (dev), izinkan semua (warning) — tapi log.
    Jika LARAVEL_API_KEY terisi, wajib cocok exact.

    Returns (valid, reason) — reason kosong jika valid.
    """
    try:
        from app.config import LARAVEL_API_KEY
    except ImportError:
        LARAVEL_API_KEY = ""
    if not LARAVEL_API_KEY:
        # Dev mode: tidak ada key terkonfigurasi → izinkan tapi log warning di caller
        return True, "no_api_key_configured"
    if not provided:
        return False, "missing X-API-Key header"
    # Timing-safe? Untuk key statis, compare_digest optional tapi pakai == dulu
    import hmac as _hmac

    if _hmac.compare_digest(provided, LARAVEL_API_KEY):
        return True, ""
    return False, "invalid X-API-Key"


def sanitize_filename_for_lookup(filename: str) -> str:
    """
    Sanitasi filename yang dipakai untuk lookup GET /ocr/{filename}.
    Hanya izinkan karakter aman, cegah path traversal.

    Args:
        filename: Nama file dari URL path parameter.

    Returns:
        Nama file yang sudah disanitasi (basename saja).
    """
    # Ambil basename (hapus direktori)
    base = os.path.basename(filename)
    # Hapus karakter berbahaya
    base = re.sub(r"[^a-zA-Z0-9._-]", "", base)
    return base


# ---------------------------------------------------------------------------
# Check Batch — 5 Images (PLAN_CHECK_5_IMAGES)
# ---------------------------------------------------------------------------
_BATCH_RE = re.compile(r"^[A-Za-z0-9\-_]+$")


def validate_batch_code(batch_code: str) -> tuple[bool, str]:
    """Validasi batch_code anti traversal, regex, panjang 1..64."""
    if not batch_code or len(batch_code) > 64:
        return False, "batch_code must be 1..64 chars"
    if not _BATCH_RE.match(batch_code):
        return False, "batch_code must match ^[A-Za-z0-9\\-_]+$"
    if batch_code != os.path.basename(batch_code):
        return False, "batch_code must not contain path"
    return True, ""
