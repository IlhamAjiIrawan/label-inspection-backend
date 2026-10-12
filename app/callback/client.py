"""
app/callback/client.py — HTTP client untuk push cleaned_text ke Laravel.

Fitur:
 - Validasi SSRF (allowlist host) dan skema
 - HMAC-SHA256 signing (X-Signature)
 - Retry exponential backoff 2/4/8s + jitter, bedakan 4xx vs 5xx
 - Idempotency header X-Idempotency-Key = filename
 - Logging ke uploads/callback_log.jsonl dan dead letter
 - Support sync (httpx.Client) dan async (httpx.AsyncClient) — pilih satu sesuai pemanggil

Performa: sync client dipakai dari process_ocr_background (background thread) — tidak blokir response POST.
Keamanan: tidak log secret/api_key, tidak eksekusi input user, timeout ketat.
"""

import json
import logging
import random
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import httpx

from app.config import (
    CALLBACK_ALLOWED_HOSTS,
    CALLBACK_DEAD_LETTER_FILE,
    CALLBACK_ENABLED,
    CALLBACK_INCLUDE_RAW_TEXT,
    CALLBACK_LOG_FILE,
    CALLBACK_RETRIES,
    CALLBACK_RETRY_BACKOFF_BASE,
    CALLBACK_SEND_ON_FAILED,
    CALLBACK_TIMEOUT_SECONDS,
    LARAVEL_API_KEY,
    LARAVEL_CALLBACK_URL,
    LARAVEL_HMAC_SECRET,
)
from app.callback.payload import build_callback_payload, build_failed_payload
from app.callback.signer import sign_payload

logger = logging.getLogger(__name__)
JAKARTA_TZ = timezone(timedelta(hours=7))

# Batas ukuran log file sebelum rotate (10 MB)
LOG_ROTATE_SIZE_BYTES = 10 * 1024 * 1024

# ---------------------------------------------------------------------------
# R5: pool callback terpisah + httpx.Client bersama (keep-alive).
# Worker OCR tidak lagi tertahan retry sleep; callback jalan di pool sendiri.
# ---------------------------------------------------------------------------
from concurrent.futures import ThreadPoolExecutor as _TPE

try:
    from app import config as _cb_cfg

    _CB_WORKERS = max(1, int(getattr(_cb_cfg, "CALLBACK_WORKERS", 2) or 2))
except Exception:
    _CB_WORKERS = 2

CALLBACK_EXECUTOR = _TPE(max_workers=_CB_WORKERS, thread_name_prefix="callback")

_http_shared: "httpx.Client | None" = None
_http_lock = __import__("threading").Lock()


def _shared_client() -> "httpx.Client":
    """Satu httpx.Client bersama (connection pool keep-alive) — R5."""
    global _http_shared
    if _http_shared is not None:
        return _http_shared
    with _http_lock:
        if _http_shared is not None:
            return _http_shared
        timeout = httpx.Timeout(CALLBACK_TIMEOUT_SECONDS, connect=2.0)
        _http_shared = httpx.Client(
            timeout=timeout,
            follow_redirects=False,
            limits=httpx.Limits(max_connections=4, max_keepalive_connections=2),
        )
        return _http_shared


def submit_callback(saved_filename: str, result: dict, metadata: dict | None = None):
    """
    Submit callback non-blocking ke CALLBACK_EXECUTOR (R5).

    Metadata diambil lazy di worker bila tidak diberikan, agar caller
    (worker OCR) bebas segera setelah store.put.
    """
    def _run():
        try:
            meta = metadata
            if meta is None:
                try:
                    from app.storage import load_metadata as _lm

                    meta = _lm().get(saved_filename)
                except Exception:
                    meta = None
            cb_result = send_callback_sync(result, meta)
            # Update metadata callback status untuk observability (R5, di pool callback).
            try:
                from app.storage import load_metadata as _lm2, save_metadata as _sm2
                from datetime import datetime as _dt

                try:
                    from zoneinfo import ZoneInfo as _ZI  # noqa

                    _tz = _ZI("Asia/Jakarta")
                    _now = _dt.now(_tz).isoformat()
                except Exception:
                    from datetime import timezone as _tzc, timedelta as _td

                    _now = _dt.now(_tzc(_td(hours=7))).isoformat()
                meta_all2 = _lm2()
                if saved_filename in meta_all2:
                    meta_all2[saved_filename]["callback_last_attempt"] = _now
                    meta_all2[saved_filename]["callback_success"] = bool(cb_result.get("success", False))
                    meta_all2[saved_filename]["callback_attempts"] = int(cb_result.get("attempts", 0) or 0)
                    if cb_result.get("error"):
                        meta_all2[saved_filename]["callback_error"] = str(cb_result.get("error"))[:500]
                    else:
                        meta_all2[saved_filename].pop("callback_error", None)
                    _sm2(meta_all2)
            except Exception as e2:
                logger.warning(f"submit_callback metadata update failed for {saved_filename}: {e2}")
        except Exception as e:
            logger.warning(f"submit_callback worker error for {saved_filename}: {e}", exc_info=True)

    try:
        CALLBACK_EXECUTOR.submit(_run)
    except Exception as e:
        logger.warning(f"submit_callback submit failed for {saved_filename}: {e}")
        # Fallback sync agar callback tidak hilang diam-diam.
        try:
            _run()
        except Exception:
            pass


def _is_url_allowed(url: str) -> tuple[bool, str]:
    """
    Cek apakah URL callback diizinkan (cegah SSRF).

    Returns:
        (allowed, reason) — reason kosong jika allowed.
    """
    try:
        parsed = urlparse(url)
    except Exception as e:
        return False, f"Invalid URL: {e}"

    # Hanya izinkan http/https
    if parsed.scheme not in ("http", "https"):
        return False, f"Scheme not allowed: {parsed.scheme} (only http/https)"

    if not parsed.hostname:
        return False, "Missing hostname"

    hostname = parsed.hostname.lower()

    # Blokir metadata service & loopback aneh (defense-in-depth)
    blocked_hosts = {"169.254.169.254", "metadata.google.internal"}
    if hostname in blocked_hosts:
        return False, f"Blocked host: {hostname}"

    # Jika allowlist kosong, izinkan semua http/https (tapi tetap blokir di atas)
    if not CALLBACK_ALLOWED_HOSTS:
        return True, ""

    # Cek allowlist
    if hostname in CALLBACK_ALLOWED_HOSTS:
        return True, ""

    # Izinkan subdomain? Untuk sekarang strict: hanya exact match
    # Jika butuh wildcard, tambah logic di sini
    return False, f"Host not in allowlist {CALLBACK_ALLOWED_HOSTS}: {hostname}"


def _append_log(path: Path, data: Dict[str, Any]) -> None:
    """Append JSON Lines ke file log (buat folder jika belum ada). Rotate jika >10MB."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)

        # Rotate jika terlalu besar
        try:
            if path.exists() and path.stat().st_size > LOG_ROTATE_SIZE_BYTES:
                rotated = path.with_suffix(path.suffix + ".1")
                # Hapus rotated lama jika ada
                if rotated.exists():
                    rotated.unlink()
                path.rename(rotated)
                logger.info(f"Rotated callback log {path} -> {rotated}")
        except Exception as e:
            logger.warning(f"Failed to rotate log {path}: {e}")

        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning(f"Failed to write callback log {path}: {e}")


def _should_retry_status(status_code: int) -> bool:
    """
    Tentukan apakah status code boleh di-retry.
    - 5xx → retry
    - 429 → retry
    - 408 → retry
    - 4xx lain → jangan retry (client error, percuma)
    """
    if 500 <= status_code <= 599:
        return True
    if status_code in (429, 408):
        return True
    return False


def _build_headers(payload: Dict[str, Any], timestamp: str) -> Dict[str, str]:
    """Bangun headers callback (dipanggil per attempt karena timestamp & signature baru)."""
    signature = sign_payload(payload, timestamp, LARAVEL_HMAC_SECRET)
    return {
        "Content-Type": "application/json",
        "X-API-Key": LARAVEL_API_KEY,
        "X-Timestamp": timestamp,
        "X-Signature": f"sha256={signature}",
        "X-Idempotency-Key": payload.get("filename", ""),
        "User-Agent": "FastAPI-OCR-Callback/1.0",
    }


# ---------------------------------------------------------------------------
# SYNC version — dipakai dari process_ocr_background (thread pool, BackgroundTasks)
# ---------------------------------------------------------------------------
def send_callback_sync(
    ocr_result: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
    include_raw_text: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Kirim hasil OCR ke Laravel secara SYNC dengan retry.

    Dipanggil dari process_ocr_background (sync, di BackgroundTasks thread).
    Tidak raise exception — selalu return dict status agar OCR tidak gagal karena callback.

    Args:
        ocr_result: Dict hasil OCR (wajib 'filename').
        metadata: Dict metadata untuk enrichment (opsional).
        include_raw_text: Override CALLBACK_INCLUDE_RAW_TEXT jika diisi.

    Returns:
        {"success": bool, "attempts": int, "last_status": int|None, "error": str|None, "skipped": bool}
    """
    start_total = time.time()

    # --- Gate: enabled? ---
    if not CALLBACK_ENABLED:
        logger.info("Callback disabled (CALLBACK_ENABLED=false), skip")
        return {"success": False, "attempts": 0, "last_status": None, "error": "disabled", "skipped": True}

    # --- Gate: config lengkap? ---
    if not LARAVEL_CALLBACK_URL or not LARAVEL_API_KEY or not LARAVEL_HMAC_SECRET:
        logger.warning("Callback config incomplete (LARAVEL_CALLBACK_URL/API_KEY/HMAC_SECRET kosong) — skip. Cek .env")
        _append_log(
            CALLBACK_LOG_FILE,
            {
                "ts": datetime.now(JAKARTA_TZ).isoformat(),
                "filename": ocr_result.get("filename", "unknown"),
                "error": "config incomplete",
                "skipped": True,
            },
        )
        return {"success": False, "attempts": 0, "last_status": None, "error": "config incomplete", "skipped": True}

    # --- Gate: SSRF allowlist ---
    allowed, reason = _is_url_allowed(LARAVEL_CALLBACK_URL)
    if not allowed:
        logger.error(f"Callback URL not allowed ({reason}): {LARAVEL_CALLBACK_URL}")
        _append_log(
            CALLBACK_LOG_FILE,
            {
                "ts": datetime.now(JAKARTA_TZ).isoformat(),
                "filename": ocr_result.get("filename", "unknown"),
                "error": f"URL not allowed: {reason}",
                "url": LARAVEL_CALLBACK_URL,
            },
        )
        return {"success": False, "attempts": 0, "last_status": None, "error": reason, "skipped": True}

    filename = ocr_result.get("filename", "unknown")
    status = ocr_result.get("status", "unknown")

    # --- Gate: hanya kirim success? Fix Blank 2026-09-06 ---
    # Laravel di C:\laragon\www\labelIns sudah guard empty success → failed,
    # jadi FastAPI sekarang kirim status "failed" untuk empty setelah 2-pass.
    # Gate lama skip semua failed (CALLBACK_SEND_ON_FAILED=false) → Laravel tidak tahu empty.
    # Fix: izinkan failed jika is_empty atau error empty cleaned_text (biar Laravel dapat failed flag)
    is_empty_gate = ocr_result.get("is_empty") is True or ocr_result.get("empty_reason") is not None
    error_text = str(ocr_result.get("error") or "")
    is_empty_error = "empty cleaned_text" in error_text.lower() or "no_text_detected" in error_text.lower()
    allow_failed_empty = is_empty_gate or is_empty_error
    if status != "success" and not CALLBACK_SEND_ON_FAILED and not allow_failed_empty:
        logger.info(f"Skip callback for {filename} status={status} (CALLBACK_SEND_ON_FAILED=false, not empty)")
        return {"success": False, "attempts": 0, "last_status": None, "error": f"status {status} skipped", "skipped": True}
    if allow_failed_empty and status == "failed":
        logger.info(f"Callback allow failed-empty for {filename} (is_empty={is_empty_gate}, error empty={is_empty_error}) — will PUSH despite SEND_ON_FAILED=false")

    # --- Build payload ---
    try:
        use_raw = CALLBACK_INCLUDE_RAW_TEXT if include_raw_text is None else include_raw_text
        if status == "failed" and CALLBACK_SEND_ON_FAILED:
            payload = build_failed_payload(ocr_result, metadata)
        else:
            payload = build_callback_payload(ocr_result, metadata, include_raw_text=use_raw)
    except Exception as e:
        logger.error(f"Failed to build callback payload for {filename}: {e}", exc_info=True)
        return {"success": False, "attempts": 0, "last_status": None, "error": str(e), "skipped": True}

    # --- HTTP loop dengan retry (R5: pakai shared client keep-alive) ---
    last_status: Optional[int] = None
    last_error: Optional[str] = None
    attempts = 0

    try:
        client = _shared_client()
        _use_shared = True
    except Exception:
        timeout = httpx.Timeout(CALLBACK_TIMEOUT_SECONDS, connect=2.0)
        client = httpx.Client(timeout=timeout, follow_redirects=False)
        _use_shared = False
    try:
        for attempt in range(1, CALLBACK_RETRIES + 1):
            attempts = attempt
            timestamp = datetime.now(JAKARTA_TZ).isoformat()
            headers = _build_headers(payload, timestamp)
            # Update attempt di payload untuk audit Laravel (tapi jangan re-sign setelah update)
            # Kita update payload sebelum sign agar signature cover attempt
            payload["callback_attempt"] = attempt
            # Re-sign karena payload berubah
            headers = _build_headers(payload, timestamp)

            try:
                resp = client.post(LARAVEL_CALLBACK_URL, json=payload, headers=headers)
                last_status = resp.status_code
                latency_ms = int((time.time() - start_total) * 1000)

                # Log setiap attempt (tanpa secret)
                _append_log(
                    CALLBACK_LOG_FILE,
                    {
                        "ts": timestamp,
                        "filename": filename,
                        "attempt": attempt,
                        "status_code": resp.status_code,
                        "latency_ms": latency_ms,
                        # Hanya 500 char pertama response untuk debug, tanpa PII penuh
                        "response_preview": resp.text[:500] if resp.text else "",
                    },
                )

                if 200 <= resp.status_code < 300:
                    logger.info(f"Callback OK {filename} attempt {attempt} -> {resp.status_code} ({latency_ms}ms)")
                    print(f"[CALLBACK] Success: {filename} -> Laravel {resp.status_code} (attempt {attempt})")
                    return {"success": True, "attempts": attempts, "last_status": last_status, "error": None, "skipped": False}

                # 4xx non-retry → dead letter
                if 400 <= resp.status_code < 500 and not _should_retry_status(resp.status_code):
                    logger.error(f"Callback client error {resp.status_code} for {filename} — no retry: {resp.text[:500]}")
                    # HINT untuk 401 — Fix Retake 401
                    if resp.status_code == 401:
                        logger.error(
                            f"HINT 401 Invalid API key for {filename} — cek LARAVEL_API_KEY di fastAPI/.env vs Laravel C:\\laragon\\www\\labelIns\\.env. "
                            f"FastAPI key prefix: {LARAVEL_API_KEY[:8]}... (len {len(LARAVEL_API_KEY)}), "
                            f"expected Laravel key: <cek Laravel .env OCR_CALLBACK_API_KEY> (Cek config/services.php ocr.callback_api_key). "
                            f"Sinkron 3 vars: LARAVEL_API_KEY / LARAVEL_HMAC_SECRET / LARAVEL_CALLBACK_URL harus identik. "
                            f"Lihat plans/fix_fastapi_external_service_ocr_and_callback.md §7"
                        )
                        print(f"[CALLBACK HINT] 401 for {filename}: check .env sync — fastAPI key {LARAVEL_API_KEY[:8]}... vs Laravel .env — see .env.example")
                    _append_log(CALLBACK_DEAD_LETTER_FILE, {**payload, "_dead_reason": f"client_error_{resp.status_code}", "_ts": timestamp})
                    print(f"[CALLBACK] Dead letter (client error {resp.status_code}): {filename}")
                    return {"success": False, "attempts": attempts, "last_status": last_status, "error": f"client_error {resp.status_code}", "skipped": False}

                # 5xx / 429 → retry
                last_error = f"http_{resp.status_code}"
                logger.warning(f"Callback {filename} attempt {attempt} got {resp.status_code}, will retry if attempts left...")

            except (httpx.ConnectError, httpx.TimeoutException, httpx.NetworkError, httpx.ReadError) as e:
                last_error = f"network:{type(e).__name__}:{str(e)[:200]}"
                last_status = None
                logger.warning(f"Callback network error attempt {attempt} for {filename}: {e}")
                _append_log(
                    CALLBACK_LOG_FILE,
                    {
                        "ts": datetime.now(JAKARTA_TZ).isoformat(),
                        "filename": filename,
                        "attempt": attempt,
                        "error": last_error,
                    },
                )
            except Exception as e:
                last_error = f"unexpected:{type(e).__name__}:{str(e)[:200]}"
                logger.error(f"Callback unexpected error attempt {attempt} for {filename}: {e}", exc_info=True)
                _append_log(
                    CALLBACK_LOG_FILE,
                    {
                        "ts": datetime.now(JAKARTA_TZ).isoformat(),
                        "filename": filename,
                        "attempt": attempt,
                        "error": last_error,
                    },
                )

            # Sleep sebelum retry berikutnya
            if attempt < CALLBACK_RETRIES:
                backoff = CALLBACK_RETRY_BACKOFF_BASE ** attempt  # 2, 4, 8
                jitter = random.uniform(0, 1)
                sleep_s = backoff + jitter
                logger.info(f"Callback retry sleep {sleep_s:.2f}s before attempt {attempt+1} for {filename}")
                time.sleep(sleep_s)

    finally:
        # Tutup client non-shared saja; shared client dipakai ulang (keep-alive).
        try:
            if not _use_shared:
                client.close()
        except Exception:
            pass

    # Gagal setelah semua retry
    logger.error(f"Callback failed after {CALLBACK_RETRIES} attempts for {filename} last={last_error} status={last_status}")
    print(f"[CALLBACK] Failed after {CALLBACK_RETRIES} attempts: {filename} -> dead letter")
    try:
        _append_log(CALLBACK_DEAD_LETTER_FILE, {**payload, "_dead_reason": last_error or "max_retries", "_ts": datetime.now(JAKARTA_TZ).isoformat()})
    except Exception:
        pass

    return {"success": False, "attempts": attempts, "last_status": last_status, "error": last_error, "skipped": False}


# ---------------------------------------------------------------------------
# ASYNC version — dipakai jika process_ocr_background diubah jadi async
# ---------------------------------------------------------------------------
async def send_callback(
    ocr_result: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
    include_raw_text: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Versi ASYNC dari send_callback_sync (pakai httpx.AsyncClient).

    Hanya dipakai jika caller adalah async def (misal BackgroundTasks dengan async task).
    Logic sama persis, hanya pakai await + asyncio.sleep.
    """
    import asyncio

    start_total = time.time()

    if not CALLBACK_ENABLED:
        logger.info("Callback disabled, skip")
        return {"success": False, "attempts": 0, "last_status": None, "error": "disabled", "skipped": True}

    if not LARAVEL_CALLBACK_URL or not LARAVEL_API_KEY or not LARAVEL_HMAC_SECRET:
        logger.warning("Callback config incomplete — skip")
        return {"success": False, "attempts": 0, "last_status": None, "error": "config incomplete", "skipped": True}

    allowed, reason = _is_url_allowed(LARAVEL_CALLBACK_URL)
    if not allowed:
        logger.error(f"Callback URL not allowed: {reason}")
        return {"success": False, "attempts": 0, "last_status": None, "error": reason, "skipped": True}

    filename = ocr_result.get("filename", "unknown")
    status = ocr_result.get("status", "unknown")
    # Fix Blank: izinkan failed-empty agar Laravel tahu
    is_empty_gate = ocr_result.get("is_empty") is True or ocr_result.get("empty_reason") is not None
    error_text = str(ocr_result.get("error") or "")
    is_empty_error = "empty cleaned_text" in error_text.lower() or "no_text_detected" in error_text.lower()
    allow_failed_empty = is_empty_gate or is_empty_error
    if status != "success" and not CALLBACK_SEND_ON_FAILED and not allow_failed_empty:
        logger.info(f"Skip callback {filename} status={status}")
        return {"success": False, "attempts": 0, "last_status": None, "error": f"status {status} skipped", "skipped": True}
    if allow_failed_empty and status == "failed":
        logger.info(f"Callback allow async failed-empty for {filename}")

    try:
        use_raw = CALLBACK_INCLUDE_RAW_TEXT if include_raw_text is None else include_raw_text
        if status == "failed" and CALLBACK_SEND_ON_FAILED:
            payload = build_failed_payload(ocr_result, metadata)
        else:
            payload = build_callback_payload(ocr_result, metadata, include_raw_text=use_raw)
    except Exception as e:
        logger.error(f"Failed to build payload {filename}: {e}")
        return {"success": False, "attempts": 0, "last_status": None, "error": str(e), "skipped": True}

    timeout = httpx.Timeout(CALLBACK_TIMEOUT_SECONDS, connect=2.0)
    last_status: Optional[int] = None
    last_error: Optional[str] = None
    attempts = 0

    async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
        for attempt in range(1, CALLBACK_RETRIES + 1):
            attempts = attempt
            timestamp = datetime.now(JAKARTA_TZ).isoformat()
            payload["callback_attempt"] = attempt
            headers = _build_headers(payload, timestamp)

            try:
                resp = await client.post(LARAVEL_CALLBACK_URL, json=payload, headers=headers)
                last_status = resp.status_code
                latency_ms = int((time.time() - start_total) * 1000)
                _append_log(
                    CALLBACK_LOG_FILE,
                    {
                        "ts": timestamp,
                        "filename": filename,
                        "attempt": attempt,
                        "status_code": resp.status_code,
                        "latency_ms": latency_ms,
                        "response_preview": resp.text[:500] if resp.text else "",
                    },
                )
                if 200 <= resp.status_code < 300:
                    logger.info(f"Callback OK async {filename} attempt {attempt}")
                    print(f"[CALLBACK] Success async: {filename} -> {resp.status_code}")
                    return {"success": True, "attempts": attempts, "last_status": last_status, "error": None, "skipped": False}
                if 400 <= resp.status_code < 500 and not _should_retry_status(resp.status_code):
                    logger.error(f"Callback async client error {resp.status_code} for {filename} — no retry: {resp.text[:500] if hasattr(resp, 'text') else ''}")
                    if resp.status_code == 401:
                        logger.error(f"HINT async 401 for {filename}: check LARAVEL_API_KEY sync — see .env.example § Callback")
                    _append_log(CALLBACK_DEAD_LETTER_FILE, {**payload, "_dead_reason": f"client_error_{resp.status_code}", "_ts": timestamp})
                    return {"success": False, "attempts": attempts, "last_status": last_status, "error": f"client_error {resp.status_code}", "skipped": False}
                last_error = f"http_{resp.status_code}"
                logger.warning(f"Callback async {filename} attempt {attempt} got {resp.status_code}, retry...")
            except (httpx.ConnectError, httpx.TimeoutException, httpx.NetworkError, httpx.ReadError) as e:
                last_error = f"network:{type(e).__name__}:{str(e)[:200]}"
                logger.warning(f"Callback async network error {attempt}: {e}")
                _append_log(CALLBACK_LOG_FILE, {"ts": datetime.now(JAKARTA_TZ).isoformat(), "filename": filename, "attempt": attempt, "error": last_error})
            except Exception as e:
                last_error = f"unexpected:{type(e).__name__}:{str(e)[:200]}"
                logger.error(f"Callback async unexpected {attempt}: {e}", exc_info=True)
                _append_log(CALLBACK_LOG_FILE, {"ts": datetime.now(JAKARTA_TZ).isoformat(), "filename": filename, "attempt": attempt, "error": last_error})

            if attempt < CALLBACK_RETRIES:
                backoff = CALLBACK_RETRY_BACKOFF_BASE ** attempt
                jitter = random.uniform(0, 1)
                await asyncio.sleep(backoff + jitter)

    logger.error(f"Callback async failed after {CALLBACK_RETRIES} for {filename}")
    try:
        _append_log(CALLBACK_DEAD_LETTER_FILE, {**payload, "_dead_reason": last_error or "max_retries", "_ts": datetime.now(JAKARTA_TZ).isoformat()})
    except Exception:
        pass
    return {"success": False, "attempts": attempts, "last_status": last_status, "error": last_error, "skipped": False}
