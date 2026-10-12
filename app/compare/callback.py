"""
app/compare/callback.py — Kirim hasil Visual Compare ke eksternal service (Laravel).

POST compare_result.json (1x per batch) ke CHECK_COMPARE_CALLBACK_URL dengan:
  - Header X-API-Key (LARAVEL_API_KEY)
  - Header X-Signature: sha256=<HMAC-SHA256> (reuse app/callback/signer.py)
  - Header X-Timestamp (ISO8601, anti-replay di Laravel)

Aturan (meniru callback OCR):
  - CHECK_COMPARE_CALLBACK_ENABLED=false → skip (return skipped).
  - HTTP 401/404/422 → NO-RETRY (401 key salah, 404 route Laravel belum ada,
    422 validasi) — hanya di-log, compare tetap dianggap sukses.
  - Timeout / 5xx / network error → retry 3x exponential backoff.
  - Apapun hasilnya SELALU print blok [COMPARE CALLBACK] ke console (sementara).

Dipakai oleh: routers/check.py::_process_batch_compare() (background task).
"""

import json
import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

JAKARTA_TZ = timezone(timedelta(hours=7))

# HTTP status yang TIDAK di-retry (percuma: perlu perbaikan config/kode, bukan waktu)
NO_RETRY_STATUS = {401, 404, 422}


def send_compare_callback(
    result: dict,
    timeout: float = 5.0,
    retries: int = 3,
    backoff_base: float = 2.0,
) -> dict:
    """
    POST hasil compare ke Laravel + print ke console.

    Args:
        result: Dict hasil compare (isi compare_result.json apa adanya).
        timeout: Timeout HTTP per attempt (detik).
        retries: Jumlah maksimal attempt.
        backoff_base: Base exponential backoff (attempt ke-n tunggu base**n detik).

    Returns:
        dict: {"success": bool, "http_code": int|None, "error": str|None,
               "skipped_reason": str|None, "attempts": int}
    """
    batch_code = result.get("batch_code", "?")

    # --- Lazy config (agar import tetap aman saat testing) ---
    try:
        from app.config import (
            CHECK_COMPARE_CALLBACK_URL as callback_url,
            CHECK_COMPARE_CALLBACK_ENABLED as cb_enabled,
            LARAVEL_API_KEY as api_key,
            LARAVEL_HMAC_SECRET as hmac_secret,
            CALLBACK_ALLOWED_HOSTS as allowed_hosts,
        )
    except ImportError:
        callback_url = ""
        cb_enabled = True
        api_key = ""
        hmac_secret = ""
        allowed_hosts = set()

    print("=" * 60)
    print(f"  [COMPARE CALLBACK] batch={batch_code}")
    print(f"  URL: {callback_url or '(belum dikonfigurasi)'}")
    print("=" * 60)

    # --- Skip jika dimatikan ---
    if not cb_enabled:
        msg = "CHECK_COMPARE_CALLBACK_ENABLED=false — callback di-skip"
        print(f"[COMPARE CALLBACK] SKIP: {msg}")
        logger.info(f"[COMPARE CALLBACK] skip {batch_code}: {msg}")
        return {"success": False, "http_code": None, "error": None,
                "skipped_reason": msg, "attempts": 0}

    if not callback_url:
        msg = "CHECK_COMPARE_CALLBACK_URL kosong — callback di-skip"
        print(f"[COMPARE CALLBACK] SKIP: {msg}")
        logger.warning(f"[COMPARE CALLBACK] skip {batch_code}: {msg}")
        return {"success": False, "http_code": None, "error": None,
                "skipped_reason": msg, "attempts": 0}

    # --- Guard SSRF: allowlist host (reuse pola callback OCR) ---
    try:
        from urllib.parse import urlparse

        host = (urlparse(callback_url).hostname or "").lower()
        if allowed_hosts and host and host not in allowed_hosts:
            msg = f"host {host} tidak di allowlist {sorted(allowed_hosts)} — callback di-skip"
            print(f"[COMPARE CALLBACK] BLOCKED: {msg}")
            logger.warning(f"[COMPARE CALLBACK] blocked {batch_code}: {msg}")
            return {"success": False, "http_code": None, "error": msg,
                    "skipped_reason": msg, "attempts": 0}
    except Exception:
        pass

    # --- Siapkan headers (API key + HMAC + timestamp) ---
    body_json = json.dumps(result, ensure_ascii=False, separators=(",", ":"))
    timestamp = datetime.now(JAKARTA_TZ).isoformat()
    headers = {"Content-Type": "application/json", "X-Timestamp": timestamp}
    if api_key:
        headers["X-API-Key"] = api_key
    try:
        from app.callback.signer import build_signature_header

        if hmac_secret:
            headers["X-Signature"] = build_signature_header(result, timestamp, hmac_secret)
    except Exception as e:
        logger.warning(f"[COMPARE CALLBACK] sign failed (lanjut tanpa signature): {e}")

    # --- Kirim dengan retry ---
    try:
        import httpx
    except ImportError as e:
        msg = f"httpx belum install: {e}"
        print(f"[COMPARE CALLBACK] SKIP: {msg}")
        return {"success": False, "http_code": None, "error": msg,
                "skipped_reason": msg, "attempts": 0}

    last_error: Optional[str] = None
    for attempt in range(1, retries + 1):
        try:
            with httpx.Client(timeout=httpx.Timeout(timeout, connect=2.0)) as client:
                resp = client.post(callback_url, content=body_json.encode("utf-8"), headers=headers)
            code = resp.status_code
            if 200 <= code < 300:
                print(f"[COMPARE CALLBACK] OK HTTP {code} (attempt {attempt}) — {resp.text[:200]}")
                logger.info(f"[COMPARE CALLBACK] OK {batch_code} HTTP {code} attempt {attempt}")
                return {"success": True, "http_code": code, "error": None,
                        "skipped_reason": None, "attempts": attempt}
            # Non-2xx
            last_error = f"HTTP {code}: {resp.text[:300]}"
            if code in NO_RETRY_STATUS:
                hint = {401: "cek API key sinkron Laravel",
                        404: "route Laravel belum ada (lihat .md §7) — hanya di-log",
                        422: "payload ditolak Laravel"}.get(code, "")
                print(f"[COMPARE CALLBACK] HTTP {code} NO-RETRY: {resp.text[:300]} {hint}")
                logger.warning(f"[COMPARE CALLBACK] {batch_code} HTTP {code} no-retry: {resp.text[:200]}")
                return {"success": False, "http_code": code, "error": last_error,
                        "skipped_reason": None, "attempts": attempt}
            print(f"[COMPARE CALLBACK] HTTP {code} retry {attempt}/{retries}: {resp.text[:200]}")
            logger.warning(f"[COMPARE CALLBACK] {batch_code} HTTP {code} retry {attempt}/{retries}")
        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"[:300]
            print(f"[COMPARE CALLBACK] exception attempt {attempt}/{retries}: {last_error}")
            logger.warning(f"[COMPARE CALLBACK] {batch_code} exception attempt {attempt}: {e}")
        if attempt < retries:
            time.sleep(backoff_base ** attempt)

    print(f"[COMPARE CALLBACK] FAILED setelah {retries}x: {last_error} (compare tetap sukses)")
    logger.error(f"[COMPARE CALLBACK] FAILED {batch_code} after {retries}x: {last_error}")
    return {"success": False, "http_code": None, "error": last_error,
            "skipped_reason": None, "attempts": retries}
