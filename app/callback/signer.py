"""
app/callback/signer.py — HMAC-SHA256 signing untuk callback ke Laravel.

Dipakai untuk header X-Signature: sha256=<hex>
Payload + timestamp di-sign agar Laravel bisa verifikasi keaslian & anti-replay.

Format message:  "{timestamp}.{payload_json_sorted}"
Di mana payload_json_sorted = json.dumps(payload, sort_keys=True, separators=(',',':'), ensure_ascii=False)
"""

import hashlib
import hmac
import json


def sign_payload(payload: dict, timestamp: str, secret: str) -> str:
    """
    Return hex HMAC-SHA256 dari payload + timestamp.

    Args:
        payload: Dict payload yang akan dikirim (sudah final, termasuk callback_attempt).
        timestamp: ISO8601 string dari header X-Timestamp (contoh: 2026-08-29T13:45:00+07:00).
        secret: LARAVEL_HMAC_SECRET (min 32 char).

    Returns:
        Hex string HMAC-SHA256 (64 char hex).
    """
    if not secret:
        raise ValueError("HMAC secret is empty")
    # Canonical JSON: sort_keys + no whitespace + keep unicode
    payload_str = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    message = f"{timestamp}.{payload_str}".encode("utf-8")
    return hmac.new(secret.encode("utf-8"), message, hashlib.sha256).hexdigest()


def verify_signature(payload: dict, timestamp: str, signature: str, secret: str) -> bool:
    """
    Verifikasi signature yang diterima (untuk testing / Laravel reference).

    Args:
        signature: Bisa "sha256=<hex>" atau "<hex>" saja.

    Returns:
        True jika valid, False jika tidak.
    """
    # Strip prefix sha256= jika ada
    if signature.startswith("sha256="):
        signature = signature[len("sha256=") :]

    try:
        expected = sign_payload(payload, timestamp, secret)
    except Exception:
        return False

    # timing-safe compare
    return hmac.compare_digest(expected, signature)


def build_signature_header(payload: dict, timestamp: str, secret: str) -> str:
    """Return header value 'sha256=<hex>'."""
    return f"sha256={sign_payload(payload, timestamp, secret)}"
