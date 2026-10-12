"""
app/callback/payload.py — Build payload callback ke Laravel dari ocr_result + metadata.

ocr_result: dict dari main.py:136 (app/ocr/store.py:82) — sudah punya filename, cleaned_text, lines, dll.
metadata: dict dari uploads/metadata.json (app/storage.py:17) — punya source, template_id, dll.

Payload final dikirim ke POST {LARAVEL_CALLBACK_URL} dengan headers HMAC.
"""

from typing import Any, Dict, Optional


def build_callback_payload(
    ocr_result: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
    include_raw_text: bool = False,
) -> Dict[str, Any]:
    """
    Bangun payload yang akan dikirim ke Laravel.

    Args:
        ocr_result: Dict hasil OCR (wajib punya 'filename'). Contoh dari main.py:136.
        metadata: Dict metadata dari metadata.json untuk filename yang sama (opsional).
        include_raw_text: Jika True, sertakan raw_text juga (default False untuk hemat).

    Returns:
        Dict payload siap json.dumps dan sign HMAC.

    Raises:
        ValueError: Jika filename tidak ada.
    """
    filename = ocr_result.get("filename")
    if not filename:
        raise ValueError("ocr_result missing 'filename'")

    # Fix Blank: tambah is_empty & empty_reason agar Laravel tahu retake
    is_empty = ocr_result.get("is_empty")
    if is_empty is None:
        # Kontrak 'none' (2026-09-11): cleaned_text 'none' = VALID, bukan kosong.
        try:
            from app.ocr.parser import is_none_text

            if is_none_text(ocr_result.get("cleaned_text", "")):
                is_empty = False
            else:
                _ct = (ocr_result.get("cleaned_text") or "").strip()
                _lines = ocr_result.get("lines") or []
                is_empty = not _ct and not any((ln or "").strip() for ln in _lines)
        except ImportError:
            _ct_raw = (ocr_result.get("cleaned_text") or "")
            if isinstance(_ct_raw, str) and _ct_raw.strip().lower() == "none":
                is_empty = False
            else:
                _ct = _ct_raw.strip() if isinstance(_ct_raw, str) else ""
                _lines = ocr_result.get("lines") or []
                is_empty = not _ct and not any((ln or "").strip() for ln in _lines)

    payload: Dict[str, Any] = {
        "event": "ocr.completed" if not is_empty else "ocr.failed",
        "filename": filename,
        "original_filename": ocr_result.get("original_filename", ""),
        "cleaned_text": ocr_result.get("cleaned_text", ""),
        "lines": ocr_result.get("lines", []),
        "status": ocr_result.get("status", "unknown"),
        "engine": ocr_result.get("engine", "tesseract"),
        "language": ocr_result.get("language", "eng"),
        # OCR Engine Upgrade (2026-09-11): kontrak 'none' — 'none' = success valid.
        "text_present": ocr_result.get("text_present"),
        "det_threshold": ocr_result.get("det_threshold"),
        "rec_threshold": ocr_result.get("rec_threshold"),
        # Tuning RapidOCR (2026-09-12): nama model det/rec agar Laravel bisa
        # tampilkan badge + audit (small=cepat, medium=akurat fix H->K).
        # Aditif — Laravel lama abaikan field tak dikenal.
        "det_model": ocr_result.get("det_model"),
        "rec_model": ocr_result.get("rec_model"),
        "preprocessed": ocr_result.get("preprocessed", True),
        "preprocessing_steps": ocr_result.get("preprocessing_steps", []),
        # Fix Blank fields
        "is_empty": bool(is_empty),
        "empty_reason": ocr_result.get("empty_reason"),
        "preprocessing_attempts": ocr_result.get("preprocessing_attempts", []),
        "preprocessed_pass": ocr_result.get("preprocessed_pass", 1),
        "processed_at": ocr_result.get("processed_at", ""),
        "processing_time_seconds": ocr_result.get("processing_time_seconds"),
        "file_size_bytes": ocr_result.get("file_size_bytes", 0),
    }

    # Tambah info dari metadata jika ada (source, template)
    if metadata and isinstance(metadata, dict):
        # metadata bisa berupa entry langsung atau dict {filename: {...}}
        # Jika metadata adalah full metadata.json, caller sudah ambil entry-nya, tapi handle keduanya
        meta_entry = metadata
        if filename in metadata and isinstance(metadata[filename], dict):
            meta_entry = metadata[filename]

        if isinstance(meta_entry, dict):
            # Field yang relevan untuk Laravel (tidak sensitif)
            for key in ("source", "template_id", "template_name", "template_width", "template_height", "resolution_w", "resolution_h"):
                if key in meta_entry:
                    payload[key] = meta_entry[key]
            # --- Checking Label 2-capture korelasi (PLAN_CHECKING_LABEL_2_CAPTURE_OCR_CALLBACK §5.1) ---
            # Aditif: Laravel lama abaikan field tak dikenal. Kosong bila alur lama.
            if "check_session_id" in meta_entry:
                payload["check_session_id"] = meta_entry.get("check_session_id", "")
            if "label_slot" in meta_entry:
                _slot = meta_entry.get("label_slot", "")
                try:
                    payload["label_slot"] = int(str(_slot).strip()) if str(_slot).strip() != "" else ""
                except (ValueError, TypeError):
                    payload["label_slot"] = _slot
            if "reference_id" in meta_entry:
                payload["reference_id"] = meta_entry.get("reference_id", "")

    # Opsional: raw_text untuk audit (jika diminta)
    if include_raw_text:
        payload["raw_text"] = ocr_result.get("raw_text", "")

    # Error field jika failed atau empty (untuk debugging Laravel)
    if (ocr_result.get("status") == "failed" or is_empty) and ocr_result.get("error"):
        payload["error"] = ocr_result.get("error")
    # Jika empty tapi belum ada error, beri default
    if is_empty and not payload.get("error"):
        payload["error"] = "empty cleaned_text — dark low-contrast, need retake or better lighting"

    return payload


def build_failed_payload(
    ocr_result: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Bangun payload untuk kasus OCR failed (event berbeda).
    Laravel bisa pakai ini untuk alert, bukan untuk simpan cleaned_text.
    """
    payload = build_callback_payload(ocr_result, metadata, include_raw_text=False)
    payload["event"] = "ocr.failed"
    # cleaned_text kosong, tapi kirim error
    return payload
