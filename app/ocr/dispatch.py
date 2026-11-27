"""
app/ocr/dispatch.py — 1 pintu router OCR_ENGINE + bentuk result standar.

- OCR_ENGINE=paddle    → RapidOCR ONNX (deteksi+recognisi, kontrak 'none').
- OCR_ENGINE=vitstr    → ViTSTR-Tiny Vision Transformer (STR Hub PyTorch, ultra-cepat).
- OCR_ENGINE=surya     → Surya OCR (SegFormer + Transformer, presisi tinggi multilingual).
- OCR_ENGINE=tesseract → jalur LAMA 2-pass CLAHE (fallback, perilaku tidak berubah).
- OCR_ENGINE=paddle_native → BELUM AKTIF (stub eksperimental).

Return extract_label_text(): dict standar:
  {"engine","raw_text","cleaned_text","lines","text_present",
   "is_empty","empty_reason","det_threshold","rec_threshold",
   "det_model","rec_model"} — 2 field terakhir ADITIF (Laravel abaikan bila tak dikenal).
"""

from pathlib import Path

from app.config import OCR_DET_THRESHOLD, OCR_ENGINE, OCR_REC_THRESHOLD


def _resolve_profile(profile_override: str | None = None) -> tuple:
    """Resolve (engine, rec_model, cascade_on, profile_name) dari runtime_config atau fallback."""
    try:
        from app.runtime_config import runtime_config, VALID_OCR_PROFILES

        if profile_override and profile_override in VALID_OCR_PROFILES:
            prof = VALID_OCR_PROFILES[profile_override]
            return prof["engine"], prof["rec_model"], prof["cascade"], profile_override
        rc = runtime_config.get_config()
        return rc["ocr_engine"], rc["rapidocr_rec_model"], rc["ocr_cascade_enabled"], rc["ocr_profile"]
    except Exception:
        return (OCR_ENGINE or "paddle"), "small", False, "rapidocr_small"


def _blank_ocr_result(engine_name: str, profile_name: str, blank_info: dict) -> dict:
    """Bangun result OCR standar 'none' saat crop terdeteksi sebagai label kosong."""
    return {
        "engine": engine_name,
        "ocr_profile": profile_name,
        "raw_text": "",
        "cleaned_text": "none",
        "lines": [],
        "text_present": False,
        "is_empty": False,
        "is_blank": True,
        "empty_reason": "blank_label",
        "blank_metrics": blank_info,
        "det_threshold": OCR_DET_THRESHOLD if engine_name == "rapidocr" else None,
        "rec_threshold": OCR_REC_THRESHOLD if engine_name == "rapidocr" else None,
        "det_model": None,
        "rec_model": None,
        "image_scale": 1.0,
        "timings": {"blank_skip": True},
    }


def extract_label_text_from_bytes(
    image_bytes: bytes,
    original_filename: str = "",
    ocr_profile: str | None = None,
) -> dict:
    """
    Ekstrak teks dari bytes (R3) — tanpa tempfile.

    Return standar sama seperti extract_label_text + tambahan aditif:
    image_scale, timings{decode_s,infer_s,det_s,cls_s,rec_s}.
    """
    engine, rec_model, cascade_on, profile_name = _resolve_profile(ocr_profile)
    engine_label = "rapidocr" if engine == "paddle" else engine

    try:
        from app.vision.blank_detector import detect_blank_label_bytes

        blank_info = detect_blank_label_bytes(image_bytes)
        if blank_info.get("is_blank"):
            return _blank_ocr_result(engine_label, profile_name, blank_info)
    except Exception:
        pass

    if engine == "paddle":
        from app.ocr.paddle_engine import active_model_label, extract_from_bytes
        from app.ocr.parser import build_paddle_result

        raw = extract_from_bytes(image_bytes, rec_model=rec_model, cascade=cascade_on)
        built = build_paddle_result(raw["lines_raw"], OCR_DET_THRESHOLD, OCR_REC_THRESHOLD)
        try:
            lbl = active_model_label(rec_model_override=rec_model)
        except Exception:
            lbl = {"det_model": None, "rec_model": None}
        return {
            "engine": "rapidocr",
            "ocr_profile": profile_name,
            "raw_text": built["raw_text"],
            "cleaned_text": built["cleaned_text"],
            "lines": built["lines"],
            "text_present": built["text_present"],
            "is_empty": False,
            "empty_reason": built["empty_reason"],
            "det_threshold": OCR_DET_THRESHOLD,
            "rec_threshold": OCR_REC_THRESHOLD,
            "det_model": lbl.get("det_model"),
            "rec_model": lbl.get("rec_model"),
            "image_scale": raw.get("image_scale", 1.0),
            "timings": raw.get("timings", {}),
        }

    if engine == "vitstr":
        from app.ocr.vitstr_engine import extract_from_bytes as vitstr_extract
        from app.ocr.parser import clean_ocr_text

        out = vitstr_extract(image_bytes)
        text = out.get("text", "")
        cleaned = clean_ocr_text(text)
        text_present = bool(text and text.strip())
        return {
            "engine": "vitstr",
            "ocr_profile": profile_name,
            "raw_text": cleaned["raw_text"],
            "cleaned_text": cleaned["cleaned_text"] if text_present else "none",
            "lines": cleaned["lines"],
            "text_present": text_present,
            "is_empty": False,
            "empty_reason": None if text_present else "no_text_detected",
            "det_threshold": None,
            "rec_threshold": None,
            "det_model": "vitstr_tiny",
            "rec_model": "vitstr_tiny",
            "image_scale": 1.0,
            "timings": out.get("timings", {}),
        }

    if engine == "surya":
        from app.ocr.surya_engine import extract_from_bytes as surya_extract
        from app.ocr.parser import clean_ocr_text

        out = surya_extract(image_bytes)
        raw_text = "\n".join(out.get("lines", []))
        cleaned = clean_ocr_text(raw_text)
        text_present = bool(cleaned["cleaned_text"] and cleaned["cleaned_text"].strip())
        return {
            "engine": "surya",
            "ocr_profile": profile_name,
            "raw_text": cleaned["raw_text"],
            "cleaned_text": cleaned["cleaned_text"] if text_present else "none",
            "lines": cleaned["lines"],
            "text_present": text_present,
            "is_empty": False,
            "empty_reason": None if text_present else "no_text_detected",
            "det_threshold": None,
            "rec_threshold": None,
            "det_model": "surya_det3",
            "rec_model": "surya_rec2",
            "image_scale": 1.0,
            "timings": out.get("timings", {}),
        }

    # Fallback tesseract dari bytes: tulis tempfile lazy hanya saat dipakai.
    import tempfile as _tf

    _tmp = _tf.NamedTemporaryFile(prefix="labelcap_", suffix=".png", delete=False)
    try:
        _tmp.write(image_bytes)
        _tmp.close()
        return extract_label_text(_tmp.name, original_filename, ocr_profile=ocr_profile)
    finally:
        try:
            import os as _os

            _os.unlink(_tmp.name)
        except Exception:
            pass


def extract_label_text(
    image_path: str | Path,
    original_filename: str = "",
    ocr_profile: str | None = None,
) -> dict:
    """
    Ekstrak teks 1 gambar label via engine aktif.

    Args:
        image_path: Path ke file gambar.
        original_filename: Nama asli client (info saja, untuk log).
        ocr_profile: Opsional nama profil untuk override runtime.

    Returns:
        dict standar (lihat modul docstring).
    """
    engine, rec_model, cascade_on, profile_name = _resolve_profile(ocr_profile)
    engine_label = "rapidocr" if engine == "paddle" else engine

    try:
        from app.vision.blank_detector import detect_blank_label_path

        blank_info = detect_blank_label_path(image_path)
        if blank_info.get("is_blank"):
            return _blank_ocr_result(engine_label, profile_name, blank_info)
    except Exception:
        pass

    if engine == "paddle":
        from app.ocr.paddle_engine import active_model_label, extract_text_paddle
        from app.ocr.parser import build_paddle_result

        out = extract_text_paddle(image_path, rec_model=rec_model, cascade=cascade_on)
        built = build_paddle_result(out["lines_raw"], OCR_DET_THRESHOLD, OCR_REC_THRESHOLD)
        try:
            lbl = active_model_label(rec_model_override=rec_model)
        except Exception:
            lbl = {"det_model": None, "rec_model": None}
        return {
            "engine": "rapidocr",
            "ocr_profile": profile_name,
            "raw_text": built["raw_text"],
            "cleaned_text": built["cleaned_text"],
            "lines": built["lines"],
            "text_present": built["text_present"],
            "is_empty": False,  # 'none' adalah nilai valid, bukan kosong
            "empty_reason": built["empty_reason"],
            "det_threshold": OCR_DET_THRESHOLD,
            "rec_threshold": OCR_REC_THRESHOLD,
            "det_model": lbl.get("det_model"),
            "rec_model": lbl.get("rec_model"),
            "image_scale": out.get("image_scale", 1.0),
            "timings": out.get("timings", {}),
        }

    if engine == "vitstr":
        from app.ocr.vitstr_engine import extract_from_path as vitstr_extract_path
        from app.ocr.parser import clean_ocr_text

        out = vitstr_extract_path(image_path)
        text = out.get("text", "")
        cleaned = clean_ocr_text(text)
        text_present = bool(text and text.strip())
        return {
            "engine": "vitstr",
            "ocr_profile": profile_name,
            "raw_text": cleaned["raw_text"],
            "cleaned_text": cleaned["cleaned_text"] if text_present else "none",
            "lines": cleaned["lines"],
            "text_present": text_present,
            "is_empty": False,
            "empty_reason": None if text_present else "no_text_detected",
            "det_threshold": None,
            "rec_threshold": None,
            "det_model": "vitstr_tiny",
            "rec_model": "vitstr_tiny",
            "image_scale": 1.0,
            "timings": out.get("timings", {}),
        }

    if engine == "surya":
        from app.ocr.surya_engine import extract_from_path as surya_extract_path
        from app.ocr.parser import clean_ocr_text

        out = surya_extract_path(image_path)
        raw_text = "\n".join(out.get("lines", []))
        cleaned = clean_ocr_text(raw_text)
        text_present = bool(cleaned["cleaned_text"] and cleaned["cleaned_text"].strip())
        return {
            "engine": "surya",
            "ocr_profile": profile_name,
            "raw_text": cleaned["raw_text"],
            "cleaned_text": cleaned["cleaned_text"] if text_present else "none",
            "lines": cleaned["lines"],
            "text_present": text_present,
            "is_empty": False,
            "empty_reason": None if text_present else "no_text_detected",
            "det_threshold": None,
            "rec_threshold": None,
            "det_model": "surya_det3",
            "rec_model": "surya_rec2",
            "image_scale": 1.0,
            "timings": out.get("timings", {}),
        }

    if OCR_ENGINE == "paddle_native":
        from app.ocr.paddle_native_engine import extract_label_text_native

        return extract_label_text_native(image_path, original_filename)

    # --- Fallback: Tesseract 2-pass + R10 timeout & time budget ---
    import time as _time

    from app.ocr.engine import extract_text_tesseract, get_tesseract_cmd
    from app.ocr.parser import clean_ocr_text
    from app.ocr.preprocess import preprocess_v1, preprocess_v2_adaptive

    try:
        from app import config as _cfg

        _tess_to = float(getattr(_cfg, "OCR_TESSERACT_TIMEOUT_SECONDS", 10) or 10)
    except Exception:
        _tess_to = 10.0
    _t0 = _time.perf_counter()

    preprocessing_steps = ["grayscale", "clahe", "autocontrast", "upscale_2x", "sharpen", "medianBlur_3"]
    preprocessing_attempts = ["v1_psm6"]
    used_pass = 1

    preprocessed = preprocess_v1(str(image_path))
    try:
        raw_text = extract_text_tesseract(preprocessed, timeout=_tess_to)
    except Exception:
        # Timeout pytesseract raise; anggap blank agar fallback/time-budget jalan.
        raw_text = ""
        preprocessing_attempts.append("v1_timeout")
    cleaned = clean_ocr_text(raw_text)

    def _empty(c) -> bool:
        if not c:
            return True
        ct = c.get("cleaned_text", "")
        lines = c.get("lines", [])
        if not ct or not ct.strip():
            return True
        return not (lines and any(ln.strip() for ln in lines if isinstance(ln, str)))

    if _empty(cleaned):
        preprocessed_v2 = preprocess_v2_adaptive(str(image_path))
        best_cleaned, best_raw, best_psm, found = cleaned, raw_text, "6", False
        for psm in ["6", "11", "3"]:
            # R10 time budget: hentikan bila total sudah lewat 2x timeout.
            try:
                if (_time.perf_counter() - _t0) > max(5.0, _tess_to * 2):
                    preprocessing_attempts.append("v2_budget_exceeded")
                    break
            except Exception:
                pass
            try:
                import pytesseract

                cmd = get_tesseract_cmd()
                if cmd:
                    pytesseract.pytesseract.tesseract_cmd = cmd
                raw_v2 = pytesseract.image_to_string(
                    preprocessed_v2, lang="eng", config=f"--psm {psm}",
                    timeout=_tess_to if _tess_to and _tess_to > 0 else None,
                )
                cleaned_v2 = clean_ocr_text(raw_v2)
                preprocessing_attempts.append(f"v2_adaptive_psm{psm}")
                if not _empty(cleaned_v2):
                    best_cleaned, best_raw, best_psm = cleaned_v2, raw_v2, psm
                    found = True
                    break
                if len(cleaned_v2.get("cleaned_text", "")) > len(best_cleaned.get("cleaned_text", "")):
                    best_cleaned, best_raw, best_psm = cleaned_v2, raw_v2, psm
            except Exception:
                preprocessing_attempts.append(f"v2_psm{psm}_error")
                continue
        cleaned, raw_text = best_cleaned, best_raw
        preprocessing_steps = preprocessing_steps + ["adaptiveThreshold_11_2", f"psm{best_psm}" if found else "psm6,11,3_all_blank"]
        used_pass = 2

    return {
        "engine": "tesseract",
        "raw_text": cleaned["raw_text"] if cleaned else "",
        "cleaned_text": cleaned["cleaned_text"] if cleaned else "",
        "lines": cleaned["lines"] if cleaned else [],
        "text_present": None,  # konsep text_present hanya untuk paddle
        "is_empty": _empty(cleaned),
        "empty_reason": "no_text_detected_after_2pass" if _empty(cleaned) else None,
        "det_threshold": None,
        "rec_threshold": None,
        "preprocessing_steps": preprocessing_steps,
        "preprocessing_attempts": preprocessing_attempts,
        "preprocessed_pass": used_pass,
    }
