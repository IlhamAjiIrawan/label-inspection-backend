"""
app/ocr/dispatch.py — 1 pintu router OCR_ENGINE + bentuk result standar.

- OCR_ENGINE=paddle    → RapidOCR ONNX (deteksi+recognisi, kontrak 'none').
- OCR_ENGINE=tesseract → jalur LAMA 2-pass CLAHE (fallback, perilaku tidak berubah).
- OCR_ENGINE=paddle_native → BELUM AKTIF (stub eksperimental
  app/ocr/paddle_native_engine.py, butuh paddlepaddle yang tak ada wheel py3.14).

Return extract_label_text(): dict standar:
  {"engine","raw_text","cleaned_text","lines","text_present",
   "is_empty","empty_reason","det_threshold","rec_threshold",
   "det_model","rec_model"} — 2 field terakhir ADITIF (Laravel abaikan bila tak dikenal).
"""

from pathlib import Path

from app.config import OCR_DET_THRESHOLD, OCR_ENGINE, OCR_REC_THRESHOLD


def extract_label_text_from_bytes(image_bytes: bytes, original_filename: str = "") -> dict:
    """
    Ekstrak teks dari bytes (R3) — tanpa tempfile.

    Return standar sama seperti extract_label_text + tambahan aditif:
    image_scale, timings{decode_s,infer_s,det_s,cls_s,rec_s}.
    """
    if OCR_ENGINE == "paddle":
        from app.ocr.paddle_engine import extract_from_bytes
        from app.ocr.parser import build_paddle_result

        raw = extract_from_bytes(image_bytes)
        built = build_paddle_result(raw["lines_raw"], OCR_DET_THRESHOLD, OCR_REC_THRESHOLD)
        try:
            from app.ocr.paddle_engine import active_model_label

            lbl = active_model_label()
        except Exception:
            lbl = {"det_model": None, "rec_model": None}
        return {
            "engine": "rapidocr",
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

    # Fallback tesseract dari bytes: tulis tempfile lazy hanya saat dipakai.
    import tempfile as _tf

    _tmp = _tf.NamedTemporaryFile(prefix="labelcap_", suffix=".png", delete=False)
    try:
        _tmp.write(image_bytes)
        _tmp.close()
        return extract_label_text(_tmp.name, original_filename)
    finally:
        try:
            import os as _os

            _os.unlink(_tmp.name)
        except Exception:
            pass


def extract_label_text(image_path: str | Path, original_filename: str = "") -> dict:
    """
    Ekstrak teks 1 gambar label via engine aktif.

    Args:
        image_path: Path ke file gambar.
        original_filename: Nama asli client (info saja, untuk log).

    Returns:
        dict standar (lihat modul docstring). Untuk paddle:
        tidak ada teks → cleaned_text "none", text_present False,
        is_empty False, empty_reason "no_text_detected".
    """
    if OCR_ENGINE == "paddle":
        from app.ocr.paddle_engine import active_model_label, extract_text_paddle
        from app.ocr.parser import build_paddle_result

        out = extract_text_paddle(image_path)
        built = build_paddle_result(out["lines_raw"], OCR_DET_THRESHOLD, OCR_REC_THRESHOLD)
        try:
            lbl = active_model_label()
        except Exception:
            lbl = {"det_model": None, "rec_model": None}
        return {
            "engine": "rapidocr",
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

    if OCR_ENGINE == "paddle_native":
        # Stub eksperimental — selalu raise RuntimeError informatif sampai §6.0 hijau.
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
