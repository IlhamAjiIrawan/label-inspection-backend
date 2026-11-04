"""
app/ocr/surya_engine.py — Engine Surya OCR (Multilingual Document & Apparel Label OCR).

Model berkemampuan tinggi membaca teks label multi-baris, kode RN, nomor serial,
serta mendukung 90+ bahasa.
"""

import io
import logging
import threading
import time
from pathlib import Path
from PIL import Image

logger = logging.getLogger(__name__)

_det_model = None
_det_processor = None
_rec_model = None
_rec_processor = None
_model_lock = threading.Lock()


def is_surya_available() -> bool:
    """Cek apakah library surya-ocr terpasang."""
    try:
        from surya.ocr import run_ocr
        return True
    except ImportError:
        return False


def get_surya_models():
    """Lazy loader singleton untuk model deteksi dan rekognisi Surya OCR."""
    global _det_model, _det_processor, _rec_model, _rec_processor

    if _rec_model is not None:
        return _det_model, _det_processor, _rec_model, _rec_processor

    with _model_lock:
        if _rec_model is not None:
            return _det_model, _det_processor, _rec_model, _rec_processor

        logger.info("[Surya] Memuat model deteksi dan rekognisi Surya OCR...")
        try:
            from surya.model.detection.model import load_model as load_det_model, load_processor as load_det_processor
        except ImportError:
            from surya.model.detection.segformer import load_model as load_det_model, load_processor as load_det_processor

        from surya.model.recognition.model import load_model as load_rec_model
        from surya.model.recognition.processor import load_processor as load_rec_processor

        _det_processor = load_det_processor()
        _det_model = load_det_model()
        _rec_model = load_rec_model()
        _rec_processor = load_rec_processor()

        logger.info("[Surya] Seluruh model Surya OCR siap digunakan.")
        return _det_model, _det_processor, _rec_model, _rec_processor


def extract_from_bytes(image_bytes: bytes, langs: list[str] | None = None) -> dict:
    """Ekstrak teks dari raw bytes gambar menggunakan Surya OCR."""
    t0 = time.perf_counter()
    det_model, det_processor, rec_model, rec_processor = get_surya_models()
    t_load = time.perf_counter() - t0

    from surya.ocr import run_ocr

    t_dec0 = time.perf_counter()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    t_decode = time.perf_counter() - t_dec0

    active_langs = langs or ["en"]

    t_inf0 = time.perf_counter()
    predictions = run_ocr(
        [img],
        [active_langs],
        det_model,
        det_processor,
        rec_model,
        rec_processor,
    )
    t_infer = time.perf_counter() - t_inf0

    lines = []
    confidences = []
    if predictions and len(predictions) > 0:
        page = predictions[0]
        text_lines = getattr(page, "text_lines", None)
        if text_lines is None and isinstance(page, dict):
            text_lines = page.get("text_lines", [])

        if text_lines:
            for line in text_lines:
                text = getattr(line, "text", None) if not isinstance(line, dict) else line.get("text")
                conf = getattr(line, "confidence", None) if not isinstance(line, dict) else line.get("confidence")
                if text and text.strip():
                    lines.append(text.strip())
                    if conf is not None:
                        confidences.append(float(conf))

    avg_conf = sum(confidences) / len(confidences) if confidences else 1.0

    return {
        "lines": lines,
        "confidence": avg_conf,
        "timings": {
            "load_s": round(t_load, 4),
            "decode_s": round(t_decode, 4),
            "infer_s": round(t_infer, 4),
            "total_s": round(time.perf_counter() - t0, 4),
        },
    }


def extract_from_path(image_path: str | Path, langs: list[str] | None = None) -> dict:
    """Ekstrak teks dari file gambar path."""
    with open(image_path, "rb") as f:
        return extract_from_bytes(f.read(), langs=langs)

