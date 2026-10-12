"""
app/ocr/paddle_native_engine.py — Engine PaddleOCR NATIVE (PP-OCRv5/v6, CPU-only).

STATUS: EKSPERIMENTAL / NONAKTIF (2026-09-12 / PADDLEOCR_NATIVE_PLAN §6.0 VERDICT: RED).
- `paddlepaddle` tidak punya wheel untuk Python 3.14 Windows
  ("No matching distribution found", terverifikasi di venv isolasi
  C:\\Users\\Aji\\AppData\\Local\\Temp\\opencode\\paddle-verify).
  `paddleocr` 3.7.0 sendiri ter-download (146KB) tapi butuh inference engine
  (paddlepaddle) untuk jalan → native path TERBLOKIR di env ini.
- File ini adalah implementasi lengkap yang DITULIS dari API PaddleOCR 3.7.0
  (signature PaddleOCR.__init__/predict diinspeksi dari wheel), tapi BELUM
  PERNAH DI-RUN. Jangan jadikan default sebelum:
  1. §6.0 plan hijau (venv Python 3.11/3.12 + `pip install paddleocr paddlepaddle`), dan
  2. `extract_text_native` lolos benchmark `test_native_compare.py` di env itu.
- Peningkatan akurasi yang TERUKUR saat ini dicapai TANPA file ini, via
  RAPIDOCR_REC_MODEL=medium di app/ocr/paddle_engine.py (model PP-OCRv6_rec_medium
  ONNX, tetap runtime onnxruntime CPU, tanpa dependensi baru).

Kontrak (bila kelak diaktifkan): return SAMA seperti paddle_engine —
{"boxes", "lines_raw": [{"text","det_score","rec_score"}]}. Aturan 'none'
tetap diputus di parser, bukan di sini. Singleton + threading.Lock.
"""

import logging
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

_engine = None
_engine_lock = threading.Lock()

# Syarat runtime native: KEDUA paket harus importable.
_NATIVE_IMPORT_ERROR = ""


def is_native_available() -> bool:
    """True hanya jika paddleocr DAN paddle bisa diimport (inference jalan)."""
    try:
        __import__("paddleocr")
        __import__("paddle")
        return True
    except ImportError as e:
        global _NATIVE_IMPORT_ERROR
        _NATIVE_IMPORT_ERROR = str(e)
        return False


def _explain_unavailable() -> str:
    return (
        "PaddleOCR native tidak tersedia di env ini. "
        "Butuh: pip install paddleocr paddlepaddle "
        "(gagal di Python 3.14 Windows — belum ada wheel paddlepaddle; "
        "pakai venv Python 3.11/3.12 atau tunggu wheel baru). "
        f"Detail: {_NATIVE_IMPORT_ERROR or 'import gagal'}. "
        "Fallback: OCR_ENGINE=paddle (RapidOCR ONNX, default) atau tesseract."
    )


def get_native_engine():
    """
    Load singleton PaddleOCR native sekali (lazy, thread-safe, CPU-only).

    Raises:
        RuntimeError: selalu, sampai dependensi terinstall (lihat _explain_unavailable).
    """
    global _engine
    if _engine is not None:
        return _engine
    with _engine_lock:
        if _engine is not None:
            return _engine
        if not is_native_available():
            raise RuntimeError(_explain_unavailable())
        from paddleocr import PaddleOCR

        # Nilai default konservatif untuk teks kecil (selaras RAPIDOCR_* di config):
        # deteksi dilonggarkan, orientasi baris mati (label tegak), CPU.
        # Tambah argumen HANYA yang terbukti ada di PaddleOCR.__init__ 3.7.0.
        _engine = PaddleOCR(
            lang="en",
            ocr_version="PP-OCRv5",
            use_textline_orientation=False,
            text_det_box_thresh=0.3,
            text_rec_score_thresh=0.5,
        )
        logger.info("PaddleOCR native loaded (CPU)")
        print("[OCR ENGINE] PaddleOCR native loaded (CPU)")
        return _engine


def _rows_from_ocr_result(res) -> tuple[list, list]:
    """
    Normalisasi 1 OCRResult PaddleOCR 3.x → (boxes, rows).

    Format kanonik: res.json["res"] = {"rec_texts": [...], "rec_scores": [...],
    "dt_polys": [[[x,y]x4], ...]}. Defensif: struktur tak dikenal → ([], []),
    tidak pernah raise (aturan 'none' diputus di parser).
    """
    try:
        payload = res.json.get("res", {}) if hasattr(res, "json") else {}
        texts = payload.get("rec_texts") or []
        scores = payload.get("rec_scores") or []
        polys = payload.get("dt_polys") or []
    except Exception:
        return [], []
    boxes: list = []
    rows: list = []
    try:
        n = len(texts)
    except TypeError:
        return [], []
    for i in range(n):
        try:
            text = str(texts[i]).strip() if texts[i] is not None else ""
        except Exception:
            text = ""
        if not text:
            continue
        try:
            rec_score = float(scores[i]) if i < len(scores) else None
        except Exception:
            rec_score = None
        try:
            poly = polys[i] if i < len(polys) else []
            box = [[float(x), float(y)] for x, y in poly]
            y_min = min(p[1] for p in box) if box else 0.0
            x_min = min(p[0] for p in box) if box else 0.0
        except Exception:
            box, y_min, x_min = [], 0.0, 0.0
        boxes.append(box)
        rows.append({"text": text, "rec_score": rec_score,
                     "y_min": y_min, "x_min": x_min})
    rows.sort(key=lambda r: (round(r["y_min"] / 10), r["x_min"]))
    lines_raw = [{"text": r["text"], "det_score": None, "rec_score": r["rec_score"]}
                 for r in rows]
    return boxes, lines_raw


def extract_text_native(image_path: str | Path) -> dict:
    """
    Jalankan deteksi + recognisi native pada file gambar ORIGINAL.

    BELUM TERUJI (lihat status modul). Implementasi mengikuti pola
    paddle_engine.extract_text_paddle: filter panjang minimal, urut atas→bawah,
    blank → lines_raw=[] (BUKAN error).

    Raises:
        RuntimeError: jika native tak tersedia (selalu, sampai §6.0 hijau).
    """
    from app.config import OCR_MIN_TEXT_LEN

    engine = get_native_engine()  # raise informatif bila tak tersedia
    path_str = str(image_path)
    with _engine_lock:
        results = engine.predict(path_str)
    if not results:
        return {"boxes": [], "lines_raw": []}
    boxes, lines_raw = _rows_from_ocr_result(results[0])
    kept = [it for it in lines_raw
            if len((it.get("text") or "").strip()) >= OCR_MIN_TEXT_LEN]
    boxes = boxes[: len(kept)] if boxes else []
    return {"boxes": boxes, "lines_raw": kept}


def extract_label_text_native(image_path: str | Path, original_filename: str = "") -> dict:
    """
    Bentuk result standar dispatch untuk engine paddle_native.

    Dipanggil dari app/ocr/dispatch.py bila OCR_ENGINE=paddle_native.
    Saat ini selalu raise RuntimeError informatif (native terkunci).
    """
    from app.config import OCR_DET_THRESHOLD, OCR_REC_THRESHOLD
    from app.ocr.parser import build_paddle_result

    out = extract_text_native(image_path)  # raise bila tak tersedia
    built = build_paddle_result(out["lines_raw"], OCR_DET_THRESHOLD, OCR_REC_THRESHOLD)
    return {
        "engine": "paddle_native",
        "raw_text": built["raw_text"],
        "cleaned_text": built["cleaned_text"],
        "lines": built["lines"],
        "text_present": built["text_present"],
        "is_empty": False,
        "empty_reason": built["empty_reason"],
        "det_threshold": OCR_DET_THRESHOLD,
        "rec_threshold": OCR_REC_THRESHOLD,
        "det_model": "PP-OCRv5_det (native)",
        "rec_model": "PP-OCRv5_rec (native)",
    }
