"""
app/ocr/paddle_engine.py — Engine OCR RapidOCR ONNX (model PP-OCR, CPU-only).

Menggantikan Tesseract sebagai engine default (2026-09-11 / OCR_ENGINE_UPGRADE).
- Deteksi teks (DB) + recognisi (CRNN) dalam 1 panggilan.
- Input = file ORIGINAL atau bytes (R3: bytes tanpa tempfile).
- Gambar blank/object-only → lines_raw=[] (aturan 'none' ada di parser, bukan di sini).
- R1: engine thread-local (per worker thread) + warm-up, bukan 1 global + Lock serial.
- R2: pre-resize sisi terpanjang (OCR_MAX_SIDE_PX) + Det.limit_side_len.
- R6: tuning thread ONNX (intra/inter_op).
- R7: cascade rec small -> medium selektif (default OFF).
- R8: angle classifier opsional (Global.use_cls).
- Fase 0: timings per tahap (decode/infer/det/cls/rec) + image_scale.

Terverifikasi: rapidocr 3.9.2 + onnxruntime 1.30.0, Python 3.14 Windows CPU.
API: RapidOCR()(path|ndarray|bytes) -> RapidOCROutput(boxes, txts, scores, elapse_list).
"""

import logging
import threading
import time
from pathlib import Path

from app.config import (
    OCR_MIN_TEXT_LEN,
    OCR_MODEL_DIR,
    RAPIDOCR_BOX_THRESH,
    RAPIDOCR_DET_MODEL,
    RAPIDOCR_DET_VERSION,
    RAPIDOCR_REC_LANG,
    RAPIDOCR_REC_MODEL,
    RAPIDOCR_REC_VERSION,
    RAPIDOCR_TEXT_SCORE,
    RAPIDOCR_UNCLIP_RATIO,
)

logger = logging.getLogger(__name__)

# Legacy global singleton (dipertahankan untuk kompatibilitas get_ocr_engine).
_engine = None
_engine_lock = threading.Lock()

# R1: engine per-thread (paling aman bila wrapper RapidOCR belum tentu stateless).
_tls = threading.local()

# Mapping nilai .env (string) -> Enum rapidocr. Diluar ini = fallback default aman.
_VERSION_MAP = {"PP-OCRv4": "PPOCRV4", "PP-OCRv5": "PPOCRV5", "PP-OCRv6": "PPOCRV6"}
_MODEL_MAP = {
    "tiny": "TINY",
    "small": "SMALL",
    "medium": "MEDIUM",
    "mobile": "MOBILE",
    "server": "SERVER",
}


def _build_engine_params(rec_model_override: str | None = None) -> dict:
    """
    Susun params dotted-key untuk RapidOCR(params={...}) dari ENV.

    Return {} bila config tak valid / rapidocr tak mendukung → pemanggil pakai
    default bawaan paket (perilaku lama). TIDAK PERNAH raise.
    """
    try:
        from rapidocr.utils.typings import ModelType, OCRVersion
        from app import config as _cfg

        params: dict = {}
        rec_model_name = (rec_model_override or RAPIDOCR_REC_MODEL or "small").lower()
        rec_v = _VERSION_MAP.get((RAPIDOCR_REC_VERSION or "").upper(), "PPOCRV6")
        rec_m = _MODEL_MAP.get(rec_model_name, "SMALL")
        params["Rec.ocr_version"] = OCRVersion[rec_v]
        params["Rec.model_type"] = ModelType[rec_m]
        if RAPIDOCR_REC_LANG:
            params["Rec.lang_type"] = RAPIDOCR_REC_LANG.strip().lower()

        det_v = _VERSION_MAP.get((RAPIDOCR_DET_VERSION or "").upper(), "PPOCRV6")
        det_m = _MODEL_MAP.get((RAPIDOCR_DET_MODEL or "").lower(), "SMALL")
        params["Det.ocr_version"] = OCRVersion[det_v]
        params["Det.model_type"] = ModelType[det_m]

        # R2: batas sisi deteksi (cek config.yaml versi terpasang).
        try:
            det_limit = int(getattr(_cfg, "OCR_DET_LIMIT_SIDE", 0) or 0)
        except Exception:
            det_limit = 0
        if det_limit and det_limit > 0:
            params["Det.limit_side_len"] = det_limit
            params["Det.limit_type"] = "max"

        # R8: angle classifier opsional.
        try:
            use_cls = bool(getattr(_cfg, "OCR_USE_CLS", True))
        except Exception:
            use_cls = True
        params["Global.use_cls"] = use_cls

        # R6: tuning thread ONNX per sesi.
        try:
            intra = int(getattr(_cfg, "OCR_ORT_INTRA_THREADS", 0) or 0)
            inter = int(getattr(_cfg, "OCR_ORT_INTER_THREADS", 1) or 1)
        except Exception:
            intra, inter = 0, 1
        if intra and intra > 0:
            params["EngineConfig.onnxruntime.intra_op_num_threads"] = intra
        if inter and inter > 0:
            params["EngineConfig.onnxruntime.inter_op_num_threads"] = inter

        if OCR_MODEL_DIR:
            p = Path(OCR_MODEL_DIR)
            if p.is_dir():
                params["Det.model_dir"] = str(p)
                params["Rec.model_dir"] = str(p)
            else:
                logger.warning(f"OCR_MODEL_DIR diabaikan (bukan folder): {OCR_MODEL_DIR}")
        return params
    except Exception as e:
        logger.warning(f"Gagal susun RapidOCR params, pakai default bawaan: {e}")
        return {}


def active_model_label(rec_model_override: str | None = None) -> dict:
    """Label konfigurasi model aktif (untuk JSON/health, aditif, bukan path file)."""
    rec_m = (rec_model_override or RAPIDOCR_REC_MODEL or "small").lower()
    return {
        "det_model": f"{RAPIDOCR_DET_VERSION}_det_{RAPIDOCR_DET_MODEL}",
        "rec_model": f"{RAPIDOCR_REC_VERSION}_rec_{rec_m}",
        "rec_lang": RAPIDOCR_REC_LANG,
        "box_thresh": RAPIDOCR_BOX_THRESH,
        "text_score": RAPIDOCR_TEXT_SCORE,
        "unclip_ratio": RAPIDOCR_UNCLIP_RATIO,
    }


def _build_engine(rec_model: str):
    """Bangun 1 instance RapidOCR untuk model rec tertentu."""
    try:
        from rapidocr import RapidOCR
    except ImportError as e:
        raise RuntimeError(
            "rapidocr/onnxruntime belum install. Jalankan: pip install rapidocr onnxruntime"
        ) from e
    params = _build_engine_params(rec_model_override=rec_model)
    eng = RapidOCR(params=params) if params else RapidOCR()
    return eng


def _get_tls_engine(rec_model: str):
    """Ambil atau inisialisasi instance RapidOCR per-model di thread ini."""
    if not hasattr(_tls, "engines_by_model"):
        _tls.engines_by_model = {}
    model_key = (rec_model or "small").strip().lower()
    if model_key not in _tls.engines_by_model:
        with _engine_lock:
            if model_key not in _tls.engines_by_model:
                _tls.engines_by_model[model_key] = _build_engine(model_key)
                logger.info(f"[OCR ENGINE] Loaded thread-local RapidOCR ({model_key})")
    return _tls.engines_by_model[model_key]


def get_ocr_engine():
    """Load singleton RapidOCR sekali (lazy, thread-safe) — LEGACY API."""
    fast, _ = _engines()
    return fast


def _engines(rec_model_override: str | None = None, cascade_override: bool | None = None):
    """
    Engine per thread dinamis — R1.
    Mendukung pergantian profil runtime tanpa reload jika model sudah ter-cache di thread.
    """
    try:
        from app.runtime_config import runtime_config
        rc = runtime_config.get_config()
        cascade_on = rc.get("ocr_cascade_enabled", False) if cascade_override is None else cascade_override
        default_rec = rc.get("rapidocr_rec_model", "small")
    except Exception:
        cascade_on = bool(OCR_CASCADE_ENABLED) if cascade_override is None else cascade_override
        default_rec = RAPIDOCR_REC_MODEL or "small"

    fast_model = (rec_model_override or default_rec).strip().lower()
    fast_engine = _get_tls_engine(fast_model)

    acc_engine = None
    if cascade_on:
        acc_engine = _get_tls_engine("medium")

    _tls.fast = fast_engine
    _tls.acc = acc_engine
    return fast_engine, acc_engine


def is_paddle_available() -> bool:
    """Cek apakah rapidocr + onnxruntime bisa diimport (tanpa load model)."""
    try:
        __import__("rapidocr")
        __import__("onnxruntime")
        return True
    except ImportError:
        return False


def warmup_engine() -> bool:
    """Pre-load model saat startup agar inferensi pertama tidak lambat. Return True jika OK."""
    try:
        _engines()
        return True
    except Exception as e:
        logger.warning(f"RapidOCR warmup failed: {e}")
        return False


def warmup_engines() -> bool:
    """
    Warm-up di tiap worker thread OCR (R1) — dipanggil dari lifespan.

    Menjalankan inferensi dummy agar sesi ONNX + model terinisialisasi penuh,
    bukan sekadar load objek. Aman dipanggil walau OCR_EXECUTOR belum siap
    (fallback ke thread ini saja).
    """
    try:
        import cv2
        import numpy as np

        dummy = np.full((320, 640, 3), 255, dtype=np.uint8)
        try:
            cv2.putText(dummy, "WARMUP 123", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
        except Exception:
            pass
        # Coba warm-up paralel di tiap worker bila executor tersedia.
        try:
            from app.ocr.executor import OCR_EXECUTOR
            from app import config as _cfg

            n = max(1, int(getattr(_cfg, "OCR_WORKERS", 1) or 1))

            def _w():
                fast, _ = _engines()
                try:
                    fast(dummy)
                except Exception as e:
                    logger.warning(f"Warmup inferensi gagal: {e}")

            futs = [OCR_EXECUTOR.submit(_w) for _ in range(n)]
            for f in futs:
                try:
                    f.result(timeout=120)
                except Exception as e:
                    logger.warning(f"Warmup worker gagal: {e}")
            print(f"[OCR ENGINE] Warmup done on {n} worker(s)")
            return True
        except Exception:
            # Fallback: warm-up thread ini saja.
            fast, _ = _engines()
            try:
                fast(dummy)
            except Exception as e:
                logger.warning(f"Warmup inferensi gagal: {e}")
            return True
    except Exception as e:
        logger.warning(f"RapidOCR warmup_engines failed: {e}")
        return False


def decode_and_limit(image_bytes: bytes, max_side: int):
    """
    Decode bytes → BGR ndarray + batasi sisi terpanjang (R2).

    Return (img, scale). scale = max_side / max(h,w) bila di-resize, else 1.0.
    boxes hasil OCR berada pada koordinat gambar resize; bagi dengan scale
    untuk kembali ke koordinat asli bila diperlukan konsumen.
    """
    import cv2
    import numpy as np

    if not image_bytes:
        raise ValueError("image_decode_failed: empty bytes")
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("image_decode_failed")
    h, w = img.shape[:2]
    m = max(h, w)
    scale = 1.0
    try:
        ms = int(max_side or 0)
    except Exception:
        ms = 0
    if ms and m > ms:
        scale = ms / float(m)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img, float(scale)


def _crop_box(img, box, pad: int = 4):
    """Crop region box + padding, aman terhadap batas gambar."""
    import numpy as np

    try:
        pts = np.array(box, dtype=np.int32)
        x0, y0 = int(pts[:, 0].min()) - pad, int(pts[:, 1].min()) - pad
        x1, y1 = int(pts[:, 0].max()) + pad, int(pts[:, 1].max()) + pad
        h, w = img.shape[:2]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(h, y1)
        if x1 <= x0 or y1 <= y0:
            return img
        return img[y0:y1, x0:x1]
    except Exception:
        return img


def looks_critical(text: str) -> bool:
    """True bila baris berpola kritis (batch/expiry/ukuran/angka) — R7."""
    try:
        t = (text or "").lower()
        if any(k in t for k in ("batch", "exp", "mfg", "lot", "ml", "gr", "kg", "g ", "l ", "pcs")):
            return True
        return any(c.isdigit() for c in t)
    except Exception:
        return False


def _parse_output(output, timings_extra: dict | None = None) -> dict:
    """Ubah RapidOCROutput → {boxes, lines_raw, timings} + sorting atas→bawah."""
    boxes = getattr(output, "boxes", None)
    txts = getattr(output, "txts", None)
    scores = getattr(output, "scores", None)
    elapse = list(getattr(output, "elapse_list", None) or [])

    def _f(v):
        try:
            return round(float(v), 4) if isinstance(v, (int, float)) and v else None
        except Exception:
            return None

    timings = {
        "det_s": _f(elapse[0]) if len(elapse) > 0 else None,
        "cls_s": _f(elapse[1]) if len(elapse) > 1 else None,
        "rec_s": _f(elapse[2]) if len(elapse) > 2 else None,
    }
    if timings_extra:
        timings.update(timings_extra)

    if boxes is None or txts is None or scores is None:
        return {"boxes": [], "lines_raw": [], "timings": timings}

    rows = []
    try:
        n = len(txts)
    except TypeError:
        return {"boxes": [], "lines_raw": [], "timings": timings}

    for i in range(n):
        try:
            text = str(txts[i]).strip() if txts[i] is not None else ""
        except Exception:
            text = ""
        if len(text) < OCR_MIN_TEXT_LEN:
            continue
        try:
            rec_score = float(scores[i])
        except Exception:
            rec_score = 0.0
        try:
            pts = boxes[i]
            y_min = float(min(p[1] for p in pts))
            x_min = float(min(p[0] for p in pts))
            box_list = [[float(x), float(y)] for x, y in pts]
        except Exception:
            y_min, x_min, box_list = 0.0, 0.0, []
        rows.append(
            {
                "text": text,
                "det_score": None,
                "rec_score": rec_score,
                "y_min": y_min,
                "x_min": x_min,
                "box": box_list,
            }
        )

    rows.sort(key=lambda r: (round(r["y_min"] / 10), r["x_min"]))

    # R7 cascade: verifikasi selektif dengan model medium (hanya bila acc tersedia).
    try:
        from app import config as _cfg

        cascade_conf = float(getattr(_cfg, "OCR_CASCADE_CONF", 0.88) or 0.88)
    except Exception:
        cascade_conf = 0.88

    # acc diambil dari thread-local (dibuat saat _engines).
    acc = getattr(_tls, "acc", None)
    if acc is not None and rows:
        # Butuh img untuk crop — disimpan di _tls sementara oleh extract_from_bytes.
        img_ref = getattr(_tls, "_last_img", None)
        for r in rows:
            try:
                sc = float(r.get("rec_score") or 0.0)
            except Exception:
                sc = 0.0
            if sc >= cascade_conf and not looks_critical(r.get("text", "")):
                continue
            if img_ref is None:
                continue
            try:
                crop = _crop_box(img_ref, r["box"], pad=4)
                if crop is None or getattr(crop, "size", 0) == 0:
                    continue
                rr = acc(crop, use_det=False, use_cls=False, use_rec=True)
                rtxts = list(getattr(rr, "txts", None) or [])
                rscores = list(getattr(rr, "scores", None) or [])
                if rtxts and rscores:
                    try:
                        ns = float(rscores[0])
                    except Exception:
                        ns = sc
                    nt = str(rtxts[0]).strip() if rtxts[0] is not None else ""
                    # Pakai hasil medium hanya bila tidak lebih buruk.
                    if nt and ns >= sc:
                        r["text"] = nt
                        r["rec_score"] = ns
            except Exception as e:
                logger.warning(f"Cascade verify gagal: {e}")
                continue
        # Bersihkan referensi sementara.
        try:
            _tls._last_img = None
        except Exception:
            pass

    lines_raw = [
        {"text": r["text"], "det_score": r["det_score"], "rec_score": r["rec_score"]}
        for r in rows
    ]
    boxes_out = [r["box"] for r in rows]
    return {"boxes": boxes_out, "lines_raw": lines_raw, "timings": timings}


def extract_from_bytes(
    image_bytes: bytes,
    rec_model: str | None = None,
    cascade: bool | None = None,
) -> dict:
    """
    Jalankan deteksi + recognisi dari bytes (R2+R3) — entry utama baru.

    Return {"boxes","lines_raw","image_scale","timings":{decode_s,infer_s,det_s,cls_s,rec_s}}.
    """
    from app import config as _cfg

    t0 = time.perf_counter()
    max_side = int(getattr(_cfg, "OCR_MAX_SIDE_PX", 0) or 0)
    img, scale = decode_and_limit(image_bytes, max_side)
    t_decode = time.perf_counter() - t0

    fast, _ = _engines(rec_model_override=rec_model, cascade_override=cascade)
    # Simpan ref untuk cascade crop (dibersihkan di _parse_output).
    try:
        _tls._last_img = img
    except Exception:
        pass
    t1 = time.perf_counter()
    out = fast(
        img,
        box_thresh=RAPIDOCR_BOX_THRESH,
        text_score=RAPIDOCR_TEXT_SCORE,
        unclip_ratio=RAPIDOCR_UNCLIP_RATIO,
    )
    t_infer = time.perf_counter() - t1

    parsed = _parse_output(out, {"decode_s": round(t_decode, 4), "infer_s": round(t_infer, 4)})
    parsed["image_scale"] = round(float(scale), 4)
    return parsed


def extract_text_paddle(
    image_path: str | Path,
    rec_model: str | None = None,
    cascade: bool | None = None,
) -> dict:
    """
    Jalankan deteksi + recognisi pada file gambar.

    Dipertahankan untuk kompatibilitas (dispatch lama, check batch).
    Implementasi kini: baca bytes → decode_and_limit → engine(ndarray),
    sehingga R1+R2 otomatis berlaku juga untuk jalur path.
    Return {"boxes","lines_raw","image_scale","timings"} (aditif).
    """
    # Baca bytes sekali (R3) agar decode terpusat + kena resize R2.
    try:
        data = Path(image_path).read_bytes()
    except Exception:
        fast, _ = _engines(rec_model_override=rec_model, cascade_override=cascade)
        t1 = time.perf_counter()
        output = fast(
            str(image_path),
            box_thresh=RAPIDOCR_BOX_THRESH,
            text_score=RAPIDOCR_TEXT_SCORE,
            unclip_ratio=RAPIDOCR_UNCLIP_RATIO,
        )
        t_infer = time.perf_counter() - t1
        parsed = _parse_output(output, {"decode_s": None, "infer_s": round(t_infer, 4)})
        parsed["image_scale"] = 1.0
        return parsed
    parsed = extract_from_bytes(data, rec_model=rec_model, cascade=cascade)
    return parsed
