"""
app/ocr/vitstr_engine.py — Engine OCR ViTSTR (Vision Transformer for Scene Text Recognition).

Model Vision Transformer cepat, ringan (~25 MB), khusus untuk rekognisi teks label.
Dimuat via STR Hub (PyTorch).
"""

import io
import logging
import threading
import time
from pathlib import Path
from PIL import Image

logger = logging.getLogger(__name__)

_model = None
_model_lock = threading.Lock()
_device = None


def is_vitstr_available() -> bool:
    """Cek apakah PyTorch dan dependensi ViTSTR tersedia."""
    try:
        import torch
        return True
    except ImportError:
        return False


def get_vitstr_model():
    """Lazy loader singleton untuk model ViTSTR-Tiny."""
    global _model, _device
    if _model is not None:
        return _model, _device

    with _model_lock:
        if _model is not None:
            return _model, _device

        import torch

        _device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"[ViTSTR] Memuat model ViTSTR pada perangkat: {_device}")
        
        # Load model dari STR Hub (baudm/parseq)
        _model = torch.hub.load("baudm/parseq", "vitstr", pretrained=True, trust_repo=True).eval().to(_device)
        logger.info(f"[ViTSTR] Model berhasil dimuat. img_size: {_model.hparams.img_size}")
        return _model, _device


def _get_image_transform(model):
    """Ambil transform preprocessing resmi dari STR Hub atau fallback torchvision."""
    try:
        from strhub.data.module import SceneTextDataModule
        return SceneTextDataModule.get_transform(model.hparams.img_size)
    except Exception:
        from torchvision import transforms
        return transforms.Compose([
            transforms.Resize(model.hparams.img_size),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),
        ])


def extract_from_bytes(image_bytes: bytes) -> dict:
    """Ekstrak teks dari raw bytes gambar menggunakan ViTSTR."""
    t0 = time.perf_counter()
    model, device = get_vitstr_model()
    t_load = time.perf_counter() - t0

    t_dec0 = time.perf_counter()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    transform = _get_image_transform(model)
    img_tensor = transform(img).unsqueeze(0).to(device)
    t_decode = time.perf_counter() - t_dec0

    t_inf0 = time.perf_counter()
    import torch
    with torch.inference_mode():
        logits = model(img_tensor)
        pred_probs = logits.softmax(-1)
        pred_text, confidence = model.tokenizer.decode(pred_probs)
    t_infer = time.perf_counter() - t_inf0

    text = pred_text[0] if pred_text else ""
    conf_val = float(confidence[0].mean().item()) if confidence is not None and len(confidence) > 0 and hasattr(confidence[0], 'mean') else 1.0

    return {
        "text": text,
        "confidence": conf_val,
        "lines": [text] if text.strip() else [],
        "timings": {
            "load_s": round(t_load, 4),
            "decode_s": round(t_decode, 4),
            "infer_s": round(t_infer, 4),
            "total_s": round(time.perf_counter() - t0, 4),
        },
    }


def extract_from_path(image_path: str | Path) -> dict:
    """Ekstrak teks dari file gambar path."""
    with open(image_path, "rb") as f:
        return extract_from_bytes(f.read())

