"""
app/ocr/engine.py — Engine OCR Tesseract (local, CPU-only).

Satu-satunya engine yang digunakan (tanpa fallback multi-engine) sesuai keputusan.
Menangani: deteksi tesseract binary, ekstraksi teks mentah, graceful handling jika tidak terinstall.
"""

import os
import shutil
import logging
from pathlib import Path
from PIL import Image

from app.config import TESSERACT_CANDIDATE_PATHS, TESSERACT_LANG, TESSERACT_CONFIG

logger = logging.getLogger(__name__)

# Cache hasil deteksi binary agar tidak check berulang
_tesseract_cmd_cached: str | None = None
_tesseract_available_cached: bool | None = None


def _find_tesseract_cmd() -> str | None:
    """
    Cari binary tesseract di system.

    Urutan:
    1. Cek shutil.which (PATH)
    2. Cek kandidat path Windows

    Returns:
        Path string ke binary tesseract, atau None jika tidak ditemukan.
    """
    # 1. Cek PATH
    which = shutil.which("tesseract")
    if which:
        return which

    # 2. Cek kandidat path Windows
    for candidate in TESSERACT_CANDIDATE_PATHS:
        if os.path.isfile(candidate):
            return candidate

    return None


def get_tesseract_cmd() -> str | None:
    """Return path tesseract binary yang terdeteksi (cached)."""
    global _tesseract_cmd_cached, _tesseract_available_cached
    if _tesseract_cmd_cached is not None or _tesseract_available_cached is not None:
        return _tesseract_cmd_cached

    cmd = _find_tesseract_cmd()
    _tesseract_cmd_cached = cmd
    _tesseract_available_cached = cmd is not None
    return cmd


def is_tesseract_available() -> bool:
    """Cek apakah Tesseract tersedia di system (cached)."""
    global _tesseract_available_cached
    if _tesseract_available_cached is not None:
        return _tesseract_available_cached
    cmd = _find_tesseract_cmd()
    # Cache sekalian
    global _tesseract_cmd_cached
    _tesseract_cmd_cached = cmd
    _tesseract_available_cached = cmd is not None
    return _tesseract_available_cached


def get_tesseract_version() -> str | None:
    """Ambil versi tesseract jika tersedia, None jika tidak ada."""
    try:
        import pytesseract

        cmd = get_tesseract_cmd()
        if cmd:
            pytesseract.pytesseract.tesseract_cmd = cmd
        return str(pytesseract.get_tesseract_version())
    except Exception:
        return None


def extract_text_tesseract(preprocessed_image: Image.Image, timeout: float | None = None) -> str:
    """
    Jalankan Tesseract OCR pada gambar yang sudah dipreprocess.

    Berjalan di CPU, tanpa GPU.
    Bahasa: eng (English) sebagai default (TESSERACT_LANG).
    PSM: 6 (single uniform block of text).

    Args:
        preprocessed_image: Gambar PIL yang sudah di-grayscale, upscale, sharpen
                            (output dari app.ocr.preprocess.preprocess_image).

    Returns:
        str: Teks mentah hasil OCR. String kosong jika Tesseract tidak tersedia
             atau gagal, tetapi TIDAK raise exception (graceful). Pemanggil
             dapat mengecek is_tesseract_available() sebelumnya untuk logging.

    Raises:
        RuntimeError: Jika Tesseract tidak ditemukan (opsional, bisa ditangkap pemanggil).
                      Namun implementasi default return "" agar background task tidak crash.
    """
    # Cek ketersediaan
    if not is_tesseract_available():
        # Graceful: log warning dan return kosong — caller akan menyimpan status failed
        logger.warning("Tesseract binary not found. OCR cannot be performed. Install Tesseract OCR first.")
        raise RuntimeError(
            "Tesseract is not installed or not in PATH. "
            "Install Tesseract OCR from https://github.com/UB-Mannheim/tesseract "
            "and ensure it is in system PATH. On Windows try: choco install tesseract"
        )

    try:
        import pytesseract

        cmd = get_tesseract_cmd()
        if cmd:
            pytesseract.pytesseract.tesseract_cmd = cmd

        # R10: timeout + max side guard (default 10s, 2500px via dispatch).
        try:
            from app import config as _cfg

            _to = float(getattr(_cfg, "OCR_TESSERACT_TIMEOUT_SECONDS", 10) or 10)
        except Exception:
            _to = 10.0
        if timeout is not None:
            try:
                _to = float(timeout)
            except Exception:
                pass
        raw_text = pytesseract.image_to_string(
            preprocessed_image,
            lang=TESSERACT_LANG,
            config=TESSERACT_CONFIG,
            timeout=_to if _to and _to > 0 else None,
        )
        return raw_text if raw_text is not None else ""

    except RuntimeError:
        raise
    except Exception as e:
        logger.error(f"Tesseract OCR failed: {e}", exc_info=True)
        raise RuntimeError(f"OCR processing failed: {e}") from e


def extract_text_from_path(image_path: str | Path, preprocess: bool = True) -> str:
    """
    Helper convenience: preprocess + OCR dalam satu panggilan.

    Args:
        image_path: Path ke file gambar.
        preprocess: Jika True, lakukan preprocessing (grayscale/upscale/sharpen).

    Returns:
        Teks mentah hasil OCR.
    """
    if preprocess:
        from app.ocr.preprocess import preprocess_image

        preprocessed = preprocess_image(image_path)
    else:
        preprocessed = Image.open(image_path)

    return extract_text_tesseract(preprocessed)
