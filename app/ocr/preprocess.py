"""
app/ocr/preprocess.py — Preprocessing gambar untuk meningkatkan kualitas OCR.

Fix Retake 401 & Blank (2026-09-06):
- BEFORE: hanya grayscale → upscale 2x LANCZOS → sharpen (Pillow). Histogram dark low-contrast (mean 90, std 49) → Tesseract blank.
- AFTER: CLAHE + autocontrast + upscale 2x + sharpen + medianBlur + adaptiveThreshold fallback (2-pass).
  Terinspirasi dari eksperimen: cv2.adaptiveThreshold Gaussian 11,2 keluarkan teks di 222×444 dark, sementara Otsu global blank.

Berjalan di CPU (OpenCV + Pillow, tanpa GPU).
Steps v1 (utama):  grayscale → CLAHE (clip 2.0, tile 8x8) → autocontrast (normalize 0-255) → upscale 2x bicubic → sharpen → medianBlur 3
Steps v2 (fallback): v1 + adaptiveThreshold Gaussian 11,2 (dipakai jika pass1 blank)

Untuk server tanpa GPU, ini solusi ringan dan cukup efektif untuk label dark.
"""

from pathlib import Path
import logging

from PIL import Image, ImageFilter, ImageOps

from app.config import OCR_UPSCALE_FACTOR

logger = logging.getLogger(__name__)

# Try cv2 (optional but recommended for CLAHE/adaptive)
try:
    import cv2
    import numpy as np
    _CV2_AVAILABLE = True
except ImportError:
    cv2 = None  # type: ignore
    np = None  # type: ignore
    _CV2_AVAILABLE = False
    logger.warning("opencv not available — falling back to Pillow-only preprocessing (less robust for dark images)")


def _autocontrast_cv2(gray):
    """Stretch histogram 0-255 via cv2.normalize (MINMAX)."""
    if not _CV2_AVAILABLE:
        return gray
    return cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)


def preprocess_v1(image_path_or_gray) -> Image.Image:
    """
    Varian utama: grayscale → CLAHE → autocontrast → upscale 2x → sharpen → medianBlur.

    Args:
        image_path_or_gray: Path ke file atau numpy array grayscale (uint8) untuk reuse di v2.

    Returns:
        PIL Image grayscale yang sudah dipreprocess (untuk Tesseract).
        Jika cv2 tidak tersedia, fallback ke Pillow autocontrast+upscale+sharpen.
    """
    # --- Load / convert to gray ---
    if isinstance(image_path_or_gray, (str, Path)):
        path = str(image_path_or_gray)
        if _CV2_AVAILABLE:
            gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if gray is None:
                # Fallback Pillow jika cv2 gagal baca
                img_pil = Image.open(path)
                if img_pil.mode not in ("L", "RGB", "RGBA"):
                    try:
                        img_pil = img_pil.convert("RGB")
                    except Exception:
                        pass
                gray_pil = ImageOps.grayscale(img_pil)
                gray = np.array(gray_pil)
        else:
            # Pillow-only path
            img_pil = Image.open(path)
            if img_pil.mode not in ("L", "RGB", "RGBA"):
                try:
                    img_pil = img_pil.convert("RGB")
                except Exception:
                    pass
            gray_pil = ImageOps.grayscale(img_pil)
            # lanjut ke Pillow enhancement
            gray_pil = ImageOps.autocontrast(gray_pil, cutoff=1)
            from PIL import ImageEnhance
            gray_pil = ImageEnhance.Contrast(gray_pil).enhance(1.5)
            w, h = gray_pil.size
            nw, nh = w * OCR_UPSCALE_FACTOR, h * OCR_UPSCALE_FACTOR
            max_side = 6000
            if max(nw, nh) > max_side:
                scale = max_side / max(nw, nh)
                nw, nh = int(nw * scale), int(nh * scale)
            gray_up = gray_pil.resize((nw, nh), Image.LANCZOS)
            gray_up = gray_up.filter(ImageFilter.SHARPEN)
            return gray_up
    else:
        # Sudah numpy array (dari pass sebelumnya)
        gray = image_path_or_gray
        if gray is None:
            raise ValueError(f"Cannot read image: {image_path_or_gray}")

    # --- cv2 path: CLAHE → autocontrast → upscale → sharpen → denoise ---
    # 1. CLAHE — perkuat kontras lokal (untuk dark low-contrast)
    try:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
    except Exception as e:
        logger.warning(f"CLAHE failed, fallback to raw gray: {e}")
        enhanced = gray

    # 2. Autocontrast — stretch global 0-255
    try:
        enhanced = _autocontrast_cv2(enhanced)
    except Exception:
        pass

    # 3. Upscale 2x (bicubic untuk OCR, lebih baik dari LANCZOS cv2)
    try:
        upscaled = cv2.resize(enhanced, None, fx=OCR_UPSCALE_FACTOR, fy=OCR_UPSCALE_FACTOR, interpolation=cv2.INTER_CUBIC)
    except Exception as e:
        logger.warning(f"Resize failed: {e}")
        upscaled = enhanced

    # 4. Sharpen
    try:
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        sharpened = cv2.filter2D(upscaled, -1, kernel)
    except Exception:
        sharpened = upscaled

    # 5. Denoise ringan (median blur 3) — jangan 5 (blur teks)
    try:
        denoised = cv2.medianBlur(sharpened, 3)
    except Exception:
        denoised = sharpened

    # Convert ke PIL untuk Tesseract
    try:
        pil_img = Image.fromarray(denoised)
    except Exception:
        # Fallback: numpy → PIL via mode L
        pil_img = Image.fromarray(denoised.astype(np.uint8), mode="L")

    logger.debug(f"[OCR] Preprocess v1: CLAHE+autocontrast+upscale+sharpen+medianBlur (cv2) -> {pil_img.size}")
    return pil_img


def preprocess_v2_adaptive(image_path_or_gray) -> Image.Image:
    """
    Varian fallback: V1 + adaptiveThreshold Gaussian block 11, C 2.

    Untuk 2-pass OCR jika V1 hasilnya blank (0 lines).
    Block 11, C 2 terbukti keluarkan teks (berantakan tapi ada) di 222×444 dark
    sementara Otsu global blank. Jangan pakai Otsu global.

    Args:
        image_path_or_gray: Path atau gray array.

    Returns:
        PIL Image binary (0/255) hasil adaptive threshold.
    """
    if isinstance(image_path_or_gray, (str, Path)):
        # Build base dari v1 dulu (reuse logic)
        # Tapi lebih efisien: load gray dan v1, lalu adaptive
        path = str(image_path_or_gray)
        if _CV2_AVAILABLE:
            gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if gray is None:
                raise ValueError(f"Cannot read image: {path}")
            # CLAHE + autocontrast + upscale sama seperti v1
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            enhanced = _autocontrast_cv2(enhanced)
            upscaled = cv2.resize(enhanced, None, fx=OCR_UPSCALE_FACTOR, fy=OCR_UPSCALE_FACTOR, interpolation=cv2.INTER_CUBIC)
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
            sharpened = cv2.filter2D(upscaled, -1, kernel)
            base = cv2.medianBlur(sharpened, 3)
        else:
            # Pillow fallback: gunakan v1 Pillow lalu convert ke cv2-like binary via point
            pil_v1 = preprocess_v1(path)
            base = np.array(pil_v1)
    else:
        # Input sudah gray array — jalankan v1 pipeline
        base_pil = preprocess_v1(image_path_or_gray)
        base = np.array(base_pil)

    # adaptiveThreshold butuh 8-bit single channel
    try:
        adaptive = cv2.adaptiveThreshold(
            base, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )
    except Exception as e:
        logger.error(f"adaptiveThreshold failed: {e}, returning base")
        adaptive = base

    pil_adaptive = Image.fromarray(adaptive)
    logger.info(f"[OCR] Preprocess v2: adaptiveThreshold 11,2 fallback -> {pil_adaptive.size} (for blank retry)")
    return pil_adaptive


def preprocess_image(image_path: str | Path) -> Image.Image:
    """
    Preprocessing gambar untuk meningkatkan kualitas OCR (backward compat).

    Sekarang alias ke preprocess_v1 (CLAHE+autocontrast) — bukan lagi Pillow plain.
    Tetap handle Pillow RGBA → L, upscale, sharpen untuk kompatibilitas.

    Langkah berurutan (v1):
    1. Buka gambar (Pillow auto-detect format PNG/JPEG).
    2. Konversi ke grayscale
    3. CLAHE + autocontrast (via cv2, fallback Pillow autocontrast)
    4. Upscale 2x dengan bicubic/LANCZOS
    5. Sharpen + medianBlur 3

    Berjalan di CPU (tanpa GPU). Untuk gambar 5 MB, memakan ~0.2-1 detik.

    Args:
        image_path: Path absolut ke file gambar.

    Returns:
        Image.Image: Gambar yang sudah dipreprocess, siap untuk OCR.

    Raises:
        FileNotFoundError: Jika file tidak ada.
        OSError: Jika file bukan gambar valid / corrupt.
    """
    return preprocess_v1(image_path)


def preprocess_image_robust(image_path: str | Path) -> Image.Image:
    """
    Alias explicit untuk v1 robust — untuk code yang ingin jelas pakai robust.
    Sama dengan preprocess_image() sekarang.
    """
    return preprocess_v1(image_path)


def preprocess_with_adaptive_threshold(image_path: str | Path) -> Image.Image:
    """
    Alias untuk v2 adaptive — dipanggil dari 2-pass di main.py jika pass1 blank.
    """
    return preprocess_v2_adaptive(image_path)


def preprocess_image_with_optional_binarize(
    image_path: str | Path, binarize: bool = False
) -> Image.Image:
    """
    Varian dengan threshold opsional untuk gambar latar sangat ramai.

    Args:
        image_path: Path ke gambar.
        binarize: Jika True, lakukan adaptive threshold (sekarang pakai Gaussian 11,2, bukan 128 global).

    Returns:
        Image.Image yang sudah dipreprocess.
    """
    if binarize:
        return preprocess_v2_adaptive(image_path)
    return preprocess_v1(image_path)

