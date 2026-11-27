"""
app/vision/blank_detector.py — Deteksi Label Kosong (Blank Side) & Filter Bleed-Through.

Menentukan apakah sebuah potongan (crop) label benar-benar polos/kosong
meskipun terdapat:
1. Bayangan samar tembus pandang (bleed-through) dari sisi balik kain.
2. Garis potong/gunting (---✂---) yang ikut masuk di pinggir area lipatan atas/bawah crop.
3. Margin meja abu-abu di sisi kiri/kanan hasil expand_rect_short_axis(1.18).

Metode cepat CPU (<5 ms): pangkas margin meja di sumbu pendek (14%) dan margin luar
sumbu panjang (5%), estimasi kecerahan dasar kain (persentil ke-85), ukur piksel
tinta tajam (selisih intensitas + Sobel), lalu evaluasi profil 5 pita sepanjang label.
"""

import logging
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _blank_config() -> tuple[bool, float, float, float, float]:
    """Ambil parameter detektor blank dari app.config dengan fallback aman."""
    try:
        from app import config as _cfg

        enabled = bool(getattr(_cfg, "BLANK_LABEL_DETECT_ENABLED", True))
        ink_delta = float(getattr(_cfg, "BLANK_INK_DELTA", 48.0))
        min_ink_ratio = float(getattr(_cfg, "BLANK_MIN_INK_RATIO", 0.0012))
        sobel_min = float(getattr(_cfg, "BLANK_SOBEL_MIN", 45.0))
        border_trim = float(getattr(_cfg, "BLANK_BORDER_TRIM_RATIO", 0.06))
        return enabled, ink_delta, min_ink_ratio, sobel_min, border_trim
    except Exception:
        return True, 48.0, 0.0012, 45.0, 0.06


def detect_blank_label(image: np.ndarray) -> Dict[str, Any]:
    """
    Periksa apakah citra BGR/grayscale adalah label kosong (termasuk bleed-through samar
    dan garis gunting tipis di batas lipatan).

    TIDAK PERNAH raise. Jika citra terlalu gelap (<85 luma) atau gagal proses,
    kembalikan is_blank=False agar diproses normal oleh pipeline OCR.
    """
    enabled, ink_delta, min_ink_ratio, sobel_min, border_trim = _blank_config()
    if not enabled:
        return {"enabled": False, "is_blank": False, "reason": "disabled"}

    try:
        if image is None or getattr(image, "size", 0) == 0:
            return {"enabled": True, "is_blank": False, "reason": "empty_image"}

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        h, w = gray.shape[:2]
        if h < 24 or w < 24:
            return {"enabled": True, "is_blank": False, "reason": "too_small"}

        # Sumbu pendek punya padding ~18% (expand_rect_short_axis 1.18) -> pangkas 14% per sisi
        # agar tepi meja tidak dihitung; sumbu panjang pangkas minimal (5%/border_trim).
        is_vertical = h >= w
        long_trim = min(0.06, max(0.04, border_trim))
        short_trim = max(0.14, border_trim)
        trim_y = long_trim if is_vertical else short_trim
        trim_x = short_trim if is_vertical else long_trim

        by = max(2, int(h * trim_y))
        bx = max(2, int(w * trim_x))
        if (h - 2 * by) >= 16 and (w - 2 * bx) >= 16:
            roi = gray[by : h - by, bx : w - bx]
        else:
            roi = gray

        bg_luma = float(np.percentile(roi, 85))
        # Guard foto terlalu gelap/kurang cahaya -> biarkan OCR menangani + beri peringatan cahaya
        if bg_luma < 85.0:
            return {
                "enabled": True,
                "is_blank": False,
                "reason": "dark_background",
                "bg_luma": round(bg_luma, 2),
            }

        roi_f = roi.astype(np.float32)
        diff = np.clip(bg_luma - roi_f, 0.0, 255.0)

        sx = cv2.Sobel(roi_f, cv2.CV_32F, 1, 0, ksize=3)
        sy = cv2.Sobel(roi_f, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(sx, sy)

        # Tinta cetak asli memiliki ketajaman tepi (Sobel) + penurunan kecerahan signifikan,
        # sedangkan bleed-through kain terdifusi lembut dan dangkal.
        real_ink = ((diff >= ink_delta) & (mag >= sobel_min)) | (diff >= (ink_delta + 22.0))
        ink_ratio = float(np.count_nonzero(real_ink)) / float(roi.size)
        peak_diff = float(np.percentile(diff, 99.7))

        # Profil 5 pita sepanjang sumbu panjang label: jika bagian isi tengah (pita 1..3) bersih
        # (< 0.4%) dan seluruh tinta hanya berada di salah satu pita ujung (garis gunting ---✂---),
        # tetap diklasifikasikan sebagai blank_label.
        rh, rw = real_ink.shape[:2]
        axis_len = rh if is_vertical else rw
        bands = []
        for i in range(5):
            a = int(axis_len * i / 5)
            b = int(axis_len * (i + 1) / 5)
            seg = real_ink[a:b, :] if is_vertical else real_ink[:, a:b]
            bands.append(float(seg.mean()) if seg.size else 0.0)

        interior_max = max(bands[1], bands[2], bands[3]) if len(bands) >= 4 else ink_ratio
        active_bands = sum(1 for v in bands if v >= max(0.015, min_ink_ratio))

        is_globally_blank = (ink_ratio < min_ink_ratio) and (peak_diff < (ink_delta + 18.0))
        is_fold_border_only = (
            interior_max < 0.015
            and active_bands <= 1
            and min(bands[0], bands[4]) < 0.005
            and ink_ratio < 0.025
        )
        is_blank = bool(is_globally_blank or is_fold_border_only)

        return {
            "enabled": True,
            "is_blank": is_blank,
            "reason": "blank_label" if is_blank else "text_or_ink_detected",
            "bg_luma": round(bg_luma, 2),
            "ink_ratio": round(ink_ratio, 6),
            "interior_max": round(interior_max, 6),
            "peak_diff": round(peak_diff, 2),
        }
    except Exception as e:
        logger.debug(f"[BLANK] detect_blank_label error ignored: {e}")
        return {"enabled": True, "is_blank": False, "reason": "error"}


def detect_blank_label_bytes(image_bytes: bytes) -> Dict[str, Any]:
    """Decode bytes gambar lalu jalankan detect_blank_label() secara aman."""
    enabled, _, _, _, _ = _blank_config()
    if not enabled:
        return {"enabled": False, "is_blank": False, "reason": "disabled"}
    try:
        if not image_bytes:
            return {"enabled": True, "is_blank": False, "reason": "empty_bytes"}
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return {"enabled": True, "is_blank": False, "reason": "decode_failed"}
        return detect_blank_label(img)
    except Exception as e:
        logger.debug(f"[BLANK] detect_blank_label_bytes error ignored: {e}")
        return {"enabled": True, "is_blank": False, "reason": "error"}


def detect_blank_label_path(image_path: str | Path) -> Dict[str, Any]:
    """Baca file gambar dari path lalu jalankan detect_blank_label() secara aman."""
    enabled, _, _, _, _ = _blank_config()
    if not enabled:
        return {"enabled": False, "is_blank": False, "reason": "disabled"}
    try:
        with open(image_path, "rb") as f:
            return detect_blank_label_bytes(f.read())
    except Exception as e:
        logger.debug(f"[BLANK] detect_blank_label_path error ignored: {e}")
        return {"enabled": True, "is_blank": False, "reason": "error"}
