"""
app/vision/label_detector.py — Deteksi & Pelurusan Label Otomatis (Pilihan C).

Metode: Classical Computer Vision (OpenCV - Tanpa Training AI).
Fungsi:
1. Mendeteksi pita label persegi panjang (Label 1 & Label 2) pada foto utuh.
2. Melakukan Auto-Deskew (pelurusan kemiringan sudut) via warpPerspective.
3. Mengurutkan hasil deteksi dari kiri ke kanan (Slot 1 & Slot 2).
4. Menghasilkan citra terpotong yang bersih dan gambar preview dengan bounding box hijau.
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import time


def deskew_crop(image: np.ndarray, rect: Tuple[Tuple[float, float], Tuple[float, float], float]) -> np.ndarray:
    """
    Melakukan 4-point perspective transform untuk memotong dan meluruskan label
    berdasarkan minAreaRect ((center_x, center_y), (width, height), angle).
    """
    center, size, angle = rect
    width, height = int(size[0]), int(size[1])

    if width < 1 or height < 1:
        return image

    # Pastikan orientasi berdiri atau horizontal konsisten (aspek rasio pita pakaian)
    # Di garment, pita label biasanya tegak (vertikal) atau horizontal panjang
    src_pts = cv2.boxPoints(rect).astype(np.float32)

    # Urutkan titik sudut: top-left, top-right, bottom-right, bottom-left
    # Jumlah koordinat (x + y) terkecil adalah top-left, terbesar adalah bottom-right
    s = src_pts.sum(axis=1)
    diff = np.diff(src_pts, axis=1)

    tl = src_pts[np.argmin(s)]
    br = src_pts[np.argmax(s)]
    tr = src_pts[np.argmin(diff)]
    bl = src_pts[np.argmax(diff)]

    rect_pts = np.array([tl, tr, br, bl], dtype=np.float32)

    # Hitung lebar dan tinggi target
    w_a = np.linalg.norm(br - bl)
    w_b = np.linalg.norm(tr - tl)
    max_w = max(int(w_a), int(w_b))

    h_a = np.linalg.norm(tr - br)
    h_b = np.linalg.norm(tl - bl)
    max_h = max(int(h_a), int(h_b))

    if max_w < 10 or max_h < 10:
        return image

    dst_pts = np.array([
        [0, 0],
        [max_w - 1, 0],
        [max_w - 1, max_h - 1],
        [0, max_h - 1]
    ], dtype=np.float32)

    m = cv2.getPerspectiveTransform(rect_pts, dst_pts)
    warped = cv2.warpPerspective(image, m, (max_w, max_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return warped


def detect_labels(
    image: np.ndarray,
    min_area_ratio: float = 0.005,
    max_area_ratio: float = 0.95,
    min_aspect_ratio: float = 1.05,
    max_aspect_ratio: float = 10.0,
    padding_pct: float = 0.02
) -> Dict[str, Any]:
    """
    Deteksi 1 atau 2 label pada foto pakaian utuh menggunakan segmentasi kontur OpenCV.
    
    Args:
        image: Citra BGR (np.ndarray)
        min_area_ratio: Luas area minimum terhadap total resolusi gambar (default 1.5%)
        max_area_ratio: Luas area maksimum terhadap total resolusi (default 45%)
        min_aspect_ratio: Rasio aspek minimal panjang vs lebar pita
        max_aspect_ratio: Rasio aspek maksimal panjang vs lebar pita
        padding_pct: Tambahan padding di sekeliling label agar tepi huruf tidak terpotong
        
    Returns:
        Dict hasil deteksi, koordinat, gambar potongan, dan annotated preview.
    """
    t0 = time.time()
    h_img, w_img = image.shape[:2]
    total_area = float(h_img * w_img)
    min_area = total_area * min_area_ratio
    max_area = total_area * max_area_ratio

    # 1. Grayscale & Blur
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # 2. Thresholding Adaptif + Otsu untuk mengisolasi pita putih/terang
    # Kombinasi Otsu dan Morphological Closing
    _, thresh1 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Filter HSV untuk menangkap warna putih/krem satin jika background memiliki pencahayaan kuat
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 140])
    upper_white = np.array([180, 70, 255])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    # Gabungkan binary mask
    combined_mask = cv2.bitwise_or(thresh1, mask_white)

    # 3. Morphological Close: sambungkan huruf dan teks menjadi satu lempengan pita solid
    # Kernel horizontal dan vertikal
    k_w = max(7, int(w_img * 0.025))
    k_h = max(7, int(h_img * 0.025))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_w, k_h))
    closed = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 4. Cari kontur terluar
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        rect = cv2.minAreaRect(cnt)
        (cx, cy), (rw, rh), angle = rect
        if rw < 10 or rh < 10:
            continue

        # Hitung aspek rasio pita
        aspect = max(rw, rh) / min(rw, rh)
        if aspect < min_aspect_ratio or aspect > max_aspect_ratio:
            continue

        candidates.append({
            "contour": cnt,
            "rect": rect,
            "area": area,
            "cx": cx,
            "cy": cy,
            "aspect": aspect,
        })

    # Urutkan kandidat berdasarkan area terbesar (ambil maks 2 label)
    candidates.sort(key=lambda c: c["area"], reverse=True)
    selected = candidates[:2]

    # Urutkan dari kiri ke kanan berdasarkan sumbu X (kiri = Slot 1, kanan = Slot 2)
    selected.sort(key=lambda c: c["cx"])

    # Buat annotated image untuk konfirmasi visual operator
    annotated = image.copy()
    slots_output = []

    for idx, item in enumerate(selected):
        slot_num = idx + 1
        label_name = f"Label {slot_num}"
        rect = item["rect"]
        (cx, cy), (rw, rh), angle = rect

        # Box points
        box_pts = cv2.boxPoints(rect).astype(np.int32)

        # Gambar poligon hijau terang (#00FF00) dan teks label
        cv2.drawContours(annotated, [box_pts], 0, (0, 255, 64), 3)

        # Tambahkan teks pill
        text_pos = (int(cx - 40), max(30, int(cy - max(rw, rh) / 2 - 10)))
        cv2.putText(annotated, label_name, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 64), 2, cv2.LINE_AA)

        # Potong & luruskan (Auto-Deskew)
        cropped = deskew_crop(image, rect)

        slots_output.append({
            "slot": slot_num,
            "label_name": label_name,
            "rect": {
                "center": [float(cx), float(cy)],
                "size": [float(rw), float(rh)],
                "angle": float(angle)
            },
            "box_points": box_pts.tolist(),
            "cropped_image": cropped,
            "area": float(item["area"])
        })

    elapsed_ms = (time.time() - t0) * 1000.0

    return {
        "success": len(slots_output) > 0,
        "detected_count": len(slots_output),
        "slots": slots_output,
        "annotated_image": annotated,
        "elapsed_ms": round(elapsed_ms, 2)
    }
