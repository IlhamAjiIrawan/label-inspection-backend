"""
app/vision/label_detector.py — Deteksi & Pelurusan Label Otomatis Presisi Tinggi (Pilihan C).

Metode: Classical Computer Vision (OpenCV - Sobel Energy Profiling + Auto-Deskew).
Keunggulan:
1. Kebal terhadap background meja putih/terang (menggunakan densitas gradien teks, bukan warna).
2. Tight Boundary Refinement: memotong tepat di batas tepi pita kain sebenarnya, membuang
   area meja kosong di atas, bawah, kiri, dan kanan.
3. Pembagian 2-in-1 Ribbon Tepat di Tengah (Midpoint Cut): membagi pita garmen continuous
   tepat pada garis tengah (divider line) antara Label 1 dan Label 2, baik orientasi vertikal maupun horizontal.
4. Auto-Deskew 4-point perspective transform dengan inverse mapping untuk bounding box presisi.
5. Smart fallback adaptif sesuai rasio aspek kamera bila citra blur ekstrem.
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import time
import base64


def image_to_base64_data_url(image: np.ndarray, ext: str = "png") -> str:
    """Konversi citra cv2 numpy array menjadi base64 data URL string."""
    if image is None or image.size == 0:
        return ""
    success, buffer = cv2.imencode(f".{ext}", image)
    if not success:
        return ""
    b64_str = base64.b64encode(buffer).decode("utf-8")
    mime = "image/jpeg" if ext.lower() in ("jpg", "jpeg") else "image/png"
    return f"data:{mime};base64,{b64_str}"


def get_perspective_matrices(
    image: np.ndarray,
    rect: Tuple[Tuple[float, float], Tuple[float, float], float]
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int, int]:
    """
    Menghitung matriks transformasi perspektif (M) dan matriks invers (M_inv)
    serta dimensi target (max_w, max_h) berdasarkan minAreaRect.
    """
    center, size, angle = rect
    width, height = int(size[0]), int(size[1])

    if width < 1 or height < 1:
        return None, None, 0, 0

    src_pts = cv2.boxPoints(rect).astype(np.float32)

    # Urutkan titik sudut: top-left, top-right, bottom-right, bottom-left
    s = src_pts.sum(axis=1)
    diff = np.diff(src_pts, axis=1)

    tl = src_pts[np.argmin(s)]
    br = src_pts[np.argmax(s)]
    tr = src_pts[np.argmin(diff)]
    bl = src_pts[np.argmax(diff)]

    rect_pts = np.array([tl, tr, br, bl], dtype=np.float32)

    w_a = np.linalg.norm(br - bl)
    w_b = np.linalg.norm(tr - tl)
    max_w = max(int(w_a), int(w_b))

    h_a = np.linalg.norm(tr - br)
    h_b = np.linalg.norm(tl - bl)
    max_h = max(int(h_a), int(h_b))

    if max_w < 10 or max_h < 10:
        return None, None, 0, 0

    dst_pts = np.array([
        [0, 0],
        [max_w - 1, 0],
        [max_w - 1, max_h - 1],
        [0, max_h - 1]
    ], dtype=np.float32)

    m = cv2.getPerspectiveTransform(rect_pts, dst_pts)
    m_inv = cv2.getPerspectiveTransform(dst_pts, rect_pts)
    return m, m_inv, max_w, max_h


def deskew_crop(image: np.ndarray, rect: Tuple[Tuple[float, float], Tuple[float, float], float]) -> np.ndarray:
    """Melakukan 4-point perspective transform untuk memotong & meluruskan label."""
    m, _, max_w, max_h = get_perspective_matrices(image, rect)
    if m is None:
        return image
    return cv2.warpPerspective(image, m, (max_w, max_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def refine_tight_bounds(
    image: np.ndarray,
    is_vertical: bool,
    margin_ratio: float = 0.02
) -> Tuple[int, int, int, int, int]:
    """
    Menemukan batas pita label yang ketat (menghilangkan meja kosong di pinggir)
    dan menghitung titik tengah (midpoint) yang presisi berdasarkan akumulasi gradien Sobel.

    Returns:
        (x_min, x_max, y_min, y_max, midpoint)
    """
    h, w = image.shape[:2]
    if h < 20 or w < 20:
        return 0, w, 0, h, (h // 2 if is_vertical else w // 2)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(sobelx, sobely)

    # Redam 3% margin terluar agar artefak tepi warp tidak mengganggu
    bx = max(2, int(w * 0.03))
    by = max(2, int(h * 0.03))
    mag[:by, :] = 0; mag[-by:, :] = 0; mag[:, :bx] = 0; mag[:, -bx:] = 0

    pad_x = max(2, int(w * margin_ratio))
    pad_y = max(2, int(h * margin_ratio))

    if is_vertical:
        # 1. Sumbu pendek X (lebar pita kain)
        col_energy = mag.sum(axis=0)
        table_noise_x = np.percentile(col_energy[bx:-bx], 15) if w > bx * 2 else col_energy.min()
        th_x = table_noise_x + (col_energy.max() - table_noise_x) * 0.20
        act_x = np.where(col_energy > th_x)[0]
        if len(act_x) > 0:
            x_min = max(0, act_x[0] - pad_x)
            x_max = min(w, act_x[-1] + pad_x)
        else:
            x_min, x_max = 0, w

        # 2. Sumbu panjang Y (tinggi pita kain) di dalam rentang kolom x_min..x_max
        sub_mag = mag[:, x_min:x_max]
        row_energy = sub_mag.sum(axis=1)
        table_noise_y = np.percentile(row_energy[by:-by], 15) if h > by * 2 else row_energy.min()
        th_y = table_noise_y + (row_energy.max() - table_noise_y) * 0.18
        act_y = np.where(row_energy > th_y)[0]
        if len(act_y) > 0:
            y_min = max(0, act_y[0] - pad_y)
            y_max = min(h, act_y[-1] + pad_y)
        else:
            y_min, y_max = 0, h

        # Midpoint tepat di tengah rentang Y pita yang sebenarnya
        y_mid = (y_min + y_max) // 2
        return x_min, x_max, y_min, y_max, y_mid

    else:
        # Pita Horizontal
        # 1. Sumbu pendek Y (tinggi pita kain)
        row_energy = mag.sum(axis=1)
        table_noise_y = np.percentile(row_energy[by:-by], 15) if h > by * 2 else row_energy.min()
        th_y = table_noise_y + (row_energy.max() - table_noise_y) * 0.20
        act_y = np.where(row_energy > th_y)[0]
        if len(act_y) > 0:
            y_min = max(0, act_y[0] - pad_y)
            y_max = min(h, act_y[-1] + pad_y)
        else:
            y_min, y_max = 0, h

        # 2. Sumbu panjang X (lebar pita kain) di dalam rentang baris y_min..y_max
        sub_mag = mag[y_min:y_max, :]
        col_energy = sub_mag.sum(axis=0)
        table_noise_x = np.percentile(col_energy[bx:-bx], 15) if w > bx * 2 else col_energy.min()
        th_x = table_noise_x + (col_energy.max() - table_noise_x) * 0.18
        act_x = np.where(col_energy > th_x)[0]
        if len(act_x) > 0:
            x_min = max(0, act_x[0] - pad_x)
            x_max = min(w, act_x[-1] + pad_x)
        else:
            x_min, x_max = 0, w

        x_mid = (x_min + x_max) // 2
        return x_min, x_max, y_min, y_max, x_mid


def detect_labels(
    image: np.ndarray,
    min_area_ratio: float = 0.015,
    max_area_ratio: float = 0.92,
    fallback_on_miss: bool = True
) -> Dict[str, Any]:
    """
    Deteksi Label 1 & Label 2 pada foto pakaian dengan pemotongan ketat
    dan pembagian wilayah yang presisi tepat di tengah lempengan label.

    Args:
        image: Citra BGR (np.ndarray)
        min_area_ratio: Luas area minimum terhadap resolusi frame (default 1.5%)
        max_area_ratio: Luas area maksimum terhadap resolusi frame (default 92%)
        fallback_on_miss: Jika deteksi kontur gagal, gunakan fallback tengah adaptif

    Returns:
        Dict hasil deteksi berisi slots, annotated preview, dan metadata.
    """
    t0 = time.time()
    if image is None or image.size == 0:
        return {
            "success": False,
            "detected_count": 0,
            "slots": [],
            "annotated_image": np.zeros((100, 100, 3), dtype=np.uint8),
            "elapsed_ms": 0.0
        }

    h_img, w_img = image.shape[:2]
    total_area = float(h_img * w_img)

    # 1. Grayscale & Canny Edge Detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 100)

    # Nol-kan 4% margin terluar kamera frame untuk mencegah border viewport menyatu
    bx = max(8, int(w_img * 0.04))
    by = max(8, int(h_img * 0.04))
    edges[:by, :] = 0
    edges[-by:, :] = 0
    edges[:, :bx] = 0
    edges[:, -bx:] = 0

    # 2. Morphological Close terukur untuk menyatukan baris teks
    kw = max(11, int(w_img * 0.018))
    kh = max(11, int(h_img * 0.018))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 3. Cari kontur eksternal
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < (total_area * min_area_ratio) or area > (total_area * max_area_ratio):
            continue
        rect = cv2.minAreaRect(cnt)
        (cx, cy), (rw, rh), angle = rect
        if rw < 25 or rh < 25:
            continue
        if rw >= w_img * 0.90 and rh >= h_img * 0.90:
            continue
        aspect = max(rw, rh) / max(1.0, min(rw, rh))
        candidates.append({
            "contour": cnt,
            "rect": rect,
            "area": area,
            "cx": cx,
            "cy": cy,
            "rw": rw,
            "rh": rh,
            "aspect": aspect,
            "is_fallback": False
        })

    candidates.sort(key=lambda c: c["area"], reverse=True)

    slots_output = []
    annotated = image.copy()

    # Evaluasi pita tunggal 2-in-1 ribbon
    is_single_strip = False
    if len(candidates) == 1 and candidates[0]["aspect"] >= 1.30:
        is_single_strip = True
    elif len(candidates) >= 2 and candidates[0]["aspect"] >= 1.35 and (candidates[0]["area"] > candidates[1]["area"] * 2.0):
        is_single_strip = True
    elif len(candidates) == 1:
        # Bahkan bila aspect < 1.30, jika hanya 1 kontur di tengah meja garmen,
        # hampir pasti merupakan lempengan pita care label
        is_single_strip = True

    # =========================================================================
    # KASUS A: 1 Pita Panjang Continuous (Care Label 2-in-1)
    # =========================================================================
    if is_single_strip:
        cand = candidates[0]
        rect = cand["rect"]
        m, m_inv, max_w, max_h = get_perspective_matrices(image, rect)

        if m is not None:
            warped = cv2.warpPerspective(image, m, (max_w, max_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            sh, sw = warped.shape[:2]
            is_vertical = (sh > sw)

            # Refine tight bounds untuk memotong tepat di lempengan pita sebenarnya
            x0, x1, y0, y1, mid = refine_tight_bounds(warped, is_vertical=is_vertical)

            if is_vertical:
                # Pita Vertikal: Belah tepat di tengah (y_mid)
                crop1 = warped[y0:mid, x0:x1]
                crop2 = warped[mid:y1, x0:x1]

                # Titik sudut di ruang warped
                box1_w = np.array([[[x0, y0], [x1, y0], [x1, mid], [x0, mid]]], dtype=np.float32)
                box2_w = np.array([[[x0, mid], [x1, mid], [x1, y1], [x0, y1]]], dtype=np.float32)
            else:
                # Pita Horizontal: Belah tepat di tengah (x_mid)
                crop1 = warped[y0:y1, x0:mid]
                crop2 = warped[y0:y1, mid:x1]

                box1_w = np.array([[[x0, y0], [mid, y0], [mid, y1], [x0, y1]]], dtype=np.float32)
                box2_w = np.array([[[mid, y0], [x1, y0], [x1, y1], [mid, y1]]], dtype=np.float32)

            # Petakan kembali koordinat bounding box ke ruang citra asli via m_inv
            box1_orig = cv2.perspectiveTransform(box1_w, m_inv)[0].astype(np.int32)
            box2_orig = cv2.perspectiveTransform(box2_w, m_inv)[0].astype(np.int32)
            c1 = box1_orig.mean(axis=0)
            c2 = box2_orig.mean(axis=0)

            slots_output = [
                {
                    "slot": 1,
                    "label_name": "Label 1",
                    "is_fallback": False,
                    "rect": {
                        "center": [float(c1[0]), float(c1[1])],
                        "size": [float(crop1.shape[1]), float(crop1.shape[0])],
                        "angle": float(rect[2])
                    },
                    "box_points": box1_orig.tolist(),
                    "cropped_image": crop1,
                    "area": float(cand["area"] * 0.5)
                },
                {
                    "slot": 2,
                    "label_name": "Label 2",
                    "is_fallback": False,
                    "rect": {
                        "center": [float(c2[0]), float(c2[1])],
                        "size": [float(crop2.shape[1]), float(crop2.shape[0])],
                        "angle": float(rect[2])
                    },
                    "box_points": box2_orig.tolist(),
                    "cropped_image": crop2,
                    "area": float(cand["area"] * 0.5)
                }
            ]

    # =========================================================================
    # KASUS B: 2 Label Terpisah (Terdeteksi 2 Kontur Jelas)
    # =========================================================================
    elif len(candidates) >= 2:
        c0, c1 = candidates[0], candidates[1]
        # Urutkan: horizontal (kiri->kanan) atau vertikal (atas->bawah)
        if abs(c0["cx"] - c1["cx"]) >= abs(c0["cy"] - c1["cy"]):
            selected = sorted([c0, c1], key=lambda c: c["cx"])
        else:
            selected = sorted([c0, c1], key=lambda c: c["cy"])

        for idx, item in enumerate(selected):
            s_num = idx + 1
            rect = item["rect"]
            m, m_inv, max_w, max_h = get_perspective_matrices(image, rect)
            if m is not None:
                warped = cv2.warpPerspective(image, m, (max_w, max_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                sh, sw = warped.shape[:2]
                x0, x1, y0, y1, _ = refine_tight_bounds(warped, is_vertical=(sh > sw))
                cropped = warped[y0:y1, x0:x1]

                box_w = np.array([[[x0, y0], [x1, y0], [x1, y1], [x0, y1]]], dtype=np.float32)
                box_orig = cv2.perspectiveTransform(box_w, m_inv)[0].astype(np.int32)
                c_orig = box_orig.mean(axis=0)

                slots_output.append({
                    "slot": s_num,
                    "label_name": f"Label {s_num}",
                    "is_fallback": False,
                    "rect": {
                        "center": [float(c_orig[0]), float(c_orig[1])],
                        "size": [float(cropped.shape[1]), float(cropped.shape[0])],
                        "angle": float(rect[2])
                    },
                    "box_points": box_orig.tolist(),
                    "cropped_image": cropped,
                    "area": float(item["area"])
                })

    # =========================================================================
    # KASUS C: Fallback Penuh (Kontur Tidak Ditemukan / Blur Ekstrem)
    # =========================================================================
    if len(slots_output) < 2 and fallback_on_miss:
        slots_output = []
        is_vertical_img = (h_img >= w_img)
        if is_vertical_img:
            # Frame tegak / potret: potong area tengah atas dan tengah bawah
            r1 = ((w_img * 0.5, h_img * 0.28), (w_img * 0.40, h_img * 0.40), 0.0)
            r2 = ((w_img * 0.5, h_img * 0.72), (w_img * 0.40, h_img * 0.40), 0.0)
        else:
            # Frame mendatar / lanskap: potong area tengah kiri dan tengah kanan
            r1 = ((w_img * 0.30, h_img * 0.5), (w_img * 0.38, h_img * 0.65), 0.0)
            r2 = ((w_img * 0.70, h_img * 0.5), (w_img * 0.38, h_img * 0.65), 0.0)

        for idx, rect in enumerate([r1, r2]):
            s_num = idx + 1
            cropped = deskew_crop(image, rect)
            box_pts = cv2.boxPoints(rect).astype(np.int32)
            (cx, cy), (rw, rh), angle = rect
            slots_output.append({
                "slot": s_num,
                "label_name": f"Label {s_num}",
                "is_fallback": True,
                "rect": {
                    "center": [float(cx), float(cy)],
                    "size": [float(rw), float(rh)],
                    "angle": float(angle)
                },
                "box_points": box_pts.tolist(),
                "cropped_image": cropped,
                "area": float(rw * rh)
            })

    # =========================================================================
    # Visual Annotation (Bounding Box & Label Pill yang terlihat jelas di 4K)
    # =========================================================================
    line_thick = max(2, int(w_img * 0.0035))
    font_scale = max(0.65, w_img / 1400.0)
    font_thick = max(2, int(font_scale * 2))

    for s in slots_output:
        slot_num = s["slot"]
        is_fb = s["is_fallback"]
        color = (0, 165, 255) if is_fb else (0, 255, 64)  # Orange jika fallback, Hijau jika OpenCV deteksi
        box_pts = np.array(s["box_points"], dtype=np.int32)
        cv2.drawContours(annotated, [box_pts], 0, color, line_thick)

        cx, cy = s["rect"]["center"]
        text_pos = (max(10, int(cx - 50 * font_scale)), max(35, int(cy - 20)))
        tag = f"Label {slot_num}" + (" (Auto)" if is_fb else "")
        cv2.putText(annotated, tag, text_pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thick, cv2.LINE_AA)

    elapsed_ms = (time.time() - t0) * 1000.0

    return {
        "success": len(slots_output) >= 2,
        "detected_count": len(slots_output),
        "slots": slots_output,
        "annotated_image": annotated,
        "elapsed_ms": round(elapsed_ms, 2)
    }
