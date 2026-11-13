"""
scripts/test_auto_detect_c.py — Pengujian mandiri Pilihan C (OpenCV Auto-Detect & Deskew).

Cara jalan:
  python scripts/test_auto_detect_c.py [path_to_image.png]
"""

import sys
import os
from pathlib import Path
import cv2

# Tambahkan root path backend
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from app.vision.label_detector import detect_labels

def main():
    if len(sys.argv) > 1:
        img_path = Path(sys.argv[1])
    else:
        # Default cari sampel gambar di uploads/check/
        samples = list(ROOT_DIR.glob("uploads/check/**/*.png"))
        if not samples:
            print("[INFO] Tidak ada sampel gambar di uploads/check/. Berikan path gambar sebagai argumen.")
            return
        img_path = samples[0]

    if not img_path.exists():
        print(f"[ERROR] File tidak ditemukan: {img_path}")
        return

    print(f"=== UJI COBA DETEKSI LABEL PILIHAN C (OPENCV) ===")
    print(f"File Uji: {img_path}")
    image = cv2.imread(str(img_path))
    if image is None:
        print("[ERROR] Gagal membaca gambar.")
        return

    h, w = image.shape[:2]
    print(f"Resolusi Asli: {w} x {h}")

    res = detect_labels(image)

    print(f"\n--- Hasil Eksekusi ---")
    print(f"Waktu Deteksi   : {res['elapsed_ms']} ms")
    print(f"Label Terdeteksi: {res['detected_count']} buah")

    out_dir = ROOT_DIR / "uploads" / "test_auto_detect_c"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Simpan annotated preview
    annotated_path = out_dir / f"annotated_{img_path.name}"
    cv2.imwrite(str(annotated_path), res["annotated_image"])
    print(f"Preview Kotak Hijau: {annotated_path}")

    for s in res["slots"]:
        slot_file = out_dir / f"crop_slot_{s['slot']}_{img_path.name}"
        cv2.imwrite(str(slot_file), s["cropped_image"])
        sh, sw = s["cropped_image"].shape[:2]
        print(f"  -> {s['label_name']}: Sudut {s['rect']['angle']:.1f} deg | Dimensi Crop: {sw}x{sh} | Saved: {slot_file.name}")

    print("\n[SUCCESS] Pengujian deteksi selesai!")

if __name__ == "__main__":
    main()

