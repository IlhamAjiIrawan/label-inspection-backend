"""
test_surya.py — Pengujian mandiri untuk model Surya OCR (Multilingual Document & Label OCR).

Jalankan di terminal (.venv-surya):
  python test_surya.py
  python test_surya.py --image uploads/label-xxx.png
"""

import sys
import time
import argparse
from PIL import Image, ImageDraw, ImageFont

def create_sample_label_image() -> Image.Image:
    """Buat gambar simulasi label pakaian multi-baris (400x180 px)."""
    img = Image.new("RGB", (420, 180), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    try:
        font_bold = ImageFont.truetype("arialbd.ttf", 20)
        font_regular = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        try:
            font_bold = ImageFont.truetype("arial.ttf", 18)
            font_regular = ImageFont.truetype("arial.ttf", 15)
        except Exception:
            font_bold = ImageFont.load_default()
            font_regular = ImageFont.load_default()

    # Teks multi-baris khas label pakaian
    draw.text((25, 25), "100% COTTON / KATUN", fill=(0, 0, 0), font=font_bold)
    draw.text((25, 65), "MADE IN INDONESIA", fill=(0, 0, 0), font=font_regular)
    draw.text((25, 100), "RN# 12345 / CA# 00642", fill=(0, 0, 0), font=font_regular)
    draw.text((25, 135), "MACHINE WASH COLD - DO NOT BLEACH", fill=(0, 0, 0), font=font_regular)

    return img

def main():
    parser = argparse.ArgumentParser(description="Test Surya OCR Model")
    parser.add_argument("--image", type=str, default=None, help="Path gambar label (opsional)")
    parser.add_argument("--langs", type=str, default="en", help="Kode bahasa dipisah koma (default: en)")
    args = parser.parse_args()

    print("\n" + "=" * 55)
    print("      MEMERIKSA KESIAPAN SURYA OCR")
    print("=" * 55)

    # 1. Validasi Library Surya OCR
    try:
        from surya.ocr import run_ocr
        try:
            from surya.model.detection.model import load_model as load_det_model, load_processor as load_det_processor
        except ImportError:
            from surya.model.detection.segformer import load_model as load_det_model, load_processor as load_det_processor
        from surya.model.recognition.model import load_model as load_rec_model
        from surya.model.recognition.processor import load_processor as load_rec_processor
    except ImportError as e:
        print(f"\n[ERROR] Library surya-ocr belum terpasang di venv ini!")
        print(f"Detail error: {e}")
        print("\nSilakan install terlebih dahulu dengan:")
        print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu")
        print("  pip install surya-ocr\n")
        sys.exit(1)

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Perangkat komputasi: {device.upper()}")

    # 2. Muat Model Deteksi & Rekognisi Surya
    print("[*] Memuat model deteksi dan rekognisi Surya OCR...")
    print("    (Catatan: saat pertama kali dijalankan, bobot model akan diunduh otomatis dari Hugging Face)")
    t0 = time.time()
    try:
        det_processor = load_det_processor()
        det_model = load_det_model()
        rec_model = load_rec_model()
        rec_processor = load_rec_processor()
    except Exception as e:
        import traceback
        print(f"\n[ERROR] Gagal memuat bobot model Surya: {e}")
        traceback.print_exc()
        sys.exit(1)

    load_time = time.time() - t0
    print(f"[OK] Seluruh model Surya berhasil dimuat dalam {load_time:.2f} detik.")

    # 3. Siapkan Gambar
    import os
    if args.image and os.path.isfile(args.image):
        print(f"[*] Membuka gambar uji: {args.image}")
        img = Image.open(args.image).convert("RGB")
    else:
        if args.image:
            print(f"[!] Catatan: File '{args.image}' tidak ditemukan di komputer.")
            print("[*] Beralih otomatis membuat gambar label pakaian sintetis multi-baris...")
        else:
            print("[*] Membuat gambar label pakaian sintetis multi-baris...")
        img = create_sample_label_image()

    langs_list = [l.strip() for l in args.langs.split(",") if l.strip()]

    # 4. Jalankan OCR
    print(f"[*] Menjalankan Surya OCR (bahasa: {langs_list})...")
    t1 = time.time()
    try:
        predictions = run_ocr(
            [img],
            [langs_list],
            det_model,
            det_processor,
            rec_model,
            rec_processor
        )
    except Exception as e:
        print(f"\n[ERROR] Terjadi kesalahan saat inferensi: {e}")
        sys.exit(1)

    infer_time = time.time() - t1

    # 5. Tampilkan Hasil
    print("\n" + "-" * 55)
    print("  HASIL PEMBACAAN SURYA OCR:")
    print("-" * 55)

    total_lines = 0
    if predictions and len(predictions) > 0:
        page = predictions[0]
        text_lines = getattr(page, "text_lines", None)
        if text_lines is None and isinstance(page, dict):
            text_lines = page.get("text_lines", [])

        if text_lines:
            for idx, line in enumerate(text_lines, 1):
                text = getattr(line, "text", None) if not isinstance(line, dict) else line.get("text")
                confidence = getattr(line, "confidence", None) if not isinstance(line, dict) else line.get("confidence")
                conf_str = f"({confidence * 100:.1f}%)" if confidence is not None else ""
                print(f"  Baris {idx}: {text} {conf_str}")
                total_lines += 1
        else:
            print("  (Tidak ada teks yang terdeteksi)")

    print("-" * 55)
    print(f"  Total Baris Terbaca : {total_lines}")
    print(f"  Waktu Inferensi     : {infer_time:.2f} detik ({infer_time * 1000:.1f} ms)")
    print("-" * 55)
    print("\n[SUCCESS] Surya OCR 100% SIAP DIGUNAKAN!\n")

if __name__ == "__main__":
    main()

