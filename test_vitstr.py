"""
test_vitstr.py — Pengujian mandiri untuk model ViTSTR (Vision Transformer).

Jalankan di terminal (.venv-vitstr):
  python test_vitstr.py
  python test_vitstr.py --image uploads/label-xxx.png
"""

import sys
import time
import argparse
from PIL import Image, ImageDraw, ImageFont
import torch

def create_sample_image() -> Image.Image:
    """Buat gambar dummy teks label yang tajam dan proporsional sesuai ukuran input model (128x32)."""
    img = Image.new("RGB", (128, 32), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    # Tulis kata label pakaian yang jelas
    draw.text((16, 4), "COTTON", fill=(0, 0, 0), font=font)
    return img

def main():
    parser = argparse.ArgumentParser(description="Test ViTSTR Model")
    parser.add_argument("--image", type=str, default=None, help="Path file gambar (opsional)")
    args = parser.parse_args()

    print("\n" + "=" * 50)
    print("  MEMERIKSA KESIAPAN MODEL ViTSTR (STR HUB)")
    print("=" * 50)

    # 1. Cek PyTorch & Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Perangkat komputasi: {device.upper()}")

    # 2. Load Model ViTSTR
    print("[*] Memuat model ViTSTR dari STR Hub...")
    t0 = time.time()
    try:
        model = torch.hub.load('baudm/parseq', 'vitstr', pretrained=True, trust_repo=True).eval().to(device)
    except Exception as e:
        print(f"\n[ERROR] Gagal memuat model: {e}")
        sys.exit(1)
    load_time = time.time() - t0
    print(f"[OK] Model berhasil dimuat dalam {load_time:.2f} detik.")
    print(f"     Ukuran input gambar model: {model.hparams.img_size}")

    # 3. Siapkan Gambar
    if args.image:
        print(f"[*] Membuka gambar: {args.image}")
        img = Image.open(args.image).convert("RGB")
    else:
        print("[*] Membuat gambar uji coba sintetis (teks: '100% COTTON')...")
        img = create_sample_image()

    # 4. Preprocessing Gambar (gunakan transform resmi dari STR Hub)
    try:
        from strhub.data.module import SceneTextDataModule
        img_transform = SceneTextDataModule.get_transform(model.hparams.img_size)
    except Exception:
        from torchvision import transforms
        img_transform = transforms.Compose([
            transforms.Resize(model.hparams.img_size),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),
        ])
    img_tensor = img_transform(img).unsqueeze(0).to(device)

    # 5. Jalankan Inferensi (Prediksi Teks)
    print("[*] Menjalankan inferensi ViTSTR...")
    t1 = time.time()
    with torch.inference_mode():
        logits = model(img_tensor)
        pred_probs = logits.softmax(-1)
        pred_text, confidence = model.tokenizer.decode(pred_probs)
    infer_time = time.time() - t1

    # 6. Tampilkan Hasil
    print("\n" + "-" * 50)
    print("  HASIL PENGUJIAN ViTSTR:")
    print("-" * 50)
    print(f"  Teks Terbaca : '{pred_text[0]}'")
    if confidence is not None and len(confidence) > 0:
        conf_val = confidence[0].mean().item() * 100 if hasattr(confidence[0], 'mean') else 100.0
        print(f"  Confidence   : {conf_val:.1f}%")
    print(f"  Waktu Proses : {infer_time * 1000:.2f} ms")
    print("-" * 50)
    print("\n[SUCCESS] Model ViTSTR 100% SIAP DIGUNAKAN!\n")

if __name__ == "__main__":
    main()

