"""Generate synthetic bench images with known ground truth (for Fase 0).

- White/colored bg, black text, various sizes (normal, small batch text, dark).
- Ground truth written to tests/ocr_bench/ground_truth.json (human-verifiable).
Run: python scripts/make_bench_images.py
"""
from pathlib import Path
from PIL import Image, ImageDraw
import json

OUT = Path("tests/ocr_bench/images")
OUT.mkdir(parents=True, exist_ok=True)

CASES = [
    ("synth_normal_01.png", (1280, 720), [
        "TEH BOTOL",
        "SOSRO",
        "450ml",
        "Batch: 20261001",
        "Exp: 20271001",
    ], 48, "white"),
    ("synth_normal_02.png", (1280, 720), [
        "INDOMIE GORENG",
        "85g",
        "Batch: A12345",
        "MFG: 20260101",
        "Exp: 20270101",
    ], 48, "white"),
    ("synth_small_01.png", (1280, 720), [
        "Product: TEST-123",
        "BATCH B20261006",
        "Lot: 998877",
        "Netto 250gr",
    ], 28, "white"),
    ("synth_dark_01.png", (1280, 720), [
        "KOPI KAPAL API",
        "Batch: 20261001",
        "Exp: 20271001",
    ], 48, "dark"),
    ("synth_tilted_01.png", (1280, 720), [
        "SUSU UHT FULL CREAM",
        "1000ml",
        "Batch: 556677",
    ], 44, "white"),
    ("synth_1920_01.png", (1920, 1080), [
        "TEH BOTOL SOSRO",
        "450ml",
        "Batch: 20261001 Exp: 20271001",
        "PT SINAR SOSRO",
    ], 64, "white"),
]

gt = {}
# Keep existing real captures as 'none' contract cases (if present).
for p in sorted(OUT.glob("label_*.png")):
    gt[p.name] = {"text": "none", "critical_lines": []}

from PIL import ImageFont as _IF

def _font(sz: int):
    for cand in ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/ARIAL.TTF", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return _IF.truetype(cand, sz)
        except Exception:
            continue
    try:
        return _IF.load_default()
    except Exception:
        return None

for fname, size, lines, fsize, mode in CASES:
    bg = (40, 40, 40) if mode == "dark" else (255, 255, 255)
    fg = (230, 230, 230) if mode == "dark" else (0, 0, 0)
    img = Image.new("RGB", size, bg)
    d = ImageDraw.Draw(img)
    font = _font(fsize)
    y = 60
    for ln in lines:
        try:
            d.text((60, y), ln, fill=fg, font=font)
        except Exception:
            try:
                d.text((60, y), ln, fill=fg)
            except Exception:
                pass
        y += fsize + 24
    img.save(OUT / fname)
    gt[fname] = {"text": "\n".join(lines), "critical_lines": [l for l in lines if any(k in l for k in ("Batch", "Exp", "MFG", "Lot"))]}

Path("tests/ocr_bench/ground_truth.json").write_text(json.dumps(gt, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {len(CASES)} synthetic + {len(gt)-len(CASES)} real, total gt={len(gt)}")
