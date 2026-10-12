"""
test_paddle_ocr.py — Benchmark standalone engine OCR baru (RapidOCR/Paddle).

Jalankan:
  python test_paddle_ocr.py --image uploads/label-xxx.png
  python test_paddle_ocr.py --batch CHK-xxx   (uji semua slot batch)

Tanpa argumen: pakai gambar label terbaru di uploads/.

Output: tabel per file → engine | text_present | cleaned_text (120 char) | waktu.
Dipakai untuk acceptance §8 plans/OCR_ENGINE_UPGRADE_PADDLE_PLAN.md.
"""

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

# Tambah project root ke sys.path agar bisa import app.*
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env (agar OCR_ENGINE + threshold ikut)
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    print("[WARN] python-dotenv not installed, using system env vars only")


def find_check_images(check_dir: str) -> list[str]:
    """Cari semua file gambar slot di folder check, urut slot."""
    files = []
    for pat in ["slot*.png", "slot*.jpg", "slot*.jpeg"]:
        files.extend(glob.glob(os.path.join(check_dir, pat)))
    files.sort()
    return files


def bench_one(image_path: str) -> dict:
    """OCR 1 file via dispatcher, return ringkasan + waktu."""
    from app.ocr.dispatch import extract_label_text

    t0 = time.time()
    res = extract_label_text(image_path, os.path.basename(image_path))
    elapsed = time.time() - t0
    return {
        "file": os.path.basename(image_path),
        "engine": res.get("engine"),
        "text_present": res.get("text_present"),
        "is_empty": res.get("is_empty"),
        "status": "success" if (res.get("text_present") or res.get("cleaned_text")) else "failed",
        "cleaned_text": (res.get("cleaned_text") or "")[:120],
        "lines": len(res.get("lines") or []),
        "seconds": round(elapsed, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark OCR RapidOCR/Paddle")
    parser.add_argument("--image", type=str, default=None, help="Path 1 gambar label")
    parser.add_argument("--batch", type=str, default=None, help="Batch code (folder uploads/check/{batch})")
    args = parser.parse_args()

    targets: list[str] = []
    if args.image:
        targets = [str(Path(args.image).resolve())]
    elif args.batch:
        check_dir = str(PROJECT_ROOT / "uploads" / "check" / args.batch)
        targets = [str(Path(p).resolve()) for p in find_check_images(check_dir)]
        if not targets:
            print(f"ERROR: tidak ada slot di {check_dir}")
            sys.exit(1)
    else:
        uploads_dir = PROJECT_ROOT / "uploads"
        label_files = list(uploads_dir.glob("label-*.png")) + list(uploads_dir.glob("label-*.jpg"))
        if not label_files:
            print("ERROR: tidak ada label di uploads/")
            sys.exit(1)
        label_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        targets = [str(label_files[0].resolve())]
        print(f"Auto-detected: {label_files[0].name}")

    from app.config import OCR_ENGINE

    print()
    print("=" * 78)
    print(f"  OCR BENCHMARK — engine flag: {OCR_ENGINE}")
    print("=" * 78)

    results = []
    total_t0 = time.time()
    for tp in targets:
        try:
            r = bench_one(tp)
        except Exception as e:
            r = {"file": os.path.basename(tp), "engine": "?", "text_present": None,
                 "is_empty": None, "status": f"EXCEPTION: {e}", "cleaned_text": "",
                 "lines": 0, "seconds": 0}
        results.append(r)
        flag = "NONE" if r["cleaned_text"] == "none" else ("OK" if r["text_present"] else "?")
        print(f"  [{flag}] {r['file']} | engine={r['engine']} "
              f"present={r['text_present']} lines={r['lines']} {r['seconds']}s")
        print(f"       cleaned: {r['cleaned_text']!r}")

    print("=" * 78)
    total = round(time.time() - total_t0, 2)
    n_none = sum(1 for r in results if r["cleaned_text"] == "none")
    n_ok = sum(1 for r in results if r["text_present"])
    print(f"  SUMMARY: {n_ok}/{len(results)} WITH TEXT, {n_none}/{len(results)} NONE, total {total}s")
    print("=" * 78)
    print(json.dumps(results, indent=2, ensure_ascii=False))
    return results


if __name__ == "__main__":
    main()
