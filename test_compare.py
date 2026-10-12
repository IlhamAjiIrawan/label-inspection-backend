"""
test_compare.py — Script standalone untuk testing Visual Comparison via Gemini.

Jalankan:
  python test_compare.py

Atau dengan argumen custom:
  python test_compare.py --reference uploads/label-xxx.png --batch CHK-xxx

Tanpa argumen, script menggunakan file contoh yang sudah ada di uploads/.

Dependencies:
  pip install google-genai python-dotenv
"""

import os
import sys
import json
import argparse
import glob
from pathlib import Path

# Tambah project root ke sys.path agar bisa import app.*
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    print("[WARN] python-dotenv not installed, using system env vars only")


def find_check_images(check_dir: str) -> list[str]:
    """Cari semua file gambar slot di folder check, urut slot."""
    patterns = ["slot*.png", "slot*.jpg", "slot*.jpeg"]
    files = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(check_dir, pat)))
    files.sort()
    return files


def main():
    parser = argparse.ArgumentParser(
        description="Test Visual Comparison — Gemini 1.5 Flash",
    )
    parser.add_argument("--reference", type=str, default=None,
                        help="Path ke gambar reference. Default: terbaru di uploads/")
    parser.add_argument("--batch", type=str, default=None,
                        help="Batch code. Default: dari latest_batch.txt")
    parser.add_argument("--check-dir", type=str, default=None,
                        help="Path langsung ke folder check images. Override --batch")
    parser.add_argument("--api-key", type=str, default=None,
                        help="Gemini API key. Default: dari .env")
    parser.add_argument("--model", type=str, default=None,
                        help="Model Gemini. Default: dari .env atau gemini-3.8-flash")
    parser.add_argument("--no-save", action="store_true",
                        help="Jangan simpan hasil ke file")

    args = parser.parse_args()

    # --- Resolve API Key ---
    api_key = args.api_key or os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        print("ERROR: GEMINI_API_KEY tidak ditemukan!")
        print("  Set di .env: GEMINI_API_KEY=your_key_here")
        sys.exit(1)

    model = args.model or os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

    # --- Resolve Reference Image ---
    if args.reference:
        reference_path = str(Path(args.reference).resolve())
    else:
        uploads_dir = PROJECT_ROOT / "uploads"
        label_files = list(uploads_dir.glob("label-*.png")) + list(uploads_dir.glob("label-*.jpg"))
        if not label_files:
            print("ERROR: Tidak ditemukan gambar label di uploads/")
            sys.exit(1)
        label_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        reference_path = str(label_files[0])
        print(f"Auto-detected reference: {label_files[0].name}")

    # --- Resolve Check Images ---
    if args.check_dir:
        check_dir = str(Path(args.check_dir).resolve())
    elif args.batch:
        check_dir = str(PROJECT_ROOT / "uploads" / "check" / args.batch)
    else:
        latest_batch_file = PROJECT_ROOT / "uploads" / "check" / "latest_batch.txt"
        if latest_batch_file.exists():
            batch_code = latest_batch_file.read_text().strip()
            check_dir = str(PROJECT_ROOT / "uploads" / "check" / batch_code)
            print(f"Auto-detected batch: {batch_code}")
        else:
            check_root = PROJECT_ROOT / "uploads" / "check"
            batch_dirs = [d for d in check_root.iterdir() if d.is_dir() and d.name.startswith("CHK-")]
            if not batch_dirs:
                print("ERROR: Tidak ditemukan folder batch di uploads/check/")
                sys.exit(1)
            batch_dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
            check_dir = str(batch_dirs[0])
            print(f"Auto-detected batch folder: {batch_dirs[0].name}")

    if not os.path.isdir(check_dir):
        print(f"ERROR: Folder check tidak ditemukan: {check_dir}")
        sys.exit(1)

    check_images = find_check_images(check_dir)
    if not check_images:
        print(f"ERROR: Tidak ditemukan gambar slot di: {check_dir}")
        sys.exit(1)

    batch_code = args.batch or os.path.basename(check_dir)

    # --- Print summary ---
    print()
    print("=" * 60)
    print("  VISUAL COMPARISON TEST — Gemini 1.5 Flash")
    print("=" * 60)
    print(f"  Model      : {model}")
    print(f"  Reference  : {os.path.basename(reference_path)}")
    print(f"  Batch      : {batch_code}")
    print(f"  Check files: {len(check_images)}")
    for i, cp in enumerate(check_images):
        print(f"    Slot {i+1}: {os.path.basename(cp)}")
    print("=" * 60)
    print()

    # --- Jalankan Comparison ---
    from app.compare.service import compare_images_with_reference

    result = compare_images_with_reference(
        reference_image_path=reference_path,
        check_image_paths=check_images,
        api_key=api_key,
        model=model,
        timeout=float(os.getenv("GEMINI_COMPARE_TIMEOUT", "30")),
        batch_code=batch_code,
        save_result=not args.no_save,
        save_dir=check_dir if not args.no_save else None,
    )

    # --- Print hasil ---
    print()
    print("=" * 60)
    print("  HASIL COMPARISON")
    print("=" * 60)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("=" * 60)

    if result.get("status") == "success" and result.get("results"):
        matches = sum(1 for r in result["results"] if r.get("is_match"))
        total = len(result["results"])
        print(f"\nSUMMARY: {matches}/{total} MATCH, {total - matches}/{total} MISMATCH")
        print(f"Processing time: {result.get('processing_time_seconds', '?')}s")
    elif result.get("status") == "failed":
        print(f"\nFAILED: {result.get('error', 'Unknown error')}")

    return result


if __name__ == "__main__":
    main()
