"""
test_native_compare.py — Perbandingan kandidat OCR untuk PADDLEOCR_NATIVE_PLAN §8.

Konteks (§6.0 VERDICT RED): PaddleOCR native tak terinstall di Python 3.14
(paddlepaddle tanpa wheel) → kolom "native" selalu UNAVAILABLE. Yang
dibandingkan: RapidOCR PP-OCRv6_rec_small (default cepat) vs
PP-OCRv6_rec_medium (kandidat akurat, fix HANDPHONE->KANDPHONE).

Jalankan:
  python test_native_compare.py --dir uploads/_bench_native

Tanpa argumen: pakai uploads/_bench_native.

Output: tabel per file → small (teks|waktu|skor) vs medium (teks|waktu|skor),
status HANDPHONE lulus/gagal, SUMMARY + dump JSON.
JANGAN ubah test_paddle_ocr.py.
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    print("[WARN] python-dotenv not installed, using system env vars only")


def load_ground_truth(d: Path) -> dict:
    gt_path = d / "ground_truth.csv"
    if not gt_path.exists():
        return {}
    with open(gt_path, encoding="utf-8") as f:
        return {r["filename"]: r["expected_text"]
                for r in csv.DictReader(f)}


def score_keywords(cleaned: str, expected: str) -> tuple[int, int]:
    """Recall keyword: expected dipisah '|'. Blank (expected '') lulus bila 'none'."""
    keys = [k.strip().upper() for k in (expected or "").split("|") if k.strip()]
    if not keys:
        return (1, 1) if cleaned.strip().lower() == "none" else (0, 1)
    got = cleaned.upper()
    hit = sum(1 for k in keys if k in got)
    return hit, len(keys)


def run_profile(image_path: str, rec_model: str) -> dict:
    """OCR 1 file dengan 1 profil rec model (instans RapidOCR terpisah)."""
    from rapidocr import RapidOCR
    from rapidocr.utils.typings import ModelType, OCRVersion

    eng = RapidOCR(params={
        "Rec.ocr_version": OCRVersion.PPOCRV6,
        "Rec.model_type": ModelType.MEDIUM if rec_model == "medium" else ModelType.SMALL,
        "Rec.lang_type": "en",
    })
    t0 = time.time()
    out = eng(image_path)
    dt = time.time() - t0
    txts = [str(t).strip() for t in (out.txts or []) if str(t or "").strip()]
    cleaned = "\n".join(txts) if txts else "none"
    return {"cleaned": cleaned, "lines": len(txts), "seconds": round(dt, 2)}


def main():
    ap = argparse.ArgumentParser(description="Compare rapid-small vs rapid-medium (+native status)")
    ap.add_argument("--dir", type=str, default=str(PROJECT_ROOT / "uploads" / "_bench_native"))
    args = ap.parse_args()
    d = Path(args.dir)
    targets = sorted(d.glob("*.png")) + sorted(d.glob("*.jpg"))
    if not targets:
        print(f"ERROR: tidak ada gambar di {d}")
        sys.exit(1)
    gt = load_ground_truth(d)

    try:
        from app.ocr.paddle_native_engine import is_native_available
        native_ok = is_native_available()
    except Exception:
        native_ok = False
    print()
    print("=" * 78)
    print(f"  OCR COMPARE — native_available={native_ok} (ekspektasi False di py3.14)")
    print("=" * 78)

    # Build kedua engine sekali (hemat waktu load model).
    from rapidocr import RapidOCR
    from rapidocr.utils.typings import ModelType, OCRVersion
    engines = {
        "small": RapidOCR(params={"Rec.ocr_version": OCRVersion.PPOCRV6,
                                  "Rec.model_type": ModelType.SMALL,
                                  "Rec.lang_type": "en"}),
        "medium": RapidOCR(params={"Rec.ocr_version": OCRVersion.PPOCRV6,
                                   "Rec.model_type": ModelType.MEDIUM,
                                   "Rec.lang_type": "en"}),
    }

    results = []
    for tp in targets:
        row = {"file": tp.name, "expected": gt.get(tp.name, "")}
        for prof, eng in engines.items():
            t0 = time.time()
            out = eng(str(tp))
            dt = time.time() - t0
            txts = [str(t).strip() for t in (out.txts or []) if str(t or "").strip()]
            cleaned = "\n".join(txts) if txts else "none"
            hit, total = score_keywords(cleaned, row["expected"])
            row[prof] = {"cleaned": cleaned[:120], "lines": len(txts),
                         "seconds": round(dt, 2), "hit": hit, "total": total,
                         "recall": round(hit / total, 2) if total else 0.0}
        results.append(row)
        s, m = row["small"], row["medium"]
        flag = "HANDPHONE-OK" if "HANDPHONE" in m["cleaned"] else ""
        print(f"  {row['file']}")
        print(f"    small : {s['seconds']:5.2f}s recall={s['hit']}/{s['total']} {s['cleaned']!r}")
        print(f"    medium: {m['seconds']:5.2f}s recall={m['hit']}/{m['total']} {m['cleaned']!r} {flag}")

    def agg(prof):
        rs = [r[prof] for r in results]
        full = sum(1 for r in rs if r["hit"] == r["total"] and r["total"] > 0)
        avg_rec = sum(r["recall"] for r in rs) / len(rs) if rs else 0
        avg_t = sum(r["seconds"] for r in rs) / len(rs) if rs else 0
        return full, round(avg_rec, 2), round(avg_t, 2)

    fs, rs, ts = agg("small")
    fm, rm, tm = agg("medium")
    hp_small = sum(1 for r in results if "HANDPHONE" in r["small"]["cleaned"])
    hp_med = sum(1 for r in results if "HANDPHONE" in r["medium"]["cleaned"])
    print("=" * 78)
    print(f"  SUMMARY ({len(results)} files):")
    print(f"    small : full={fs}/{len(results)} avg_recall={rs} avg_time={ts}s HANDPHONE={hp_small}")
    print(f"    medium: full={fm}/{len(results)} avg_recall={rm} avg_time={tm}s HANDPHONE={hp_med}")
    print(f"    native: {'AVAILABLE' if native_ok else 'UNAVAILABLE (§6.0 RED)'}")
    print("=" * 78)
    print(json.dumps({"native_available": native_ok, "results": results,
                      "summary": {"small": {"full": fs, "avg_recall": rs, "avg_time": ts},
                                  "medium": {"full": fm, "avg_recall": rm, "avg_time": tm}}},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
