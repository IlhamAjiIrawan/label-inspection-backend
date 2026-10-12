"""
scripts/bench_ocr.py — Benchmark OCR Fase 0 (OCR_OPTIMIZATION_PLAN Lampiran B).

Pemakaian:
  python scripts/bench_ocr.py --dir tests/ocr_bench --repeat 3 --out baseline.json
  python scripts/bench_ocr.py --dir tests/ocr_bench --repeat 1 --compare baseline.json
  python scripts/bench_ocr.py --dir tests/ocr_bench --batch5   (ukur batch 5 paralel)

Mengukur: total P50/P95/mean, decode/det/rec P50, similarity vs ground_truth,
critical exact-match, cold first request, batch 5 wall-clock.
Kontrak 'none' dihormati (text_present=false = valid).
"""
import argparse
import json
import statistics
import time
from difflib import SequenceMatcher
from pathlib import Path


def norm(s: str) -> str:
    return "\n".join(l.strip().lower() for l in (s or "").splitlines() if l.strip())


def pct(vals, p):
    if not vals:
        return None
    vals = sorted(vals)
    k = min(len(vals) - 1, int(round((p / 100) * (len(vals) - 1))))
    return round(vals[k], 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="folder tests/ocr_bench")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--out", default=None)
    ap.add_argument("--compare", default=None)
    ap.add_argument("--batch5", action="store_true", help="ukur batch 5 paralel via OCR_EXECUTOR")
    args = ap.parse_args()

    base = Path(args.dir)
    gt_path = base / "ground_truth.json"
    gt = json.loads(gt_path.read_text(encoding="utf-8")) if gt_path.exists() else {}
    images = sorted((base / "images").glob("*"))
    images = [p for p in images if p.is_file()]
    if not images:
        print(f"Tidak ada gambar di {base / 'images'}")
        return

    from app.ocr.dispatch import extract_label_text_from_bytes

    # Cold: request pertama (sebelum warm-up eksplisit).
    first = images[0].read_bytes()
    t = time.perf_counter()
    try:
        extract_label_text_from_bytes(first)
    except Exception as e:
        print(f"Cold request exception (lanjut): {e}")
    cold = time.perf_counter() - t

    totals, det, rec, decode = [], [], [], []
    sims, crit_hits, crit_total = [], 0, 0

    for img in images:
        data = img.read_bytes()
        expected = gt.get(img.name, {})
        for _ in range(args.repeat):
            t0 = time.perf_counter()
            raw = extract_label_text_from_bytes(data)
            # Dispatch sudah mem-build via parser (threshold ENV) — pakai langsung.
            res = {"cleaned_text": raw.get("cleaned_text", ""), "lines": raw.get("lines", [])}
            totals.append(time.perf_counter() - t0)
            tm = raw.get("timings") or {}
            try:
                if tm.get("decode_s") is not None:
                    decode.append(float(tm["decode_s"]))
                if tm.get("det_s") is not None:
                    det.append(float(tm["det_s"]))
                if tm.get("rec_s") is not None:
                    rec.append(float(tm["rec_s"]))
            except Exception:
                pass

        if expected:
            try:
                got = norm(res.get("cleaned_text", ""))
                sims.append(SequenceMatcher(None, norm(expected.get("text", "")), got).ratio())
                for cl in expected.get("critical_lines", []):
                    crit_total += 1
                    crit_hits += int(norm(cl) in got)
            except Exception:
                pass

    report = {
        "images": len(images),
        "repeat": args.repeat,
        "cold_first_request_s": round(cold, 4),
        "total_s": {"p50": pct(totals, 50), "p95": pct(totals, 95), "mean": round(statistics.mean(totals), 4) if totals else None},
        "decode_s_p50": pct(decode, 50),
        "det_s_p50": pct(det, 50),
        "rec_s_p50": pct(rec, 50),
        "similarity_mean": round(statistics.mean(sims), 4) if sims else None,
        "critical_exact_match": round(crit_hits / crit_total, 4) if crit_total else None,
    }

    # Batch 5 paralel (R6).
    if args.batch5:
        try:
            from concurrent.futures import wait
            from app.ocr.executor import OCR_EXECUTOR

            batch = [images[i % len(images)].read_bytes() for i in range(5)]
            t = time.perf_counter()
            futs = [OCR_EXECUTOR.submit(extract_label_text_from_bytes, b) for b in batch]
            wait(futs)
            # Pastikan tidak ada exception tersembunyi.
            for f in futs:
                try:
                    f.result()
                except Exception as e:
                    print(f"Batch slot exception: {e}")
            report["batch5_s"] = round(time.perf_counter() - t, 3)
        except Exception as e:
            report["batch5_s"] = None
            report["batch5_error"] = str(e)[:200]

    print(json.dumps(report, indent=2))

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Ditulis ke {args.out}")

    if args.compare:
        b = json.loads(Path(args.compare).read_text(encoding="utf-8"))
        print("\n=== PERBANDINGAN vs BASELINE ===")
        try:
            print(f"P50   : {b['total_s']['p50']} -> {report['total_s']['p50']}")
            print(f"P95   : {b['total_s']['p95']} -> {report['total_s']['p95']}")
            print(f"Sim   : {b.get('similarity_mean')} -> {report.get('similarity_mean')}")
            print(f"Crit  : {b.get('critical_exact_match')} -> {report.get('critical_exact_match')}")
            if b.get("batch5_s") and report.get("batch5_s"):
                print(f"Batch5: {b['batch5_s']} -> {report['batch5_s']}")
        except Exception as e:
            print(f"Compare gagal: {e}")


if __name__ == "__main__":
    main()
