"""
test_similarity.py — Unit test compare similarity teks OCR.

COMPARE_TEXT_SIMILARITY_PLAN §9 (AC1-AC7). Jalankan:
  python test_similarity.py

Output: tabel kasus + SUMMARY. Assert AC5: 5 pasang teks < 5 ms.
JANGAN ubah test_paddle_ocr.py / test_native_compare.py.
"""

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.compare_text.similarity import compare_batch, compare_slot, normalize_text  # noqa: E402

THR = 0.85
CASES = [
    ("identik", "PT MAJU JAYA\nNETTO 50 KG", "PT MAJU JAYA\nNETTO 50 KG",
     ["PT MAJU JAYA", "NETTO 50 KG"], "PASS", None),
    ("h_k_typo", "HANDPHONE", "KANDPHONE", ["KANDPHONE"], "PASS", None),
    ("beda_total", "PT MAJU JAYA\nNETTO 50 KG", "TOKO BUKU SEJAHTERA",
     ["TOKO BUKU SEJAHTERA"], "FAIL", None),
    ("slot_kosong", "PT MAJU JAYA", "", [], "FAIL", "empty_text"),
    ("ref_kosong", "", "ANY TEXT", ["ANY TEXT"], "SKIP", "no_reference_text"),
    ("ref_blank_ok", "none", "none", [], "PASS", "blank_match"),
    ("ref_blank_empty", "none", "", [], "PASS", "blank_match"),
    ("ref_blank_fail", "none", "ANY TEXT", ["ANY TEXT"], "FAIL", "unexpected_text_on_blank"),
    ("real_id_identik",
     "FREE PASS\nBAG\nHANDPHONE\nA21407\nAJI DW PRAKOSO\nNIK\n251200203\nOPERATION E\nSECURITY ACCESS",
     "FREE PASS\nBAG\nHANDPHONE\nA21407\nAJI DW PRAKOSO\nNIK\n251200203\nOPERATION E\nSECURITY ACCESS",
     ["FREE PASS", "BAG", "HANDPHONE", "A21407", "AJI DW PRAKOSO", "NIK",
      "251200203", "OPERATION E", "SECURITY ACCESS"], "PASS", None),
    ("real_id_1_typo",
     "FREE PASS\nBAG\nHANDPHONE\nA21407\nAJI DW PRAKOSO\nNIK\n251200203\nOPERATION E\nSECURITY ACCESS",
     "FREE PASS\nBAG\nKANDPHONE\nA21407\nAJI DW PRAKOSO\nNIK\n251200203\nOPERATION E\nSECURITY ACCESS",
     ["FREE PASS", "BAG", "KANDPHONE", "A21407", "AJI DW PRAKOSO", "NIK",
      "251200203", "OPERATION E", "SECURITY ACCESS"], "PASS", None),
    # TOLERANSI TANDA BACA/SPASI (2026-10-09): beda hanya titik/koma/spasi
    # wajib PASS 1.0 — sebelumnya teks pendek bisa FAIL karena 1 char
    # menempati porsi besar SequenceMatcher.
    ("titik_trailing", "HELLO WORLD.", "HELLO WORLD",
     ["HELLO WORLD"], "PASS", None),
    ("titik_tengah", "PT. MAJU JAYA", "PT MAJU JAYA",
     ["PT MAJU JAYA"], "PASS", None),
    ("spasi_ganda", "PT  MAJU   JAYA", "PT MAJU JAYA",
     ["PT MAJU JAYA"], "PASS", None),
    ("koma_spasi", "A, B, C", "A B C",
     ["A B C"], "PASS", None),
    ("desimal_koma_vs_titik", "NETTO 1,5 KG", "NETTO 1.5 KG",
     ["NETTO 1.5 KG"], "PASS", None),
    # ROBUST SEGMENTASI (2026-10-09): acuan 1 baris vs OCR banyak baris —
    # kata 99% sama wajib PASS (sebelumnya F1 per-baris runtuh → FAIL).
    ("segmentasi_asimetris", "A B C D E F G H I J K L M N O P",
     "A B C D\nE F G H\nI J K L\nM N O P",
     ["A B C D", "E F G H", "I J K L", "M N O P"], "PASS", None),
    # TOLERANSI HURUF KAPITAL (2026-10-09): beda besar/kecil huruf TIDAK
    # boleh memengaruhi verdict — isi konten sama → PASS 1.0, tanpa diff.
    ("huruf_kapital_campur", "FabriCADO", "FABRICADO",
     ["FABRICADO"], "PASS", None),
    ("huruf_kecil_semua", "fabricado na indonesia", "FABRICADO NA INDONESIA",
     ["FABRICADO NA INDONESIA"], "PASS", None),
    ("huruf_kapital_kalimat", "Hecho En Indonesia", "HECHO EN INDONESIA",
     ["HECHO EN INDONESIA"], "PASS", None),
    # CONTENT-GATE (2026-10-09): beda konten sungguhan wajib FAIL walau skor
    # tinggi — skor saja tidak cukup karena 90% kata bisa sama pada beda label.
    # Kasus nyata: beda ukuran (G/GRANDE vs XS/PP) + kata tambahan + XO hilang.
    ("xo_beda_label",
     "1/2 XO CAZADORA TALLA: G/GRANDE TAM: G EN: BODY SHELL",
     "1/2 CAZADORA TALLA: XS EXCH EXTRA CHICA TAM: PP EN: BODY SHELL",
     ["1/2 CAZADORA TALLA: XS EXCH EXTRA CHICA TAM: PP EN: BODY SHELL"],
     "FAIL", "content_mismatch"),
    # Kata tambahan (bukan typo) = beda konten → FAIL via gate.
    ("kata_tambahan", "PT MAJU JAYA", "PT MAJU JAYA ABADI",
     ["PT MAJU JAYA ABADI"], "FAIL", "content_mismatch"),
]


def main():
    print("=" * 78)
    print("  COMPARE SIMILARITY TEST — threshold", THR)
    print("=" * 78)
    passed = 0
    total = 0
    results = []
    for name, ref, slot, lines, exp_verdict, exp_reason in CASES:
        total += 1
        out = compare_slot(ref, slot, lines, THR)
        ok = out["verdict"] == exp_verdict and out["reason"] == exp_reason
        if ok:
            passed += 1
        flag = "OK " if ok else "ERR"
        results.append({"case": name, "expected": exp_verdict, "got": out,
                        "pass": ok})
        print(f"  [{flag}] {name:16s} verdict={out['verdict']:4s} "
              f"score={out['score']} reason={out['reason']}")
        assert ok, f"FAIL case {name}: {out}"

    # TOLERANSI TANDA BACA (2026-10-09): diff punct-only wajib difilter,
    # desimal beda wajib tercatat + flag numeric_diff (walau verdict PASS).
    _punct = compare_slot("HELLO WORLD.", "HELLO WORLD", ["HELLO WORLD"], THR)
    assert _punct["score"] == 1.0 and _punct["diff_details"] == [], f"punct filter gagal: {_punct}"
    assert _punct.get("punct_ignored_count", 0) >= 1, f"ignored count hilang: {_punct}"
    assert _punct.get("numeric_diff") is False
    _num = compare_slot("NETTO 1.5 KG", "NETTO 15 KG", ["NETTO 15 KG"], THR)
    assert len(_num["diff_details"]) > 0, f"diff desimal hilang: {_num}"
    assert _num.get("numeric_diff") is True, f"numeric_diff flag hilang: {_num}"
    print(f"  punct_only: score={_punct['score']} ndiff={len(_punct['diff_details'])} ignored={_punct.get('punct_ignored_count')}")
    print(f"  decimal: score={_num['score']} verdict={_num['verdict']} numeric_diff={_num.get('numeric_diff')}")

    # TOLERANSI HURUF KAPITAL (2026-10-09): beda case wajib skor 1.0 dan
    # TANPA diff (semua jalur normalisasi me-uppercase sebelum banding).
    _caps = compare_slot("FabriCADO", "FABRICADO", ["FABRICADO"], THR)
    assert _caps["score"] == 1.0 and _caps["diff_details"] == [], f"case tolerance gagal: {_caps}"
    print(f"  caps_only: score={_caps['score']} ndiff={len(_caps['diff_details'])}")

    # CONTENT-GATE (2026-10-09): beda konten sungguhan wajib FAIL walau skor
    # tinggi. Angka/desimal dikecualikan (jalur numeric_diff + warning).
    # Kasus ringkas ini FAIL via skor; unexplained_orphans tetap terisi.
    _xo_r = "1/2 XO CAZADORA TALLA G GRANDE TAM G BODY"
    _xo_s = "1/2 CAZADORA TALLA XS EXCH EXTRA CHICA TAM PP BODY"
    _xo = compare_slot(_xo_r, _xo_s, [_xo_s], THR)
    assert _xo["verdict"] == "FAIL" and _xo.get("unexplained_orphans", 0) > 0, f"gate gagal: {_xo}"
    _dig = compare_slot("NETTO 50 KG", "NETTO 55 KG", ["NETTO 55 KG"], THR)
    assert _dig["verdict"] == "PASS" and _dig.get("numeric_diff") is True, f"kebijakan digit berubah: {_dig}"
    print(f"  gate: xo={_xo['verdict']}/{_xo['reason']} orphans={_xo.get('unexplained_orphans')} | digit={_dig['verdict']}+warn")

    # AC5: overhead < 5 ms untuk 5 pasang teks (exclude import/lock noise)
    ref_big = "FREE PASS\nBAG\nHANDPHONE\nA21407\nAJI DW PRAKOSO\nNIK\n251200203\nOPERATION E\nSECURITY ACCESS"
    slot_big = ref_big
    t0 = time.perf_counter()
    for _ in range(5):
        compare_slot(ref_big, slot_big, slot_big.splitlines(), THR)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    perf_ok = elapsed_ms < 5.0
    print(f"  PERF: 5 pasang = {elapsed_ms:.3f} ms ({'OK' if perf_ok else 'TOO SLOW'})")
    assert perf_ok, "AC5 gagal: compare > 5ms"

    # compare_batch
    batch = compare_batch(ref_big, [
        {"slot": 1, "similarity": {"verdict": "PASS", "score": 0.97}},
        {"slot": 2, "similarity": {"verdict": "FAIL", "score": 0.31}},
        {"slot": 3, "similarity": {"verdict": "FAIL", "score": 0.40}},
        {"slot": 4, "similarity": {"verdict": "PASS", "score": 0.96}},
    ], THR)
    assert batch["overall"] == "MIXED" and batch["pass_count"] == 2 and batch["fail_count"] == 2
    print(f"  batch summary: pass={batch['pass_count']} fail={batch['fail_count']} "
          f"skip={batch['skip_count']} overall={batch['overall']}")

    # normalize
    assert normalize_text("  pt MAJU\n  Jaya  ") == "PT MAJU JAYA"

    print("=" * 78)
    print(f"  SUMMARY: {passed}/{total} cases passed, perf_ok={perf_ok}")
    print("=" * 78)
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()