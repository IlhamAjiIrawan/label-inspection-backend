"""
app/compare_text/similarity.py — Compare similarity teks OCR (reference vs slot).

COMPARE_TEXT_SIMILARITY_PLAN (2026-09-12). Stdlib `difflib` saja:
- Tanpa dependensi baru, tanpa GPU, tanpa LLM.
- Target overhead <5 ms untuk 5 pasang teks (lihat test_similarity.py AC5).
- TIDAK PERNAH raise — semua fungsi bungkus try/except → verdict SKIP + reason.

Kontrak result (plan §4.1):
  {"version","verdict","score","threshold","ref_lines","slot_lines",
   "matched_lines","reason"}
verdict: PASS | FAIL | SKIP (huruf besar).
reason: null | "empty_text" | "no_reference_text" | "exception:..."

Catatan: folder ini (app/compare_text) DIBEDAKAN dari app/compare (Gemini visual).

MIGRASI RapidFuzz (2026-10-09 / PLAN_MIGRASI_DIFFLIB_KE_RAPIDFUZZ): scoring
memakai rapidfuzz bila SIMILARITY_ENGINE=rapidfuzz (default), fallback difflib.
Display diff (_build_diff_details) SENGAJA tetap difflib (Tahap 1).
"""

from difflib import SequenceMatcher
from collections import Counter
from typing import Optional
import re

try:
    from rapidfuzz import fuzz as _rf_fuzz
    _RAPIDFUZZ_AVAILABLE = True
except ImportError:
    _rf_fuzz = None
    _RAPIDFUZZ_AVAILABLE = False


def _use_rapidfuzz() -> bool:
    """True bila engine rapidfuzz aktif DAN library ter-install.

    Dukungan dinamis via runtime_config (Dashboard), dengan fallback ke
    SIMILARITY_ENGINE dari config / .env. Bila rapidfuzz gagal
    di-import, otomatis fallback ke difflib agar compare tidak pernah mati.
    """
    try:
        from app.runtime_config import runtime_config
        rc = runtime_config.get_config()
        want_rf = (rc.get("compare_engine") != "difflib")
    except Exception:
        try:
            from app.config import SIMILARITY_ENGINE as _eng
            want_rf = (str(_eng or "rapidfuzz").strip().lower() != "difflib")
        except Exception:
            import os as _os
            want_rf = (_os.getenv("SIMILARITY_ENGINE", "rapidfuzz").strip().lower() != "difflib")
    return bool(want_rf and _RAPIDFUZZ_AVAILABLE)


def _ratio_0_1(a: str, b: str) -> float:
    """Rasio kemiripan 0.0-1.0. RapidFuzz bila aktif, difflib bila fallback.

    PENTING: fuzz.ratio mengembalikan 0-100 → wajib dibagi 100 agar skala
    identik dengan SequenceMatcher.ratio (0-1). Jangan lupa pembagian ini!
    """
    if _use_rapidfuzz():
        return float(_rf_fuzz.ratio(a or "", b or "") / 100.0)
    return SequenceMatcher(None, a or "", b or "").ratio()


def _token_sort_hit(a: str, b: str) -> bool:
    """True bila kata-katanya sama persis, hanya urutan beda (OCR reorder).

    RapidFuzz: token_sort_ratio == 100. Fallback difflib: perbandingan
    sorted manual (perilaku lama, dipertahankan).
    """
    if _use_rapidfuzz():
        return bool(_rf_fuzz.token_sort_ratio(a or "", b or "") >= 100)
    return " ".join(sorted((a or "").split())) == " ".join(sorted((b or "").split()))

VERSION = "v1"

# Bobot skor: full-sequence ratio (60%) + F1 baris (40%). Lihat plan §6.2.
_FULL_WEIGHT = 0.6
_LINE_WEIGHT = 0.4
_LINE_MATCH_MIN = 0.65  # ★ Dinaikkan dari 0.5: kurangi false match (baris 50% mirip bukan match)

# Batas aman compare teks panjang (label fisik max ~500 char,
# tapi kode harus defensif terhadap input ekstrem).
# REVIEW_FLOW_COMPARE_FRONT_BACK_PLAN (2026-10-05) §5.3 Optimasi 2.
MAX_COMPARE_CHARS = 10_000   # truncate full-text SequenceMatcher
MAX_COMPARE_LINES = 200      # truncate line-matching loop

_PUNCT_EDGE = ".,:;/-()\"'"


def normalize_text(s: str) -> str:
    """Normalisasi teks penuh: uppercase + collapse whitespace (plan §6.1)."""
    return " ".join((s or "").upper().split())


def line_tokens(line: str) -> set:
    """Token 1 baris: pecah spasi, kupas tanda baca tepi umum OCR."""
    return {
        t.strip(_PUNCT_EDGE).strip()
        for t in (line or "").upper().split()
        if t.strip(_PUNCT_EDGE).strip()
    }


# Pattern untuk strip semua punctuation kecuali alfanumerik dan spasi
_PUNCT_ALL = re.compile(r'[^\w\s]', re.UNICODE)


def normalize_canonical(s: str, keep_decimal: bool = True) -> str:
    """Normalisasi toleran tanda baca/spasi untuk scoring compare.

    - Uppercase + collapse whitespace (seperti normalize_text), PLUS:
    - Semua punctuation (.,:;/-()"', dll) diubah jadi spasi lalu di-collapse,
      sehingga perbedaan titik/koma/strip di tepi token ("PT." vs "PT",
      "WORLD." vs "WORLD", "A,B" vs "A B") NOL pengaruh ke skor.
    - keep_decimal=True (default): titik/koma yang diapit digit di kedua sisi
      ("1.5", "805.011") DIPERTAHANKAN karena menyangkut nilai angka/kode.
      Set False bila ingin toleransi penuh (lebih agresif).

    TIDAK PERNAH raise (gagal -> "").
    """
    try:
        t = (s or "").upper()
        if keep_decimal:
            # Satukan koma desimal ala Indonesia ("1,5" -> "1.5") dulu,
            # lalu hapus titik yang BUKAN desimal (tepi token / non-digit).
            t = re.sub(r"(?<=\d),(?=\d)", ".", t)
            t = re.sub(r"(?<!\d)[.]|[.](?!\d)", " ", t)
            t = re.sub(r"[^\w\s.]", " ", t, flags=re.UNICODE)
        else:
            t = re.sub(r"[^\w\s]", " ", t, flags=re.UNICODE)
        return " ".join(t.split())
    except Exception:
        return ""


def normalize_token(token: str) -> str:
    """
    Normalisasi token lebih agresif untuk mengurangi noise OCR.

    "PT." → "PT", "50-KG" → "50KG", "(SIZE)" → "SIZE"

    Dipakai oleh line_tokens_v2 (improvement presisi).
    """
    t = (token or "").upper().strip()
    # Strip semua non-alphanumeric
    t = _PUNCT_ALL.sub("", t)
    return t.strip()


def line_tokens_v2(line: str) -> set:
    """
    Token 1 baris (v2): pecah spasi, normalize setiap token.

    Perbedaan dari line_tokens (v1):
    - v1: hanya strip PUNCT_EDGE (ujung token)
    - v2: strip SEMUA punctuation dari dalam token juga

    "PT. MAJU JAYA (50 KG)" → {"PT", "MAJU", "JAYA", "50", "KG"}
    """
    return {
        normalize_token(t)
        for t in (line or "").split()
        if normalize_token(t)
    }


def _canonical_flags() -> tuple:
    """Baca kill-switch toleransi tanda baca (butuh restart bila diubah).

    SIMILARITY_CANONICAL_PUNCT=false = kembali ke perilaku lama (tanpa
    toleransi punct di full_ratio/fast-path). SIMILARITY_KEEP_DECIMAL=false
    = toleransi penuh termasuk titik desimal (agresif, tidak disarankan).
    """
    try:
        from app.config import (  # type: ignore
            SIMILARITY_CANONICAL_PUNCT as _cp,
            SIMILARITY_KEEP_DECIMAL as _kd,
        )
        return bool(_cp), bool(_kd)
    except Exception:
        import os as _os
        return (
            (_os.getenv("SIMILARITY_CANONICAL_PUNCT", "true").lower() == "true"),
            (_os.getenv("SIMILARITY_KEEP_DECIMAL", "true").lower() == "true"),
        )


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _line_similarity(a_tokens: set, a_str: str, b_tokens: set, b_str: str) -> float:
    """
    Kesamaan 1 baris = max(Jaccard token, SequenceMatcher karakter).

    Karakter-level penting agar typo 1 huruf OCR (HANDPHONE→KANDPHONE = 0.889)
    tetap dianggap baris match — label fisik sama, noise OCR tidak boleh FAIL.

    Optimasi eksak (REVIEW_FLOW_COMPARE_FRONT_BACK_PLAN 2026-10-05):
    nilai ini HANYA dipakai untuk keputusan best >= _LINE_MATCH_MIN di
    _evaluate (nilai pastinya dibuang). Maka bila Jaccard sudah >= threshold,
    hasilnya pasti match → lewati SequenceMatcher O(n^2) yang mahal.
    Bila Jaccard < threshold, tetap hitung karakter (typo Jaccard=0!).
    """
    j = _jaccard(a_tokens, b_tokens)
    if j >= _LINE_MATCH_MIN:
        return j  # pasti match — hemat 1x ratio (hasil verdict identik)
    if not (a_str or b_str):
        return j
    c = _ratio_0_1(a_str, b_str)
    return max(j, c)


def _evaluate(ref_text: str, slot_cleaned: str, slot_lines: list) -> dict:
    """
    Evaluasi penuh SEKALI (jangan duplikasi komputasi — AC5 <5ms untuk 5 pasang).

    Returns: {"full_ratio","line_f1","score","matched"}
    """
    norm_ref = normalize_text(ref_text)
    norm_slot = normalize_text(slot_cleaned)

    # Fast path identik (kasus paling umum PASS) — hindari line matching penuh.
    if norm_ref and norm_ref == norm_slot:
        ref_n = len([ln for ln in (ref_text or "").upper().splitlines() if line_tokens(ln)])
        return {"full_ratio": 1.0, "line_f1": 1.0, "score": 1.0, "matched": ref_n}

    # ★ TOLERANSI TANDA BACA (2026-10-09): fast path kanonis — teks yang sama
    # modulo tanda baca/spasi ("HELLO WORLD." vs "HELLO WORLD", "PT." vs "PT")
    # langsung PASS 1.0 tanpa tergantung panjang teks. Titik/koma desimal yang
    # diapit digit ("1.5") tetap dibedakan bila KEEP_DECIMAL aktif (default).
    canon_on, keep_dec = _canonical_flags()
    canon_ref = canon_slot = ""
    if canon_on:
        canon_ref = normalize_canonical(ref_text, keep_dec)
        canon_slot = normalize_canonical(slot_cleaned, keep_dec)
        if canon_ref and canon_ref == canon_slot:
            ref_n = len([ln for ln in (ref_text or "").upper().splitlines() if line_tokens(ln)])
            return {"full_ratio": 1.0, "line_f1": 1.0, "score": 1.0, "matched": ref_n}

    # ★ IMPROVEMENT: Tambahkan sorted comparison sebagai second-chance
    # Jika normalize_text tidak identik, coba sorted (handle OCR line swap).
    # MIGRASI RapidFuzz: via _token_sort_hit (token_sort_ratio bila engine
    # rapidfuzz, perbandingan sorted manual bila fallback difflib — sama).
    if norm_ref and norm_slot:
        if _token_sort_hit(norm_ref, norm_slot):
            # Teks identik tapi urutan kata/baris berbeda (OCR reorder)
            ref_n = len([ln for ln in (ref_text or "").upper().splitlines() if line_tokens(ln)])
            return {"full_ratio": 1.0, "line_f1": 1.0, "score": 1.0, "matched": ref_n}
        # Second-chance kanonis (urutan kata beda + tanda baca beda).
        if canon_on and canon_ref and canon_slot:
            if _token_sort_hit(canon_ref, canon_slot):
                ref_n = len([ln for ln in (ref_text or "").upper().splitlines() if line_tokens(ln)])
                return {"full_ratio": 1.0, "line_f1": 1.0, "score": 1.0, "matched": ref_n}

    # full_ratio dihitung pada teks kanonis bila toleransi aktif, sehingga
    # noise tanda baca (titik trailing, koma, strip) tidak menurunkan skor.
    # MIGRASI RapidFuzz: via _ratio_0_1 (fuzz.ratio/100 bila rapidfuzz).
    if canon_on and (canon_ref or canon_slot):
        full_ratio = _ratio_0_1(canon_ref, canon_slot)
    else:
        full_ratio = _ratio_0_1(norm_ref, norm_slot) if (norm_ref or norm_slot) else 0.0

    ref_lines = [ln for ln in (ref_text or "").upper().splitlines() if ln.strip()]
    slot_lines_use = slot_lines or (slot_cleaned or "").splitlines()

    # ★ IMPROVEMENT: Gunakan line_tokens_v2 (strip ALL punctuation)
    ref_toks = [(line_tokens_v2(ln), ln) for ln in ref_lines if line_tokens_v2(ln)]
    slot_toks = [(line_tokens_v2(ln), ln) for ln in slot_lines_use if line_tokens_v2(ln)]

    matched_ref = 0
    used_slot_idx: set[int] = set()
    for rt_tok, rt_str in ref_toks:
        best, best_idx = 0.0, None
        for i, (st_tok, st_str) in enumerate(slot_toks):
            if i in used_slot_idx:
                continue
            j = _line_similarity(rt_tok, rt_str, st_tok, st_str)
            if j > best:
                best, best_idx = j, i
        if best >= _LINE_MATCH_MIN and best_idx is not None:
            matched_ref += 1
            used_slot_idx.add(best_idx)

    total_ref = len(ref_toks)
    total_slot = len(slot_toks)
    rec = matched_ref / total_ref if total_ref else 0.0
    prec = matched_ref / total_slot if total_slot else 0.0
    f1 = (2 * rec * prec / (rec + prec)) if (rec + prec) > 0 else 0.0

    # ★ ROBUST SEGMENTASI (2026-10-09): F1 per-baris runtuh bila pemenggalan
    # baris acuan vs check asimetris (mis. acuan 1 baris panjang vs OCR 25
    # baris pendek — lihat log ref_lines=1, slot_lines=25): tiap baris acuan
    # gagal dapat pasangan walau kata-katanya 99% sama. Fallback token-bag
    # global (multiset kata kanonis, abaikan pemenggalan baris) lalu ambil max.
    # Teks yang benar-benar beda tetap rendah di kedua metrik → tetap FAIL.
    try:
        if canon_on and (canon_ref or canon_slot):
            _rbag = Counter(canon_ref.split())
            _sbag = Counter(canon_slot.split())
        else:
            _rbag = Counter(norm_ref.split())
            _sbag = Counter(norm_slot.split())
        _inter = sum((_rbag & _sbag).values())
        _tot = sum(_rbag.values()) + sum(_sbag.values())
        _bag_f1 = (2 * _inter / _tot) if _tot else 0.0
        if _bag_f1 > f1:
            f1 = _bag_f1
    except Exception:
        pass

    score = round(_FULL_WEIGHT * full_ratio + _LINE_WEIGHT * f1, 3)
    return {"full_ratio": round(full_ratio, 3), "line_f1": round(f1, 3),
            "score": score, "matched": matched_ref}


def _is_punct_only_diff(a: str, b: str, keep_decimal: bool = True) -> bool:
    """True bila dua baris sama persis modulo tanda baca/spasi/case.

    "HELLO WORLD." vs "HELLO WORLD" -> True (diabaikan).
    "1.5 KG" vs "15 KG" (keep_decimal) -> False (tetap ditampilkan).
    """
    try:
        ca, cb = normalize_canonical(a, keep_decimal), normalize_canonical(b, keep_decimal)
        return bool(ca) and ca == cb
    except Exception:
        return False


def _content_gate_flags() -> tuple:
    """Baca konfigurasi content-gate (butuh restart bila diubah).

    SIMILARITY_CONTENT_GATE=false = nonaktifkan gate (kembali ke perilaku
    sebelum gate: verdict murni dari skor). SIMILARITY_MAX_ORPHAN_FRACTION =
    batas fraksi kata tak-terjelaskan (default 0.08). SIMILARITY_ORPHAN_SIM_MIN
    = batas kemiripan agar orphan dianggap "masih pasangan OCR-noise"
    (default 0.6).
    """
    try:
        from app.config import (  # type: ignore
            SIMILARITY_CONTENT_GATE as _gate,
            SIMILARITY_MAX_ORPHAN_FRACTION as _maxfrac,
            SIMILARITY_ORPHAN_SIM_MIN as _simmin,
        )
        return bool(_gate), float(_maxfrac), float(_simmin)
    except Exception:
        import os as _os
        return (
            (_os.getenv("SIMILARITY_CONTENT_GATE", "true").lower() == "true"),
            float(_os.getenv("SIMILARITY_MAX_ORPHAN_FRACTION", "0.08") or 0.08),
            float(_os.getenv("SIMILARITY_ORPHAN_SIM_MIN", "0.6") or 0.6),
        )


def _orphan_explained(orphan: str, others_full: list, orphans_other: set, sim_min: float) -> bool:
    """True bila satu kata yatim masih punya pasangan plausibel di sisi lain.

    Dijelaskan bila: (a) kemiripan karakter >= sim_min (typo/merge OCR seperti
    ONLY->ONLW, YDECORACION->DECORACION), atau (b) salah satunya substring dari
    yang lain dengan panjang >= 2 (gabung/pecah kata seperti WITHLIKE->LIKE,
    SOFTENERUSE->SOFTENER), atau (c) substring dari kata yatim sisi lain
    TANPA batas panjang (pasangan split/merge: Y + DECORACION vs YDECORACION).
    Single-char terhadap kata MATCH dikecualikan dari (b) agar designator
    ukuran (G, PP, XS) tetap dihitung.
    """
    try:
        for t in others_full:
            if not t or not orphan:
                continue
            if _ratio_0_1(orphan, t) >= sim_min:
                return True
            if min(len(orphan), len(t)) >= 2 and (orphan in t or t in orphan):
                return True
        for t in orphans_other:
            if t and (orphan in t or t in orphan):
                return True
        return False
    except Exception:
        return True  # gagal analisis = jangan jatuhkan verdict (fail-open aman)


def unexplained_orphan_fraction(ref_text: str, slot_cleaned: str) -> tuple:
    """Hitung fraksi kata tak-terjelaskan antara acuan vs check.

    Kata yatim (orphan) = kata kanonis yang hanya ada di satu sisi (multiset
    diff). Yatim "dijelaskan" bila masih mirip/terkandung oleh kata sisi lain
    (noise OCR: typo, gabung/pecah kata). Sisanya = tak-terjelaskan, indikasi
    beda konten sungguhan (beda ukuran/produk: G vs XS/PP, kata tambahan).

    CATATAN KEBIJAKAN: yatim yang mengandung digit (angka/desimal/kode)
    TIDAK dihitung di sini — angka mengikuti jalur lama (skor + flag
    numeric_diff + warning verifikasi manual), bukan gate FAIL. Alasan:
    salah-baca digit OCR vs salah-cetak sulit dibedakan otomatis; operator
    yang memutuskan via warning. Gate ini khusus beda kata non-angka.

    Returns:
        (fraction, unexplained_count, total_tokens) dengan fraction =
        unexplained / max(len(ref), len(slot)). TIDAK PERNAH raise.
    """
    try:
        _digit_re = re.compile(r"\d")
        _, keep_dec = _canonical_flags()
        ref_toks = normalize_canonical(ref_text, keep_dec).split()
        slot_toks = normalize_canonical(slot_cleaned, keep_dec).split()
        total = max(len(ref_toks), len(slot_toks))
        if total == 0:
            return 0.0, 0, 0
        _gate_on, _, sim_min = _content_gate_flags()
        rbag, sbag = Counter(ref_toks), Counter(slot_toks)
        orphans_r = list((rbag - sbag).elements())
        orphans_s = list((sbag - rbag).elements())
        set_r, set_s = set(orphans_r), set(orphans_s)
        unexplained = 0
        for o in orphans_r:
            if _digit_re.search(o):
                continue  # angka → jalur numeric_diff, bukan gate
            if not _orphan_explained(o, slot_toks, set_s, sim_min):
                unexplained += 1
        for o in orphans_s:
            if _digit_re.search(o):
                continue  # angka → jalur numeric_diff, bukan gate
            if not _orphan_explained(o, ref_toks, set_r, sim_min):
                unexplained += 1
        frac = (unexplained / total) if total else 0.0
        return min(frac, 1.0), unexplained, total
    except Exception:
        return 0.0, 0, 0


def _build_diff_details(
    ref_text: str,
    slot_text: str,
    max_diffs: int = 20,
) -> tuple:
    """
    Bangun daftar perbedaan baris antara reference dan check.

    Menggunakan SequenceMatcher untuk mendeteksi baris yang:
    - changed: ada di kedua sisi tapi berbeda content
    - missing: ada di reference tapi tidak di check
    - added:   ada di check tapi tidak di reference
    - matched: sama persis (tidak dimasukkan ke diff_details)

    ★ TOLERANSI TANDA BACA (2026-10-09): entri "changed" yang kanonisnya sama
    (beda hanya titik/koma/spasi) DIHILANGKAN agar operator tidak bising;
    jumlahnya dikembalikan sebagai ignored_count. missing/added tetap tampil.

    Args:
        ref_text: cleaned_text reference (acuan).
        slot_text: cleaned_text check.
        max_diffs: batas maksimum item diff (cegah payload terlalu besar).

    Returns:
        (diffs, ignored_count) — diffs = list of dict, masing-masing:
        {
            "type": "changed" | "missing" | "added",
            "ref_line_num": int | null,      # nomor baris di reference (1-indexed)
            "check_line_num": int | null,     # nomor baris di check (1-indexed)
            "ref_text": str | null,           # teks baris di reference
            "check_text": str | null,         # teks baris di check
        }

    TIDAK PERNAH raise — dibungkus try/except.
    """
    try:
        ref_lines = [ln.strip() for ln in (ref_text or "").upper().splitlines() if ln.strip()]
        slot_lines = [ln.strip() for ln in (slot_text or "").upper().splitlines() if ln.strip()]

        if not ref_lines and not slot_lines:
            return [], 0

        canon_on, keep_dec = _canonical_flags()

        def _keep_entry(dtype: str, r, s) -> bool:
            # Filter punct-only: hanya untuk "changed" dua sisi + flag aktif.
            if canon_on and dtype == "changed" and r and s:
                if _is_punct_only_diff(r, s, keep_dec):
                    return False
            return True

        matcher = SequenceMatcher(None, ref_lines, slot_lines)
        diffs = []
        ignored = 0

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if len(diffs) >= max_diffs:
                break

            if tag == "equal":
                # Baris sama — tidak masuk diff_details
                continue

            elif tag == "replace":
                # Baris ada di kedua sisi tapi berbeda
                ref_chunk = ref_lines[i1:i2]
                slot_chunk = slot_lines[j1:j2]
                # Pasangkan per baris (zip), sisa masuk missing/added
                for k in range(max(len(ref_chunk), len(slot_chunk))):
                    if len(diffs) >= max_diffs:
                        break
                    r = ref_chunk[k] if k < len(ref_chunk) else None
                    s = slot_chunk[k] if k < len(slot_chunk) else None
                    if r and s:
                        if not _keep_entry("changed", r, s):
                            ignored += 1
                            continue
                        diffs.append({
                            "type": "changed",
                            "ref_line_num": i1 + k + 1,
                            "check_line_num": j1 + k + 1,
                            "ref_text": r,
                            "check_text": s,
                        })
                    elif r and not s:
                        diffs.append({
                            "type": "missing",
                            "ref_line_num": i1 + k + 1,
                            "check_line_num": None,
                            "ref_text": r,
                            "check_text": None,
                        })
                    elif s and not r:
                        diffs.append({
                            "type": "added",
                            "ref_line_num": None,
                            "check_line_num": j1 + k + 1,
                            "ref_text": None,
                            "check_text": s,
                        })

            elif tag == "delete":
                # Baris ada di reference tapi tidak di check
                for k in range(i1, i2):
                    if len(diffs) >= max_diffs:
                        break
                    diffs.append({
                        "type": "missing",
                        "ref_line_num": k + 1,
                        "check_line_num": None,
                        "ref_text": ref_lines[k],
                        "check_text": None,
                    })

            elif tag == "insert":
                # Baris ada di check tapi tidak di reference
                for k in range(j1, j2):
                    if len(diffs) >= max_diffs:
                        break
                    diffs.append({
                        "type": "added",
                        "ref_line_num": None,
                        "check_line_num": k + 1,
                        "ref_text": None,
                        "check_text": slot_lines[k],
                    })

        return diffs, ignored

    except Exception:
        # Jangan pernah gagalkan compare karena diff_details gagal dihitung
        return [], 0


def score_pair(ref_text: str, slot_text: str) -> dict:
    """
    Skor pasangan teks → {"full_ratio","line_f1","score"} (0..1 semua).
    Tidak pernah raise.
    """
    try:
        ev = _evaluate(ref_text, slot_text, None)
        return {"full_ratio": ev["full_ratio"], "line_f1": ev["line_f1"], "score": ev["score"]}
    except Exception:
        return {"full_ratio": 0.0, "line_f1": 0.0, "score": 0.0}


def _is_no_text(cleaned: str) -> bool:
    t = (cleaned or "").strip().lower()
    return t in ("", "none")


def compare_slot(
    ref_text: str,
    slot_cleaned: str,
    slot_lines: Optional[list] = None,
    threshold: float = 0.85,
) -> dict:
    """
    Verdict 1 slot vs reference (plan §4.2). TIDAK PERNAH raise.

    Args:
        ref_text: cleaned_text reference (string).
        slot_cleaned: cleaned_text slot.
        slot_lines: list baris slot (dipakai untuk statistik baris).
        threshold: batas PASS.

    Returns:
        dict kontrak §4.1 + diff_details (list perbedaan baris).
    """
    try:
        lines = [str(ln) for ln in (slot_lines or []) if str(ln or "").strip()]
        # MIGRASI RapidFuzz: nama engine aktif dicatat aditif (Laravel sudah
        # menerima similarity.compare_engine tanpa ubah validasi).
        try:
            _engine_name = "rapidfuzz" if _use_rapidfuzz() else "difflib"
        except Exception:
            _engine_name = "difflib"
        base = {
            "version": VERSION,
            "threshold": round(float(threshold), 3),
            "ref_lines": len([ln for ln in (ref_text or "").upper().splitlines() if ln.strip()]),
            "slot_lines": len(lines),
            "matched_lines": 0,
            "compare_engine": _engine_name,
        }
        # Reference kosong / 'none' → SKIP semua slot.
        if _is_no_text(ref_text or ""):
            return {**base, "verdict": "SKIP", "score": None, "reason": "no_reference_text",
                    "diff_details": [], "punct_ignored_count": 0, "numeric_diff": False,
                    "unexplained_orphans": 0}

        # Slot kosong / gagal / 'none' → FAIL (reference jelas ada teks).
        if _is_no_text(slot_cleaned or "") and not lines:
            return {**base, "verdict": "FAIL", "score": 0.0, "reason": "empty_text",
                    "diff_details": [{
                        "type": "missing",
                        "ref_line_num": i + 1,
                        "check_line_num": None,
                        "ref_text": ln.strip(),
                        "check_text": None,
                    } for i, ln in enumerate((ref_text or "").upper().splitlines()) if ln.strip()],
                    "punct_ignored_count": 0, "numeric_diff": False,
                    "unexplained_orphans": 0}

        # Defensive truncate untuk teks sangat panjang (>10K char / >200 baris)
        # REVIEW_FLOW_COMPARE_FRONT_BACK_PLAN (2026-10-05) §5.3 Optimasi 2.
        _ref_safe = (ref_text or "")[:MAX_COMPARE_CHARS]
        _slot_safe = (slot_cleaned or "")[:MAX_COMPARE_CHARS]
        _lines_safe = lines[:MAX_COMPARE_LINES] if lines else None
        ev = _evaluate(_ref_safe, _slot_safe, _lines_safe)
        score = ev["score"]
        verdict = "PASS" if score >= float(threshold) else "FAIL"

        # ★ BARU: Bangun diff_details (punct-only difilter di dalam).
        diff_details, punct_ignored = _build_diff_details(_ref_safe, _slot_safe)

        # ★ Flag QA (2026-10-09): True bila ada perbedaan tersisa yang menyentuh
        # digit angka ("1.5" vs "15", "805.011" vs "805011"). Dipakai Laravel/UI
        # untuk peringatan "verifikasi manual" walau verdict PASS.
        numeric_diff = False
        try:
            _digit_re = re.compile(r"\d")
            for _dd in diff_details:
                if _digit_re.search(str((_dd or {}).get("ref_text") or "")) or \
                   _digit_re.search(str((_dd or {}).get("check_text") or "")):
                    numeric_diff = True
                    break
        except Exception:
            numeric_diff = False

        # ★ CONTENT-GATE (2026-10-09): skor tinggi belum tentu konten sama.
        # Kasus nyata: beda label/ukuran (G/GRANDE vs XS/PP + kata tambahan)
        # tetap skor ~0.95 karena 90% kata sama → PASS palsu. Gate ini
        # menghitung kata yatim TAK-TERJELASKAN (beda konten sungguhan, bukan
        # typo/merge OCR). Dijalankan SELALU agar unexplained_orphans informatif
        # di semua verdict; flip PASS→FAIL + reason content_mismatch hanya bila
        # skor lolos tapi kata tak-terjelaskan melebihi batas. Skor
        # dipertahankan apa adanya (transparan di log/UI).
        reason = None
        unexplained_orphans = 0
        try:
            _gate_on, _max_frac, _ = _content_gate_flags()
            if _gate_on:
                _frac, _count, _ = unexplained_orphan_fraction(_ref_safe, _slot_safe)
                unexplained_orphans = int(_count or 0)
                if verdict == "PASS" and _frac > float(_max_frac):
                    verdict = "FAIL"
                    reason = "content_mismatch"
        except Exception:
            pass

        return {
            **base,
            "verdict": verdict,
            "score": score,
            "matched_lines": ev["matched"],
            "reason": reason,
            "diff_details": diff_details,
            "punct_ignored_count": int(punct_ignored or 0),
            "numeric_diff": bool(numeric_diff),
            "unexplained_orphans": unexplained_orphans,
        }
    except Exception as e:
        return {
            "version": VERSION,
            "verdict": "SKIP",
            "score": None,
            "threshold": round(float(threshold), 3),
            "ref_lines": 0,
            "slot_lines": 0,
            "matched_lines": 0,
            "reason": f"exception:{type(e).__name__}"[:120],
            "diff_details": [],
            "punct_ignored_count": 0,
            "numeric_diff": False,
            "unexplained_orphans": 0,
            "compare_engine": "difflib",
        }


def compare_batch(ref_text: str, slot_results: list, threshold: float = 0.85) -> dict:
    """
    Ringkasan batch (plan §4.3). slot_results = list result slot (dict dengan
    kunci 'slot' + 'similarity'). TIDAK PERNAH raise.
    """
    try:
        decisions: list = []
        pass_n = fail_n = skip_n = 0
        for r in slot_results or []:
            sim = (r or {}).get("similarity") or {}
            verdict = sim.get("verdict")
            decisions.append({
                "slot": (r or {}).get("slot"),
                "verdict": verdict,
                "score": sim.get("score"),
            })
            if verdict == "PASS":
                pass_n += 1
            elif verdict == "FAIL":
                fail_n += 1
            else:
                skip_n += 1
        overall = "SKIPPED"
        if pass_n and not fail_n:
            overall = "PASS_ALL"
        elif fail_n and not pass_n:
            overall = "FAIL_ALL"
        elif pass_n or fail_n:
            overall = "MIXED"
        return {
            "version": VERSION,
            "threshold": round(float(threshold), 3),
            "pass_count": pass_n,
            "fail_count": fail_n,
            "skip_count": skip_n,
            "decisions": decisions,
            "overall": overall,
        }
    except Exception as e:
        return {
            "version": VERSION,
            "threshold": round(float(threshold), 3),
            "pass_count": 0,
            "fail_count": 0,
            "skip_count": 0,
            "decisions": [],
            "overall": "SKIPPED",
            "reason": f"exception:{type(e).__name__}"[:120],
        }


def compare_batch_by_side(slot_results: list, threshold: float = 0.85) -> dict:
    """
    Ringkasan batch DENGAN breakdown per sisi (front/back).

    REVIEW_FLOW_COMPARE_FRONT_BACK_PLAN (2026-10-05) §6.3 TAHAP 3.

    slot_results = list dict, setiap item wajib punya:
      - "slot": int
      - "side": "front" | "back" | "unknown"
      - "similarity": {verdict, score, ...}

    Returns:
      {
        "version": "v2",
        "threshold": 0.85,
        "front": { "pass_count": N, "fail_count": N, "skip_count": N, "overall": "..." },
        "back":  { "pass_count": N, "fail_count": N, "skip_count": N, "overall": "..." },
        "combined_pass_count": N,
        "combined_fail_count": N,
        "combined_skip_count": N,
        "combined_overall": "PASS_ALL" | "FAIL_ALL" | "MIXED" | "SKIPPED",
        "decisions": [{ "slot": 1, "side": "front", "verdict": "PASS", "score": 0.97 }, ...]
      }

    TIDAK PERNAH raise.
    """
    try:
        decisions = []
        sides = {"front": {"pass": 0, "fail": 0, "skip": 0},
                 "back":  {"pass": 0, "fail": 0, "skip": 0},
                 "unknown": {"pass": 0, "fail": 0, "skip": 0}}

        for r in (slot_results or []):
            sim = (r or {}).get("similarity") or {}
            side = (r or {}).get("side", "unknown")
            verdict = sim.get("verdict")
            decisions.append({
                "slot": (r or {}).get("slot"),
                "side": side,
                "verdict": verdict,
                "score": sim.get("score"),
            })
            bucket = sides.get(side, sides["unknown"])
            if verdict == "PASS":
                bucket["pass"] += 1
            elif verdict == "FAIL":
                bucket["fail"] += 1
            else:
                bucket["skip"] += 1

        def _overall(d):
            p, f = d["pass"], d["fail"]
            if p and not f:
                return "PASS_ALL"
            if f and not p:
                return "FAIL_ALL"
            if p or f:
                return "MIXED"
            return "SKIPPED"

        total_pass = sum(d["pass"] for d in sides.values())
        total_fail = sum(d["fail"] for d in sides.values())
        total_skip = sum(d["skip"] for d in sides.values())

        return {
            "version": "v2",
            "threshold": round(float(threshold), 3),
            "front": {
                "pass_count": sides["front"]["pass"],
                "fail_count": sides["front"]["fail"],
                "skip_count": sides["front"]["skip"],
                "overall": _overall(sides["front"]),
            },
            "back": {
                "pass_count": sides["back"]["pass"],
                "fail_count": sides["back"]["fail"],
                "skip_count": sides["back"]["skip"],
                "overall": _overall(sides["back"]),
            },
            "combined_pass_count": total_pass,
            "combined_fail_count": total_fail,
            "combined_skip_count": total_skip,
            "combined_overall": (
                "PASS_ALL" if total_pass and not total_fail else
                "FAIL_ALL" if total_fail and not total_pass else
                "MIXED" if (total_pass or total_fail) else "SKIPPED"
            ),
            "decisions": decisions,
        }
    except Exception as e:
        return {
            "version": "v2",
            "threshold": round(float(threshold), 3),
            "front": {"pass_count": 0, "fail_count": 0, "skip_count": 0, "overall": "SKIPPED"},
            "back": {"pass_count": 0, "fail_count": 0, "skip_count": 0, "overall": "SKIPPED"},
            "combined_pass_count": 0, "combined_fail_count": 0, "combined_skip_count": 0,
            "combined_overall": "SKIPPED",
            "decisions": [],
            "reason": f"exception:{type(e).__name__}"[:120],
        }
