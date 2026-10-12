"""
app/ocr/parser.py — Parsing & cleaning teks hasil OCR.

Catatan: teks disimpan MENTAH (raw). Cleaning hanya meliputi
pembersihan whitespace dan formatting, BUKAN parsing struktur/template.
Tidak ada regex matching untuk field tertentu (Batch, Exp, dll) sesuai keputusan.
"""

from typing import TypedDict
import re


_WS_MULTI = re.compile(r"\s+")


class CleanResult(TypedDict):
    raw_text: str
    cleaned_text: str
    lines: list[str]


def clean_ocr_text(raw_text: str) -> CleanResult:
    """
    Bersihkan teks OCR mentah.

    Langkah:
    1. Strip whitespace keseluruhan.
    2. Pecah per baris.
    3. Strip + collapse whitespace setiap baris.
    4. Hapus baris kosong.
    5. Gabung dengan SATU spasi (single-line, tanpa newline).

    Args:
        raw_text: Teks mentah langsung dari Tesseract (bisa berisi \n berlebih/whitespace).

    Returns:
        dict dengan keys: raw_text, cleaned_text, lines.
        - raw_text: teks mentah setelah strip keseluruhan (pertahankan newline asli)
        - cleaned_text: single-line — baris bersih digabung 1 spasi (TIDAK mengandung "\n")
        - lines: list baris yang sudah bersih (tetap ada untuk kompatibilitas)

    Example:
        >>> clean_ocr_text("  hello\n\n  world  \n")
        {'raw_text': 'hello\n\n  world', 'cleaned_text': 'hello world', 'lines': ['hello', 'world']}
    """
    if raw_text is None:
        raw_text = ""

    # Pastikan string
    if not isinstance(raw_text, str):
        raw_text = str(raw_text)

    # Step 1: Strip whitespace keseluruhan
    stripped = raw_text.strip()

    # Step 2-4: Pecah per baris, collapse whitespace dalam baris, hapus baris kosong.
    # SINGLE-LINE FORMAT (2026-10-07): cleaned_text tanpa "\n" — gabung 1 spasi.
    lines = stripped.splitlines()
    cleaned_lines = [
        _WS_MULTI.sub(" ", line.strip())
        for line in lines
        if line.strip()
    ]

    return {
        "raw_text": stripped,                   # tetap multiline (jejak audit mentah)
        "cleaned_text": " ".join(cleaned_lines),  # ★ single-line, tanpa "\n"
        "lines": cleaned_lines,                 # tetap ada (kompatibilitas payload/DB)
    }


# ---------------------------------------------------------------------------
# OCR Engine Upgrade (2026-09-11): kontrak 'none' untuk RapidOCR/Paddle.
# 'none' (case-insensitive) = hasil VALID: tidak ada teks terdeteksi.
# BUKAN error, BUKAN kosong. Lihat plans/OCR_ENGINE_UPGRADE_PADDLE_PLAN.md §6.
# ---------------------------------------------------------------------------
NONE_TEXT = "none"


def is_none_text(cleaned_text: str | None) -> bool:
    """True jika teks adalah marker 'none' (case-insensitive, strip)."""
    return isinstance(cleaned_text, str) and cleaned_text.strip().lower() == NONE_TEXT


def build_paddle_result(lines_raw: list[dict], det_thr: float, rec_thr: float) -> dict:
    """
    Filter skor deteksi/recognisi + terapkan kontrak 'none'.

    Args:
        lines_raw: list {"text": str, "det_score": float|None, "rec_score": float|None}
                   (output app.ocr.paddle_engine.extract_text_paddle).
        det_thr: threshold skor deteksi (None = lolos).
        rec_thr: threshold skor recognisi.

    Returns:
        dict {"raw_text","cleaned_text","lines","text_present","empty_reason"}.
        - Ada baris valid → gabungan baris, text_present True.
        - Tidak ada → cleaned_text "none", lines [], text_present False,
          empty_reason "no_text_detected" (raw_text "" — bukan "none").
    """
    kept: list[str] = []
    for it in lines_raw or []:
        t = (it.get("text") or "").strip() if isinstance(it, dict) else str(it or "").strip()
        if not t:
            continue
        ds = it.get("det_score") if isinstance(it, dict) else None
        rs = it.get("rec_score") if isinstance(it, dict) else None
        if ds is not None and ds < det_thr:
            continue
        if rs is not None and rs < rec_thr:
            continue
        kept.append(_WS_MULTI.sub(" ", t))
    if not kept:
        return {
            "raw_text": "",
            "cleaned_text": NONE_TEXT,
            "lines": [],
            "text_present": False,
            "empty_reason": "no_text_detected",
        }
    return {
        "raw_text": "\n".join(kept),      # tetap multiline (jejak audit mentah)
        "cleaned_text": " ".join(kept),   # ★ single-line, tanpa "\n"
        "lines": kept,                    # tetap ada (kompatibilitas payload/DB)
        "text_present": True,
        "empty_reason": None,
    }
