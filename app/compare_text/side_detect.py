"""
app/compare_text/side_detect.py — Deteksi sisi label dari nama file.

Dipakai _process_batch_ocr untuk mencocokkan slot check dengan
reference_text yang benar (front vs back).
"""

import re
from typing import Literal

SideType = Literal["front", "back", "unknown"]

# Pattern: "-front" atau "-front-N" sebelum ekstensi
_FRONT_RE = re.compile(r'-front(-\d+)?\.[a-z]+$', re.IGNORECASE)
_BACK_RE = re.compile(r'-back(-\d+)?\.[a-z]+$', re.IGNORECASE)


def detect_side(filename: str) -> SideType:
    """
    Deteksi sisi label dari nama file.

    Returns: "front" | "back" | "unknown"

    Contoh:
      "slot1-label-ref-m-front.png"   -> "front"
      "slot2-label-ref-m-back.png"    -> "back"
      "slot3-label-ref-m-front-1.png" -> "front"
      "slot4-label-ref-m-back-2.png"  -> "back"
      "slot1-CHK-20261005.png"        -> "unknown"
    """
    name = (filename or "").strip()
    if not name:
        return "unknown"
    if _FRONT_RE.search(name):
        return "front"
    if _BACK_RE.search(name):
        return "back"
    return "unknown"
