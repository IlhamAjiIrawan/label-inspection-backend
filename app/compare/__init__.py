"""
app/compare/ — Module untuk Visual Comparison menggunakan Gemini 1.5 Flash.

Membandingkan 1 gambar reference dengan N gambar check secara visual.
Output: JSON {is_match, confidence_score, notes} per gambar.
"""

from app.compare.service import compare_images_with_reference
from app.compare.callback import send_compare_callback

__all__ = ["compare_images_with_reference", "send_compare_callback"]
