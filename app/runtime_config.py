"""
app/runtime_config.py — Manajer Konfigurasi AI & Multi-Model Dinamis (Runtime).

Memungkinkan pergantian profil OCR Engine dan Compare Similarity Engine
secara real-time dari Dashboard tanpa perlu restart .env atau service FastAPI.

State disimpan di memori (thread-safe RLock) dan di-persist ke
uploads/runtime_settings.json agar bertahan saat server di-restart.
"""

import json
import logging
import os
import threading
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

from app.config import (
    UPLOAD_DIR,
    OCR_ENGINE,
    RAPIDOCR_REC_MODEL,
    OCR_CASCADE_ENABLED,
    SIMILARITY_ENGINE,
    GEMINI_API_KEY,
)

logger = logging.getLogger(__name__)

JAKARTA_TZ = timezone(timedelta(hours=7))
RUNTIME_SETTINGS_FILE = UPLOAD_DIR / "runtime_settings.json"

_lock = threading.RLock()

# Profil OCR yang dikenali
VALID_OCR_PROFILES = {
    "rapidocr_cascade": {
        "name": "RapidOCR Hybrid Cascade",
        "description": "Model small super cepat dengan auto-eskalasi ke medium jika confidence < 0.88 atau baris kritis.",
        "engine": "paddle",
        "rec_model": "small",
        "cascade": True,
        "speed": "Cepat (~4-6 detik / 4 slot)",
        "recommended": True,
    },
    "rapidocr_small": {
        "name": "RapidOCR Fast (Small)",
        "description": "Model small murni tanpa cascade. Paling ringan untuk CPU standar.",
        "engine": "paddle",
        "rec_model": "small",
        "cascade": False,
        "speed": "Sangat Cepat (~3-4 detik / 4 slot)",
        "recommended": False,
    },
    "rapidocr_medium": {
        "name": "RapidOCR High-Accuracy (Medium)",
        "description": "Model PP-OCRv6 medium pada semua baris. Akurasi maksimal untuk karakter ambigu dan 50 bahasa.",
        "engine": "paddle",
        "rec_model": "medium",
        "cascade": False,
        "speed": "Sedang (~12-16 detik / 4 slot)",
        "recommended": False,
    },
    "surya": {
        "name": "Surya OCR (Multilingual Transformer)",
        "description": "Arsitektur SegFormer + Transformer untuk dokumen ekspor dan 90+ bahasa internasional.",
        "engine": "surya",
        "rec_model": "surya",
        "cascade": False,
        "speed": "Lambat (~15-25 detik / 4 slot)",
        "recommended": False,
    },
    "vitstr": {
        "name": "ViTSTR-Tiny (Vision Transformer)",
        "description": "Vision Transformer Scene Text Recognition ultra-cepat untuk teks crop baris/kata tunggal.",
        "engine": "vitstr",
        "rec_model": "vitstr_tiny",
        "cascade": False,
        "speed": "Ultra Cepat (~1-2 detik / crop)",
        "recommended": False,
    },
    "tesseract": {
        "name": "Tesseract OCR (2-Pass CLAHE)",
        "description": "Engine Tesseract klasik dengan preprocessing bertingkat sebagai fallback kompatibilitas.",
        "engine": "tesseract",
        "rec_model": "tesseract",
        "cascade": False,
        "speed": "Sedang (~8-15 detik / 4 slot)",
        "recommended": False,
    },
}

# Profil Similarity yang dikenali
VALID_COMPARE_PROFILES = {
    "rapidfuzz": {
        "name": "RapidFuzz C++ (Levenshtein + Token Sort)",
        "description": "Pencocokan deterministik ultra cepat (<5ms), offline penuh, dengan content-gate.",
        "engine": "rapidfuzz",
        "recommended": True,
    },
    "difflib": {
        "name": "Python Difflib (SequenceMatcher)",
        "description": "Algoritma pencocokan urutan karakter standar library Python.",
        "engine": "difflib",
        "recommended": False,
    },
    "gemini_flash": {
        "name": "Gemini Flash AI (Visual & Semantic Diff)",
        "description": "Audit multimodal visual & semantik untuk mendeteksi salah cetak dan perbedaan regulasi kritis.",
        "engine": "gemini_flash",
        "recommended": False,
    },
}


def _determine_default_ocr_profile() -> str:
    """Tentukan profil default awal berdasarkan konfigurasi .env."""
    eng = (OCR_ENGINE or "paddle").strip().lower()
    if eng == "paddle":
        if bool(OCR_CASCADE_ENABLED):
            return "rapidocr_cascade"
        rec = (RAPIDOCR_REC_MODEL or "small").strip().lower()
        if rec == "medium":
            return "rapidocr_medium"
        return "rapidocr_small"
    if eng in VALID_OCR_PROFILES:
        return eng
    return "rapidocr_cascade"


def _determine_default_compare_profile() -> str:
    """Tentukan profil compare default awal berdasarkan konfigurasi .env."""
    sim = (SIMILARITY_ENGINE or "rapidfuzz").strip().lower()
    if sim in VALID_COMPARE_PROFILES:
        return sim
    return "rapidfuzz"


class RuntimeConfigManager:
    """Singleton manager konfigurasi runtime thread-safe."""

    def __init__(self):
        self._current_config: Dict[str, Any] = {
            "ocr_profile": _determine_default_ocr_profile(),
            "compare_profile": _determine_default_compare_profile(),
            "updated_at": datetime.now(JAKARTA_TZ).isoformat(),
            "updated_by": "system_env",
        }
        self._load_from_disk()

    def _load_from_disk(self):
        """Muat konfigurasi tersimpan dari uploads/runtime_settings.json jika ada."""
        try:
            if RUNTIME_SETTINGS_FILE.exists():
                with open(RUNTIME_SETTINGS_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    ocr_p = data.get("ocr_profile")
                    comp_p = data.get("compare_profile")
                    if ocr_p in VALID_OCR_PROFILES:
                        self._current_config["ocr_profile"] = ocr_p
                    if comp_p in VALID_COMPARE_PROFILES:
                        self._current_config["compare_profile"] = comp_p
                    if data.get("updated_at"):
                        self._current_config["updated_at"] = data.get("updated_at")
                    if data.get("updated_by"):
                        self._current_config["updated_by"] = data.get("updated_by")
                    logger.info(
                        f"[RUNTIME CONFIG] Loaded from disk: ocr={self._current_config['ocr_profile']}, "
                        f"compare={self._current_config['compare_profile']}"
                    )
        except Exception as e:
            logger.warning(f"[RUNTIME CONFIG] Failed to load disk settings: {e}")

    def _save_to_disk(self):
        """Simpan konfigurasi ke disk secara atomic."""
        try:
            tmp = RUNTIME_SETTINGS_FILE.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._current_config, f, indent=2, ensure_ascii=False)
            tmp.replace(RUNTIME_SETTINGS_FILE)
        except Exception as e:
            logger.error(f"[RUNTIME CONFIG] Failed to save settings to disk: {e}")

    def get_config(self) -> Dict[str, Any]:
        """Ambil salinan konfigurasi aktif saat ini."""
        with _lock:
            prof_info = VALID_OCR_PROFILES.get(self._current_config["ocr_profile"], VALID_OCR_PROFILES["rapidocr_cascade"])
            comp_info = VALID_COMPARE_PROFILES.get(self._current_config["compare_profile"], VALID_COMPARE_PROFILES["rapidfuzz"])

            return {
                "ocr_profile": self._current_config["ocr_profile"],
                "ocr_engine": prof_info["engine"],
                "rapidocr_rec_model": prof_info["rec_model"],
                "ocr_cascade_enabled": prof_info["cascade"],
                "compare_profile": self._current_config["compare_profile"],
                "compare_engine": comp_info["engine"],
                "updated_at": self._current_config.get("updated_at"),
                "updated_by": self._current_config.get("updated_by", "system"),
            }

    def set_config(self, ocr_profile: Optional[str] = None, compare_profile: Optional[str] = None, updated_by: str = "operator") -> Dict[str, Any]:
        """Ubah konfigurasi saat runtime."""
        with _lock:
            if ocr_profile:
                if ocr_profile not in VALID_OCR_PROFILES:
                    raise ValueError(f"Profil OCR tidak valid: {ocr_profile}. Pilihan: {list(VALID_OCR_PROFILES.keys())}")
                self._current_config["ocr_profile"] = ocr_profile

            if compare_profile:
                if compare_profile not in VALID_COMPARE_PROFILES:
                    raise ValueError(f"Profil Compare tidak valid: {compare_profile}. Pilihan: {list(VALID_COMPARE_PROFILES.keys())}")
                self._current_config["compare_profile"] = compare_profile

            self._current_config["updated_at"] = datetime.now(JAKARTA_TZ).isoformat()
            self._current_config["updated_by"] = updated_by
            self._save_to_disk()
            logger.info(f"[RUNTIME CONFIG] Updated by {updated_by}: {self._current_config}")
            return self.get_config()

    def check_capabilities(self) -> Dict[str, Any]:
        """Periksa kesiapan dependensi masing-masing profil model."""
        # 1. RapidOCR & ONNX
        try:
            from app.ocr.paddle_engine import is_paddle_available
            rapidocr_avail = is_paddle_available()
        except Exception:
            rapidocr_avail = False

        # 2. Surya OCR
        try:
            from app.ocr.surya_engine import is_surya_available
            surya_avail = is_surya_available()
        except Exception:
            surya_avail = False

        # 3. ViTSTR & PyTorch
        try:
            from app.ocr.vitstr_engine import is_vitstr_available
            vitstr_avail = is_vitstr_available()
        except Exception:
            vitstr_avail = False

        # 4. Tesseract
        try:
            from app.ocr.engine import is_tesseract_available
            tess_avail = is_tesseract_available()
        except Exception:
            tess_avail = False

        # 5. Gemini Compare
        has_gemini_key = bool(GEMINI_API_KEY and GEMINI_API_KEY.strip() and GEMINI_API_KEY != "YOUR_GEMINI_API_KEY_HERE")
        try:
            import google.genai  # noqa: F401
            gemini_lib = True
        except Exception:
            gemini_lib = False
        gemini_avail = gemini_lib and has_gemini_key

        ocr_items = []
        for key, p in VALID_OCR_PROFILES.items():
            installed = False
            reason = None
            if p["engine"] == "paddle":
                installed = rapidocr_avail
                if not installed:
                    reason = "Paket rapidocr/onnxruntime belum terpasang."
            elif p["engine"] == "surya":
                installed = surya_avail
                if not installed:
                    reason = "Paket surya-ocr belum terpasang di Python backend."
            elif p["engine"] == "vitstr":
                installed = vitstr_avail
                if not installed:
                    reason = "Paket torch/torchvision belum terpasang di Python backend."
            elif p["engine"] == "tesseract":
                installed = tess_avail
                if not installed:
                    reason = "Binary Tesseract-OCR belum ditemukan di sistem."

            ocr_items.append({
                "profile": key,
                "name": p["name"],
                "description": p["description"],
                "engine": p["engine"],
                "speed": p["speed"],
                "recommended": p["recommended"],
                "is_active": (key == self._current_config["ocr_profile"]),
                "is_installed": installed,
                "unavailable_reason": reason,
            })

        compare_items = []
        for key, p in VALID_COMPARE_PROFILES.items():
            installed = True
            reason = None
            if key == "gemini_flash":
                installed = gemini_avail
                if not gemini_lib:
                    reason = "Paket google-genai belum terpasang."
                elif not has_gemini_key:
                    reason = "GEMINI_API_KEY belum diisi di backend .env"

            compare_items.append({
                "profile": key,
                "name": p["name"],
                "description": p["description"],
                "engine": p["engine"],
                "recommended": p["recommended"],
                "is_active": (key == self._current_config["compare_profile"]),
                "is_installed": installed,
                "unavailable_reason": reason,
            })

        return {
            "active_config": self.get_config(),
            "ocr_profiles": ocr_items,
            "compare_profiles": compare_items,
        }


# Global singleton instance
runtime_config = RuntimeConfigManager()
