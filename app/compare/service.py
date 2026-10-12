"""
app/compare/service.py — Business logic untuk Visual Comparison.

Fungsi utama: compare_images_with_reference()
- Load gambar reference + check dari disk
- Kirim ke Gemini via gemini_client
- Parse dan enrichment hasil
- Simpan ke disk (JSON)
- Return ke caller

Digunakan oleh:
1. Endpoint API (routers atau main.py)
2. Script testing standalone (test_compare.py)
"""

import json
import logging
import time
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

from app.compare.gemini_client import create_gemini_client, call_gemini_comparison
from app.compare.prompt import build_comparison_prompt

logger = logging.getLogger(__name__)

# Timezone Jakarta
JAKARTA_TZ = timezone(timedelta(hours=7))


def compare_images_with_reference(
    reference_image_path: str,
    check_image_paths: list[str],
    api_key: str,
    model: str = "gemini-3.8-flash",
    timeout: float = 30.0,
    batch_code: Optional[str] = None,
    save_result: bool = True,
    save_dir: Optional[str] = None,
) -> dict:
    """
    Fungsi utama: Bandingkan gambar reference dengan gambar check menggunakan Gemini.

    Args:
        reference_image_path: Path absolut ke gambar acuan (reference).
        check_image_paths: List path absolut ke gambar-gambar check (slot 1..N).
        api_key: Gemini API key.
        model: Model Gemini yang digunakan (default: gemini-3.8-flash).
        timeout: Timeout per request dalam detik.
        batch_code: Opsional batch code (untuk metadata di hasil).
        save_result: Jika True, simpan hasil ke JSON file.
        save_dir: Folder untuk menyimpan compare_result.json.
                  Default: folder yang sama dengan check images.

    Returns:
        dict: Hasil comparison lengkap.

    Raises:
        FileNotFoundError: Jika reference atau check image tidak ditemukan.
        ValueError: Jika API key kosong atau response tidak valid.
    """
    start = time.time()
    ref_basename = os.path.basename(reference_image_path)
    check_basenames = [os.path.basename(p) for p in check_image_paths]

    print(f"[COMPARE] Starting visual comparison")
    print(f"[COMPARE] Reference: {ref_basename}")
    print(f"[COMPARE] Check images: {len(check_image_paths)} files")
    for i, cb in enumerate(check_basenames):
        print(f"[COMPARE]   Slot {i+1}: {cb}")

    try:
        # --- Validasi file ada ---
        if not os.path.exists(reference_image_path):
            raise FileNotFoundError(
                f"Gambar reference tidak ditemukan: {reference_image_path}"
            )

        for cp in check_image_paths:
            if not os.path.exists(cp):
                raise FileNotFoundError(f"Gambar check tidak ditemukan: {cp}")

        # --- Buat client ---
        client = create_gemini_client(api_key)

        # --- Bangun prompt ---
        prompt = build_comparison_prompt(check_basenames)

        # --- Kirim ke Gemini ---
        gemini_result = call_gemini_comparison(
            client=client,
            model=model,
            prompt=prompt,
            reference_image_path=reference_image_path,
            check_image_paths=check_image_paths,
            timeout=timeout,
        )

        elapsed = time.time() - start

        # --- Enrichment: tambahkan metadata yang tidak dari Gemini ---
        results_list = gemini_result.get("results", [])

        # Pastikan setiap result punya slot number yang benar
        for i, res in enumerate(results_list):
            if "slot" not in res:
                res["slot"] = i + 1
            if "file_name" not in res and i < len(check_basenames):
                res["file_name"] = check_basenames[i]
            if "confidence_score" in res:
                res["confidence_score"] = round(float(res["confidence_score"]), 2)

        # --- Bangun hasil lengkap ---
        full_result = {
            "reference_image": ref_basename,
            "batch_code": batch_code,
            "total_checked": len(check_image_paths),
            "model": model,
            "compared_at": datetime.now(JAKARTA_TZ).isoformat(),
            "processing_time_seconds": round(elapsed, 3),
            "status": "success",
            "error": None,
            "results": results_list,
        }

        print(f"[COMPARE] Success: {len(results_list)} results in {elapsed:.2f}s")
        for res in results_list:
            match_str = "MATCH" if res.get("is_match") else "MISMATCH"
            print(
                f"[COMPARE]   Slot {res.get('slot', '?')}: {match_str} "
                f"(confidence: {res.get('confidence_score', '?')}) "
                f"— {res.get('notes', '')[:80]}"
            )

        # --- Simpan ke disk jika diminta ---
        if save_result:
            if not save_dir and check_image_paths:
                save_dir = str(Path(check_image_paths[0]).parent)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                result_path = os.path.join(save_dir, "compare_result.json")
                with open(result_path, "w", encoding="utf-8") as f:
                    json.dump(full_result, f, indent=2, ensure_ascii=False)
                print(f"[COMPARE] Result saved to: {result_path}")

        return full_result

    except Exception as e:
        elapsed = time.time() - start
        logger.error(f"[COMPARE] Failed: {e}", exc_info=True)
        print(f"[COMPARE] Failed: {e}")

        error_result = {
            "reference_image": ref_basename,
            "batch_code": batch_code,
            "total_checked": len(check_image_paths),
            "model": model,
            "compared_at": datetime.now(JAKARTA_TZ).isoformat(),
            "processing_time_seconds": round(elapsed, 3),
            "status": "failed",
            "error": str(e),
            "results": [],
        }

        # Simpan error result juga (agar bisa di-audit)
        if save_result and save_dir:
            try:
                os.makedirs(save_dir, exist_ok=True)
                result_path = os.path.join(save_dir, "compare_result.json")
                with open(result_path, "w", encoding="utf-8") as f:
                    json.dump(error_result, f, indent=2, ensure_ascii=False)
            except Exception as save_err:
                logger.warning(f"[COMPARE] Failed to save error result: {save_err}")

        return error_result
