"""
app/storage.py — Helper penyimpanan file dan metadata.

Menangani: load/save metadata.json, helper file terbaru (fallback),
dan utilitas path aman.
"""

import json
import glob
import os
from pathlib import Path
from typing import Optional

from app.config import UPLOAD_DIR, METADATA_FILE


def load_metadata() -> dict:
    """Baca semua metadata dari metadata.json. Return {} jika belum ada."""
    if METADATA_FILE.exists():
        try:
            with open(METADATA_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Pastikan return dict (bukan list/null jika file corrupt)
                if isinstance(data, dict):
                    return data
                return {}
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


_metadata_lock = __import__("threading").Lock()


def save_metadata(data: dict) -> None:
    """Tulis metadata.json atomik (tmp + os.replace) + lock (R9)."""
    try:
        with _metadata_lock:
            tmp = METADATA_FILE.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            try:
                import os as _os

                _os.replace(tmp, METADATA_FILE)
            except Exception:
                # Fallback bila replace gagal (mis. lock Windows sesaat).
                import shutil as _sh

                _sh.move(str(tmp), str(METADATA_FILE))
    except OSError as e:
        # Jangan crash service; log ke stdout
        print(f"[STORAGE ERROR] Failed to save metadata: {e}")


def get_latest_image(upload_dir: Optional[Path] = None) -> Optional[str]:
    """
    Fallback: cari file gambar terbaru berdasarkan modification time (mtime).

    Digunakan untuk endpoint manual POST /api/test/ocr/run tanpa upload baru.
    Untuk kasus POST upload, gunakan filename dari request langsung (lebih akurat).

    Args:
        upload_dir: Folder upload (default UPLOAD_DIR dari config).

    Returns:
        Path absolut string ke file gambar terbaru, atau None jika tidak ada.
    """
    base = upload_dir or UPLOAD_DIR
    image_files: list[str] = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"):
        image_files.extend(glob.glob(str(base / ext)))

    if not image_files:
        return None

    # Filter: hanya file yang masih ada & bukan folder, dan exclude folder ocr
    valid_files = []
    for p in image_files:
        # Abaikan file di dalam subfolder ocr
        if os.path.sep + "ocr" + os.path.sep in p:
            continue
        if os.path.isfile(p):
            valid_files.append(p)

    if not valid_files:
        return None

    # Urutkan berdasarkan mtime terbaru
    valid_files.sort(key=os.path.getmtime, reverse=True)
    return valid_files[0]


def get_all_image_files(upload_dir: Optional[Path] = None) -> list[str]:
    """Return semua file gambar di upload_dir (exclude subfolder ocr)."""
    base = upload_dir or UPLOAD_DIR
    image_files: list[str] = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"):
        image_files.extend(glob.glob(str(base / ext)))

    result = []
    for p in image_files:
        if os.path.sep + "ocr" + os.path.sep in p:
            continue
        if os.path.isfile(p):
            result.append(p)
    return result


def resolve_filename(requested_filename: str) -> Optional[str]:
    """
    Resolve requested filename (bisa original ATAU saved) ke saved_filename.

    Cari di metadata.json:
      1. Jika requested == key (saved_filename) → return requested
      2. Jika requested == value["original_filename"] → return key (saved)
      3. Jika requested == value["saved_as"] → return key
    Return None jika tidak ketemu.

    Args:
        requested_filename: Nama file dari URL path (sudah disanitasi basename).

    Returns:
        saved_filename atau None
    """
    # Lazy import untuk hindari circular
    try:
        from app.security import sanitize_filename_for_lookup
    except ImportError:
        # Fallback sederhana jika security belum siap
        import re

        def sanitize_filename_for_lookup(fn: str) -> str:  # type: ignore
            base = os.path.basename(fn)
            return re.sub(r"[^a-zA-Z0-9._-]", "", base)

    safe_req = sanitize_filename_for_lookup(requested_filename)
    if not safe_req:
        return None

    metadata = load_metadata()
    # 1. Direct hit: requested == saved key
    if safe_req in metadata:
        return safe_req

    # 2. Reverse lookup: cari original_filename / saved_as yang match
    #    Jika ada duplikat original, ambil yang paling baru (received_at terlama di akhir loop,
    #    tapi kita cari yang terbaru jadi iterasi reversed by received_at)
    #    Untuk simplisitas, loop dan simpan yang terbaru berdasarkan received_at
    best_match: Optional[str] = None
    best_time: str = ""
    # Untuk deteksi duplikat original (anti-stale warning)
    dup_count = 0
    for saved_key, meta in metadata.items():
        orig = sanitize_filename_for_lookup(meta.get("original_filename", ""))
        saved_as = sanitize_filename_for_lookup(meta.get("saved_as", ""))
        if safe_req == orig or safe_req == saved_as:
            dup_count += 1
            # Pilih yang received_at paling baru
            cur_time = meta.get("received_at", "") or meta.get("timestamp", "")
            if best_match is None or cur_time > best_time:
                # Log jika ada duplikat (bantu debug stale)
                if best_match is not None:
                    print(f"[RESOLVE WARN] Duplicate original {safe_req}: {best_match} ({best_time}) vs {saved_key} ({cur_time}) -> pick latest")
                best_match = saved_key
                best_time = cur_time
    if best_match:
        if dup_count > 1:
            print(f"[RESOLVE] {safe_req} -> {best_match} (dup {dup_count}, pick latest {best_time})")
        return best_match

    # 3. Fallback: cek apakah file fisik ada dengan nama requested (untuk file lama tanpa metadata lengkap)
    candidate = UPLOAD_DIR / safe_req
    if candidate.exists() and candidate.is_file():
        return safe_req

    # 4. Cek OCR json juga (jika metadata belum ada tapi OCR json ada)
    ocr_candidate = UPLOAD_DIR / "ocr" / f"{safe_req}.json"
    if ocr_candidate.exists():
        return safe_req

    return None


def get_metadata_by_any_filename(requested_filename: str) -> Optional[dict]:
    """Ambil metadata entry pakai original ATAU saved. Return None jika tidak ada."""
    saved = resolve_filename(requested_filename)
    if not saved:
        return None
    metadata = load_metadata()
    return metadata.get(saved)


def _dual_basis(name: str) -> str:
    """
    Basis stamp pasangan front/back: hilangkan suffix -front/-back + ekstensi.

    'label-ref-m-20260915-093015-front.png' -> 'label-ref-m-20260915-093015'
    'label-20260930-093419-bb86b5df-front-1.png' -> 'label-20260930-093419-bb86b5df'
      (split Label 1/2, 2026-09-30: suffix -1/-2 ikut di-strip agar 1 file
       = 1 basis; pengelompokan 1 sesi split ditangani RETAIN_COUNT>1)
    'label-20260901-xxxx.png' (single lama) -> 'label-20260901-xxxx' (diri sendiri)

    FIX dual-side front+back (2026-09-15): dipakai cleanup keep-sepasang agar
    BACK tidak menghapus FRONT yang 1 basis stamp.
    """
    import re

    base = name.rsplit(".", 1)[0] if "." in name else name
    return re.sub(r"-(front|back)(-[12])?$", "", base, flags=re.IGNORECASE)


def cleanup_old_files(keep_filename: str) -> dict:
    """
    Hapus file lama di uploads/ dan uploads/ocr/ kecuali keep_filename.
    Dipanggil setiap POST sukses (retake). Aman: pakai is_safe_path + sanitasi.

    NOTE: uploads/references/ sengaja diabaikan — library permanen
    (PLAN_REFERENCE_LIBRARY_FASTAPI). Fungsi ini hanya glob UPLOAD_DIR/*.png
    (non-rekursif) + OCR_DIR/*.json, jadi references.db & gambar reference aman.

    Args:
        keep_filename: saved_filename yang baru (yang dipertahankan), misal label-20260901-140351-66f7b6b8.png

    Returns:
        dict {"deleted_images": int, "deleted_ocr": int, "deleted_metadata": int, "errors": []}
    """
    from app.security import sanitize_filename_for_lookup, is_safe_path
    from app.config import OCR_DIR, RETAIN_COUNT, CLEANUP_RETAKE

    if not CLEANUP_RETAKE:
        return {"deleted_images": 0, "deleted_ocr": 0, "deleted_metadata": 0, "skipped": True, "errors": []}

    # Sanitasi keep
    keep_safe = sanitize_filename_for_lookup(keep_filename)
    if not keep_safe:
        return {"deleted_images": 0, "deleted_ocr": 0, "deleted_metadata": 0, "errors": [f"invalid keep_filename: {keep_filename}"]}

    stats = {"deleted_images": 0, "deleted_ocr": 0, "deleted_metadata": 0, "errors": []}

    # 1. Hapus image lama di UPLOAD_DIR (hanya *.png/*.jpg/*.jpeg, exclude subfolder ocr)
    # Deduplicate karena glob case-insensitive bisa return file sama 2x di Windows
    try:
        seen: set[str] = set()
        candidates: list[Path] = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"):
            for p in UPLOAD_DIR.glob(ext):
                if str(p.resolve()) in seen:
                    continue
                seen.add(str(p.resolve()))
                # Skip subfolder ocr (sudah di-handle di bagian 2)
                if OCR_DIR in p.parents:
                    continue
                if not p.is_file():
                    continue
                # Jangan hapus keep
                if p.name == keep_safe:
                    continue
                # FIX dual-side front+back: jangan hapus pasangan 1 basis stamp.
                # Sama basis = pasangan (atau diri sendiri yang sudah di-skip di atas).
                # Hanya berlaku bila keep punya suffix -front/-back → single lama tidak terpengaruh.
                if _dual_basis(p.name) == _dual_basis(keep_safe):
                    import re as _re

                    if _re.search(r"-(front|back)(-[12])?\.(png|jpg|jpeg)$", keep_safe, _re.IGNORECASE):
                        continue
                candidates.append(p)

        for p in candidates:
            # Safety: pastikan di dalam UPLOAD_DIR
            if not is_safe_path(UPLOAD_DIR, p):
                stats["errors"].append(f"skip unsafe {p}")
                continue
            try:
                p.unlink()
                stats["deleted_images"] += 1
                print(f"[CLEANUP] Deleted old image: {p.name}")
            except PermissionError as e:
                # Windows lock (WinError 32) — file masih dibuka oleh proses upload (test pakai file dari uploads)
                # Tidak fatal: log sebagai warning, biarkan next cleanup yang hapus
                stats["errors"].append(f"{p.name}: locked ({e})")
                print(f"[CLEANUP WARN] Locked (skip, will retry next retake): {p.name}: {e}")
            except Exception as e:
                stats["errors"].append(f"{p.name}: {e}")
                print(f"[CLEANUP ERROR] Failed delete {p}: {e}")

        # 1b. Juga cek file tanpa ext yang aneh (fallback scan semua file di UPLOAD_DIR)
        #     Tapi jangan hapus callback_log, dead_letter, metadata.json, .gitkeep
        keep_set = {keep_safe, "metadata.json", "callback_log.jsonl", "callback_dead_letter.jsonl", ".gitkeep"}
        for p in UPLOAD_DIR.iterdir():
            if p.is_file() and p.name not in keep_set and p.suffix.lower() not in (".png", ".jpg", ".jpeg"):
                # Hanya hapus jika namanya mirip label-*.png tapi terselip (misal .tmp)
                if p.name.startswith("label-"):
                    if not is_safe_path(UPLOAD_DIR, p):
                        continue
                    try:
                        p.unlink()
                        stats["deleted_images"] += 1
                        print(f"[CLEANUP] Deleted stray file: {p.name}")
                    except Exception as e:
                        stats["errors"].append(f"{p.name}: {e}")

    except Exception as e:
        stats["errors"].append(f"image cleanup failed: {e}")

    # 2. Hapus OCR JSON lama di OCR_DIR
    try:
        keep_json = f"{keep_safe}.json"
        keep_basis = _dual_basis(keep_safe)
        import re as _re2

        keep_is_dual = bool(_re2.search(r"-(front|back)(-[12])?\.(png|jpg|jpeg)$", keep_safe, _re2.IGNORECASE))
        for jf in OCR_DIR.glob("*.json"):
            if jf.name == keep_json:
                continue
            # FIX dual-side: pertahankan json pasangan 1 basis stamp.
            if keep_is_dual and _dual_basis(jf.name.replace(".json", "")) == keep_basis:
                continue
            if not jf.is_file():
                continue
            if not is_safe_path(OCR_DIR, jf):
                continue
            try:
                jf.unlink()
                stats["deleted_ocr"] += 1
                print(f"[CLEANUP] Deleted old OCR: {jf.name}")
            except Exception as e:
                stats["errors"].append(f"{jf.name}: {e}")

    except Exception as e:
        stats["errors"].append(f"ocr cleanup failed: {e}")

    # 3. Bersihkan metadata.json (hapus entri lama, sisakan keep)
    try:
        metadata = load_metadata()
        if keep_safe in metadata:
            import re as _re3

            keep_is_dual_meta = bool(_re3.search(r"-(front|back)(-[12])?\.(png|jpg|jpeg)$", keep_safe, _re3.IGNORECASE))
            if RETAIN_COUNT == 1 and keep_is_dual_meta:
                # FIX dual-side: keep 2 entri sepasang 1 basis stamp (bukan hanya keep).
                # Bila pasangan belum lengkap (baru front), new_meta berisi 1 → wajar.
                keep_basis_meta = _dual_basis(keep_safe)
                new_meta = {k: v for k, v in metadata.items() if _dual_basis(k) == keep_basis_meta}
                # Pastikan keep selalu ada (jaga race)
                new_meta.setdefault(keep_safe, metadata[keep_safe])
                deleted = len(metadata) - len(new_meta)
                stats["deleted_metadata"] = max(0, deleted)
                save_metadata(new_meta)
                if deleted > 0:
                    print(f"[CLEANUP] metadata.json: kept pair basis {keep_basis_meta} ({len(new_meta)} entries), deleted {deleted}")
            # Jika RETAIN_COUNT == 1: hanya keep 1
            elif RETAIN_COUNT == 1:
                new_meta = {keep_safe: metadata[keep_safe]}
                deleted = len(metadata) - 1
                stats["deleted_metadata"] = deleted
                save_metadata(new_meta)
                if deleted > 0:
                    print(f"[CLEANUP] metadata.json: kept {keep_safe}, deleted {deleted} old entries")
            else:
                # RETAIN_COUNT >1: keep N terbaru by received_at
                sorted_items = sorted(metadata.items(), key=lambda kv: kv[1].get("received_at", ""), reverse=True)
                keep_keys = {k for k, _ in sorted_items[:RETAIN_COUNT]}
                # Pastikan keep_safe selalu termasuk
                keep_keys.add(keep_safe)
                new_meta = {k: v for k, v in metadata.items() if k in keep_keys}
                deleted = len(metadata) - len(new_meta)
                stats["deleted_metadata"] = deleted
                save_metadata(new_meta)
                if deleted > 0:
                    print(f"[CLEANUP] metadata.json: retain {RETAIN_COUNT}, deleted {deleted}")
        else:
            # keep belum ada di metadata (race: cleanup dipanggil sebelum metadata save) → jangan hapus dulu
            print(f"[CLEANUP WARN] keep {keep_safe} not in metadata yet, skip metadata cleanup")
    except Exception as e:
        stats["errors"].append(f"metadata cleanup failed: {e}")

    # 4. Log ringkas
    print(f"[CLEANUP] Done for {keep_safe}: images={stats['deleted_images']}, ocr={stats['deleted_ocr']}, meta={stats['deleted_metadata']}")

    return stats


def delete_by_filename(requested_filename: str) -> dict:
    """
    Hapus 1 file spesifik untuk DELETE /api/test/ocr/{filename} & DELETE /api/test/status?filename=.
    Mendukung baik saved_filename maupun original_filename (resolve via metadata).

    Filesystem External (10.4):
      uploads/label-ref-*.png        -> input capture (di-handle sebagai uploads/label-*.png)
      uploads/ocr/label-*.png.json   -> hasil OCR
     ocr/latest?filename=            -> query (di-handle di GET)

    Args:
        requested_filename: Nama file dari URL (belum disanitasi, akan disanitasi internal).

    Returns:
        dict {
            "found": bool,               # True jika ada yang terhapus / ada di metadata
            "resolved_as": str|None,     # saved_filename jika ketemu
            "deleted_images": int,       # 0/1
            "deleted_ocr": int,          # 0/1 (+ hapus semua ocr json yang match original)
            "deleted_metadata": int,     # 0/1
            "deleted_ocr_files": list,   # daftar json yang dihapus
            "errors": list
        }
    """
    from app.security import sanitize_filename_for_lookup, is_safe_path
    from app.config import OCR_DIR

    # Sanitasi input (basename + hapus karakter berbahaya)
    try:
        safe_req = sanitize_filename_for_lookup(requested_filename)
    except Exception:
        safe_req = os.path.basename(requested_filename)
        import re

        safe_req = re.sub(r"[^a-zA-Z0-9._-]", "", safe_req)

    if not safe_req:
        return {
            "found": False,
            "resolved_as": None,
            "deleted_images": 0,
            "deleted_ocr": 0,
            "deleted_metadata": 0,
            "deleted_ocr_files": [],
            "errors": [f"invalid filename: {requested_filename}"],
        }

    # Resolve ke saved_filename (mendukung original)
    resolved = resolve_filename(safe_req)
    lookup = resolved if resolved else safe_req

    stats: dict = {
        "found": False,
        "resolved_as": resolved,
        "lookup": lookup,
        "deleted_images": 0,
        "deleted_ocr": 0,
        "deleted_metadata": 0,
        "deleted_ocr_files": [],
        "errors": [],
    }

    metadata = load_metadata()
    # Cek apakah ada di metadata (baik sebagai key maupun original)
    meta_exists = lookup in metadata
    # Jika tidak ada sebagai key tapi ada sebagai original, resolve sudah handle
    # Tapi untuk kasus file fisik tanpa metadata (legacy), tetap anggap found jika file ada
    image_path = UPLOAD_DIR / lookup
    # Safety check before consider found
    if not is_safe_path(UPLOAD_DIR, image_path):
        stats["errors"].append(f"unsafe path: {lookup}")
        return stats

    # Tentukan found: ada di metadata ATAU file image ada ATAU ocr json ada
    ocr_json_path = OCR_DIR / f"{lookup}.json"
    ocr_exists = ocr_json_path.exists() and ocr_json_path.is_file() and is_safe_path(OCR_DIR, ocr_json_path)
    image_exists = image_path.exists() and image_path.is_file()

    # Juga cek jika requested original tapi ada multiple saved dengan original sama?
    # Untuk DELETE spesifik, hanya hapus yang resolved (latest). Jika ingin hapus semua dengan original sama,
    # Laravel yang loop DELETE per original; FastAPI delete 1 per request. Cukup hapus resolved.
    # Namun jika tidak ada resolved tapi file fisik requested ada (tanpa metadata), fallback ke requested
    if not meta_exists and not image_exists and not ocr_exists:
        # Coba fallback: mungkin file ada dengan nama safe_req (bukan lookup) — misalnya original tanpa resolve karena metadata corrupt
        fallback_image = UPLOAD_DIR / safe_req
        fallback_ocr = OCR_DIR / f"{safe_req}.json"
        fallback_meta = safe_req in metadata
        if fallback_meta or (fallback_image.exists() and is_safe_path(UPLOAD_DIR, fallback_image)) or (fallback_ocr.exists() and is_safe_path(OCR_DIR, fallback_ocr)):
            lookup = safe_req
            image_path = fallback_image
            ocr_json_path = fallback_ocr
            meta_exists = fallback_meta
            image_exists = fallback_image.exists() and fallback_image.is_file()
            ocr_exists = fallback_ocr.exists() and fallback_ocr.is_file()
            stats["resolved_as"] = safe_req
            stats["lookup"] = lookup
        else:
            # Tidak ketemu sama sekali
            return stats

    stats["found"] = True

    # 1. Hapus image fisik
    if image_exists:
        try:
            image_path.unlink()
            stats["deleted_images"] = 1
            print(f"[DELETE] Deleted image: {lookup}")
        except PermissionError as e:
            stats["errors"].append(f"{lookup}: locked ({e})")
            print(f"[DELETE WARN] Locked image {lookup}: {e}")
        except Exception as e:
            stats["errors"].append(f"{lookup}: {e}")
            print(f"[DELETE ERROR] Failed delete image {lookup}: {e}")
    else:
        # Jika image tidak ada tapi metadata ada, tetap count sebagai ditemukan tapi tidak ada file fisik
        print(f"[DELETE] Image not on disk (already gone): {lookup}")

    # 2. Hapus OCR JSON
    # Hapus primary json (lookup.json)
    if ocr_exists:
        try:
            ocr_json_path.unlink()
            stats["deleted_ocr"] += 1
            stats["deleted_ocr_files"].append(ocr_json_path.name)
            print(f"[DELETE] Deleted OCR json: {ocr_json_path.name}")
        except Exception as e:
            stats["errors"].append(f"{ocr_json_path.name}: {e}")
            print(f"[DELETE ERROR] Failed delete OCR {ocr_json_path}: {e}")
    # Juga hapus json lain yang original_filename==requested (edge: duplicate original)
    # Scan ocr folder untuk json yang punya original_filename == safe_req
    try:
        for jf in OCR_DIR.glob("*.json"):
            if jf.name == f"{lookup}.json":
                continue  # sudah dihapus
            if not jf.is_file() or not is_safe_path(OCR_DIR, jf):
                continue
            try:
                with open(jf, "r", encoding="utf-8") as f:
                    data = json.load(f)
                orig = data.get("original_filename", "")
                # Sanitasi perbandingan
                from app.security import sanitize_filename_for_lookup as _san

                if _san(orig) == safe_req:
                    # Hapus json duplikat dengan original sama (rare, tapi bersihkan)
                    try:
                        jf.unlink()
                        stats["deleted_ocr"] += 1
                        stats["deleted_ocr_files"].append(jf.name)
                        print(f"[DELETE] Deleted duplicate OCR (original match) {jf.name} for {safe_req}")
                    except Exception as e2:
                        stats["errors"].append(f"{jf.name}: {e2}")
            except Exception:
                continue
    except Exception as e:
        stats["errors"].append(f"scan duplicate ocr failed: {e}")

    # 3. Hapus metadata entri
    if meta_exists:
        try:
            # Re-load fresh (jika race)
            metadata2 = load_metadata()
            if lookup in metadata2:
                del metadata2[lookup]
                save_metadata(metadata2)
                stats["deleted_metadata"] = 1
                print(f"[DELETE] Deleted metadata entry: {lookup}")
            else:
                # Mungkin key berbeda karena fallback? Coba cari original match
                for k, v in list(metadata2.items()):
                    from app.security import sanitize_filename_for_lookup as _san

                    if _san(v.get("original_filename", "")) == safe_req or k == safe_req:
                        del metadata2[k]
                        save_metadata(metadata2)
                        stats["deleted_metadata"] += 1
                        print(f"[DELETE] Deleted metadata duplicate {k} for {safe_req}")
                        break
        except Exception as e:
            stats["errors"].append(f"metadata: {e}")
            print(f"[DELETE ERROR] Failed delete metadata {lookup}: {e}")

    return stats
