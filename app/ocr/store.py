"""
app/ocr/store.py — OCRResultStore: simpan hasil OCR di memory (thread-safe) + disk (JSON).

Memory: OrderedDict max 10 hasil terakhir (LRU).
Disk:   uploads/ocr/{filename}.json untuk persistensi setelah restart.

Thread-safe via threading.Lock. Simpan ke disk di luar lock agar tidak blocking lama,
tapi latest_filename dijaga di dalam lock.
"""

import json
import threading
import logging
from collections import OrderedDict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

from app.config import OCR_DIR, OCR_MAX_MEMORY_RESULTS

logger = logging.getLogger(__name__)

JAKARTA_TZ = timezone(timedelta(hours=7))


class OCRResultStore:
    """
    Menyimpan hasil OCR di memory (thread-safe) dan disk (JSON).
    Memory menyimpan maksimum OCR_MAX_MEMORY_RESULTS hasil terakhir.
    Disk menyimpan semua hasil di OCR_DIR.
    """

    def __init__(self, ocr_dir: str | Path | None = None):
        self._lock = threading.Lock()
        self._results: OrderedDict[str, dict] = OrderedDict()
        self._latest_filename: str | None = None
        self._ocr_dir = Path(ocr_dir) if ocr_dir else OCR_DIR
        self._ocr_dir.mkdir(parents=True, exist_ok=True)
        # R4 long-poll: 1 Event per key yang masih pending.
        self._events: dict[str, threading.Event] = {}
        # Saat init, load hasil yang sudah ada di disk ke memory (max 10 terbaru)
        self._load_existing_from_disk()

    def _load_existing_from_disk(self) -> None:
        """Load file JSON yang sudah ada di disk ke memory (terbaru max N)."""
        try:
            json_files = list(self._ocr_dir.glob("*.json"))
            if not json_files:
                return
            # Urutkan berdasarkan mtime terbaru
            json_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            # Ambil max N terbaru
            for jf in reversed(json_files[:OCR_MAX_MEMORY_RESULTS][::-1]):
                # reversed agar urutan tetap kronologis saat insert
                try:
                    with open(jf, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        fname = data.get("filename") or jf.stem.replace(".png", "").replace(".jpg", "").replace(".jpeg", "")
                        # fallback: nama file tanpa .json
                        if not data.get("filename"):
                            # jf.name = label-...png.json -> hilangkan .json
                            fname = jf.name[: -len(".json")] if jf.name.endswith(".json") else jf.stem
                            data["filename"] = fname
                        self._results[fname] = data
                        self._latest_filename = fname
                except Exception as e:
                    logger.warning(f"Failed to load OCR json {jf}: {e}")
            # Pastikan latest adalah file dengan mtime terbaru
            if json_files:
                latest_jf = json_files[0]
                try:
                    with open(latest_jf, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        fname = data.get("filename")
                        if fname:
                            self._latest_filename = fname
                        else:
                            self._latest_filename = latest_jf.name[: -len(".json")] if latest_jf.name.endswith(".json") else latest_jf.stem
                except Exception:
                    pass
        except Exception as e:
            logger.warning(f"Failed to load existing OCR results: {e}")

    def save(self, result: dict) -> None:
        """
        Simpan hasil OCR ke memory (+ disk HANYA jika OCR_DISK_PERSIST=true).

        F7 (2026-09-15): default TANPA tulis disk — polling baca memory,
        persistensi dipegang Laravel via callback (reference_ocr_results).
        Fallback baca disk di get_by_filename dibiarkan (read-only, untuk arsip lama).

        Args:
            result: dict hasil OCR dengan minimal key "filename".
        """
        filename = result.get("filename")
        if not filename:
            logger.error("OCRResultStore.save called without filename")
            return

        # Pastikan processed_at ada
        if "processed_at" not in result or not result["processed_at"]:
            result["processed_at"] = datetime.now(JAKARTA_TZ).isoformat()

        with self._lock:
            self._results[filename] = result
            self._latest_filename = filename

            # Batasi jumlah di memory (LRU)
            while len(self._results) > OCR_MAX_MEMORY_RESULTS:
                evicted, _ = self._results.popitem(last=False)
                # R4: bersihkan Event key yang ter-eviksi agar tidak bocor memori.
                try:
                    self._events.pop(evicted, None)
                except Exception:
                    pass

        # R4: bangunkan penunggu long-poll (di luar lock agar tidak deadlock).
        try:
            import os as _os

            self._event(_os.path.basename(str(filename))).set()
        except Exception:
            pass

        # Simpan ke disk (di luar lock) — HANYA jika diminta (debug offline)
        try:
            from app.config import OCR_DISK_PERSIST
        except ImportError:
            OCR_DISK_PERSIST = True  # fallback aman = perilaku lama
        if not OCR_DISK_PERSIST:
            return
        json_path = self._ocr_dir / f"{filename}.json"
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save OCR result to disk {json_path}: {e}", exc_info=True)

    def _event(self, key: str) -> threading.Event:
        """Ambil (atau buat) Event untuk key — untuk long-poll R4."""
        import os as _os

        key = _os.path.basename(key)
        with self._lock:
            ev = self._events.get(key)
            if ev is None:
                ev = threading.Event()
                self._events[key] = ev
            return ev

    def put(self, key: str, result: dict) -> None:
        """Alias save() + bangunkan penunggu long-poll (R4)."""
        self.save(result)
        try:
            self._event(key).set()
        except Exception:
            pass

    def wait(self, key: str, timeout: float) -> bool:
        """
        Tunggu hasil key hingga timeout detik (R4 long-poll).

        Return True bila hasil sudah ada (langsung atau setelah menunggu).
        Tidak ada race: put() menyimpan lalu set(); wait() cek get dulu.
        """
        import os as _os

        key = _os.path.basename(key)
        if self.get_by_filename(key) is not None:
            return True
        try:
            to = max(0.0, float(timeout or 0.0))
        except Exception:
            to = 0.0
        if to <= 0:
            return False
        try:
            return self._event(key).wait(to)
        except Exception:
            return False

    def clear_all(self) -> int:
        """
        Hapus semua entry di memory (kecuali akan diisi ulang dengan latest).
        Dipanggil bersama cleanup_old_files saat retake.
        Return jumlah yang dihapus.
        """
        with self._lock:
            count = len(self._results)
            self._results.clear()
            self._latest_filename = None
            try:
                self._events.clear()
            except Exception:
                pass
            return count

    def remove(self, filename: str) -> bool:
        """Hapus 1 entry spesifik dari memory. Return True jika ada."""
        import os

        filename = os.path.basename(filename)
        with self._lock:
            if filename in self._results:
                del self._results[filename]
                try:
                    self._events.pop(filename, None)
                except Exception:
                    pass
                # Update latest jika yang dihapus adalah latest
                if self._latest_filename == filename:
                    self._latest_filename = next(reversed(self._results), None) if self._results else None
                return True
        return False

    def retain_only(self, keep_filename: str) -> int:
        """
        Hapus semua dari memory kecuali keep_filename.
        Dipanggil saat cleanup retake.
        """
        import os

        keep = os.path.basename(keep_filename)
        with self._lock:
            to_delete = [k for k in self._results.keys() if k != keep]
            for k in to_delete:
                del self._results[k]
                try:
                    self._events.pop(k, None)
                except Exception:
                    pass
            # Pastikan latest = keep jika ada, else None atau sisa
            if keep in self._results:
                self._latest_filename = keep
            elif self._results:
                self._latest_filename = next(reversed(self._results))
            else:
                self._latest_filename = None
            return len(to_delete)

    def retain_last_n(self, n: int) -> int:
        """
        Pertahankan N entri TERBARU (insertion order) di memory, hapus sisanya.

        FIX split dual-side (2026-09-30, false-empty Label 1): alur split
        mengirim 4 label berurutan (front-1, front-2, back-1, back-2).
        retain_pair/retain_only menghapus hasil label sebelumnya karena
        dipanggil saat POST (sebelum OCR label baru selesai, keep belum
        ada di memory → to_keep kosong → hapus semua). Metode ini TIDAK
        pernah menghapus semua saat keep belum ada: hanya trim ke N
        terbaru. N diambil dari RETAIN_COUNT (default 5 = 1 sesi split
        isi 4 + 1 margin). LRU cap 10 di save() tetap berlaku.
        Return jumlah yang dihapus.
        """
        import os as _os

        try:
            n = max(1, int(n or 1))
        except Exception:
            n = 5
        with self._lock:
            removed = 0
            while len(self._results) > n:
                evicted, _ = self._results.popitem(last=False)
                try:
                    self._events.pop(evicted, None)
                except Exception:
                    pass
                removed += 1
            # Jaga latest tetap valid
            if self._latest_filename not in self._results:
                self._latest_filename = next(reversed(self._results), None) if self._results else None
            return removed

    def retain_pair(self, keep_filename: str) -> int:
        """
        FIX dual-side front+back (2026-09-15): hapus semua kecuali keep + pasangannya.

        'label-20260930-093419-bb86b5df-front.png' keep →
          pertahankan '...-front.png' + '...-back.png' bila ada (1 basis stamp).
        Single lama (tanpa suffix -front/-back) → perilaku sama seperti retain_only.
        Return jumlah yang dihapus.

        CATATAN 2026-09-30: untuk alur split Label 1/2, suffix bisa
        '-front-1'/'-back-2' (lihat generate_safe_filename). Regex di sini
        sudah menangani (-[12])? agar pasangan 1 sesi tidak terhapus.
        Untuk retensi umum multi-label, pakai retain_last_n() (dipanggil
        main.py) — metode ini dipertahankan untuk kompatibilitas.
        """
        import os as _os
        import re as _re

        keep = _os.path.basename(keep_filename)
        base_keep = _re.sub(r"-(front|back)(-[12])?(\.[^.]+)?$", "", keep, flags=_re.IGNORECASE).lower()
        is_dual = bool(_re.search(r"-(front|back)(-[12])?(\.[^.]+)?$", keep, flags=_re.IGNORECASE))
        with self._lock:
            def _base(k: str) -> str:
                return _re.sub(r"-(front|back)(-[12])?(\.[^.]+)?$", "", k, flags=_re.IGNORECASE).lower()

            if is_dual:
                to_keep = {k for k in self._results.keys() if _base(k) == base_keep}
                to_keep.add(keep)
                # Hanya yang benar-benar ada di memory
                to_keep = {k for k in to_keep if k in self._results}
            else:
                to_keep = {keep} if keep in self._results else set()
            to_delete = [k for k in list(self._results.keys()) if k not in to_keep]
            for k in to_delete:
                del self._results[k]
                try:
                    self._events.pop(k, None)
                except Exception:
                    pass
            if keep in self._results:
                self._latest_filename = keep
            elif to_keep:
                self._latest_filename = sorted(to_keep)[-1]
            elif self._results:
                self._latest_filename = next(reversed(self._results))
            else:
                self._latest_filename = None
            return len(to_delete)

    def get_latest(self) -> Optional[dict]:
        """Ambil hasil OCR terbaru dari memory. Return None jika belum ada."""
        with self._lock:
            if self._latest_filename and self._latest_filename in self._results:
                return self._results[self._latest_filename]
            # Jika latest_filename tidak ada tapi results ada, ambil yang terakhir
            if self._results:
                # OrderedDict: last item = terbaru
                last_key = next(reversed(self._results))
                return self._results[last_key]
            return None

    def get_by_filename(self, filename: str) -> Optional[dict]:
        """
        Ambil hasil OCR berdasarkan filename.

        Cek memory dulu, lalu fallback ke disk.

        Args:
            filename: Nama file gambar (misal label-20260828-xxxx.png)

        Returns:
            dict hasil OCR atau None jika tidak ditemukan.
        """
        # Sanitasi: hanya basename
        import os

        filename = os.path.basename(filename)

        with self._lock:
            if filename in self._results:
                return self._results[filename]

        # Fallback ke disk
        json_path = self._ocr_dir / f"{filename}.json"
        if json_path.exists():
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # Masukkan ke memory cache juga
                    with self._lock:
                        self._results[filename] = data
                        # Batasi ukuran
                        while len(self._results) > OCR_MAX_MEMORY_RESULTS:
                            self._results.popitem(last=False)
                    return data
            except Exception as e:
                logger.warning(f"Failed to read OCR json {json_path}: {e}")
                return None
        return None

    def list_all(self) -> list[dict]:
        """Return semua hasil di memory (copy)."""
        with self._lock:
            return list(self._results.values())

    def count_memory(self) -> int:
        with self._lock:
            return len(self._results)

    def latest_filename(self) -> Optional[str]:
        with self._lock:
            return self._latest_filename


# Singleton global instance (bisa diimport dari main.py)
ocr_store = OCRResultStore()
