"""
scripts/migrate_library_sqlite_to_mysql.py — F5 sekali jalan
(PLAN_MYSQL_BACKED_REFERENCE_STORAGE §5-F5).

Baca semua row SQLite (uploads/references/references.db) + file
ref-*.jpg / ref-*_thumb.jpg → INSERT ke MySQL (idempotent: skip by sha256).

Output ringkasan JSON: {total_sqlite, inserted, skipped_duplicate,
missing_files[], failed[]}. Exit code != 0 jika failed/missing_files.

Contoh:
    set PYTHONPATH=C:\\Users\\Aji\\OneDrive\\Desktop\\fastAPI
    python scripts\\migrate_library_sqlite_to_mysql.py
"""

import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import db_mysql as my  # noqa: E402
from app.config import REF_ROOT  # noqa: E402

SQLITE_DB = REF_ROOT / "references.db"

# Kolom SQLite -> insert MySQL (stored/thumb filename TIDAK dibawa; bytes dari file)
# FIX_REFERENCE_OCR_ACUAN_PER_LABEL (2026-10-07): + side agar sisi tidak hilang saat migrasi.
FIELDS = [
    "ref_id", "name", "template_id", "side", "source", "user_id", "sha256",
    "text_hash", "cleaned_text", "lines_json", "engine", "ocr_status",
    "is_empty", "error", "file_size_bytes", "processing_time_seconds",
    "usage_count", "created_at", "updated_at", "last_used_at",
]


def main() -> int:
    my.init_db()

    if not SQLITE_DB.exists():
        print(json.dumps({"total_sqlite": 0, "inserted": 0,
                          "skipped_duplicate": 0, "missing_files": [],
                          "failed": [{"ref_id": "-", "error": "references.db tidak ada"}]}))
        return 1

    conn = sqlite3.connect(str(SQLITE_DB))
    conn.row_factory = sqlite3.Row
    rows = [dict(r) for r in conn.execute("SELECT * FROM refs")]
    conn.close()

    summary: dict = {"total_sqlite": len(rows), "inserted": 0,
                     "skipped_duplicate": 0, "missing_files": [], "failed": []}

    for r in rows:
        ref_id = r.get("ref_id", "")
        try:
            # 1. Idempotent: skip jika sha sudah ada di MySQL
            if my.find_by_sha256(r.get("sha256", "")):
                summary["skipped_duplicate"] += 1
                continue
            # 2. Baca bytes gambar dari file arsip
            img_p = REF_ROOT / (r.get("stored_filename") or "")
            th_p = REF_ROOT / (r.get("thumb_filename") or "")
            missing = [str(p) for p in (img_p, th_p)
                       if not p.name or not p.exists() or not p.is_file()]
            if missing:
                summary["missing_files"].append({"ref_id": ref_id, "missing": missing})
                continue
            # 3. INSERT transaksi (row + blob)
            payload = {k: r.get(k) for k in FIELDS}
            payload["image_bytes"] = img_p.read_bytes()
            payload["thumb_bytes"] = th_p.read_bytes()
            try:
                my.insert_ref(payload)
            except my.DuplicateRefError:
                summary["skipped_duplicate"] += 1
                continue
            summary["inserted"] += 1
        except Exception as e:
            summary["failed"].append({"ref_id": ref_id, "error": str(e)[:300]})

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if not summary["failed"] and not summary["missing_files"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
