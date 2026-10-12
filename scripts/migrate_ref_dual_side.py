"""
scripts/migrate_ref_dual_side.py — Tambah kolom front/back pada ref_library dan ref_blobs.

REVIEW_FLOW_COMPARE_FRONT_BACK_PLAN (2026-10-05) §6.6 TAHAP 6.

Jalankan:
  python scripts/migrate_ref_dual_side.py

Idempoten: kolom yang sudah ada di-skip (cek SHOW COLUMNS dulu).
Data lama: cleaned_text_front diisi dari cleaned_text (asumsi reference lama = front).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mysql.connector
from app.config import MYSQL_HOST, MYSQL_PORT, MYSQL_DB, MYSQL_USER, MYSQL_PASSWORD


def run():
    conn = mysql.connector.connect(
        host=MYSQL_HOST, port=MYSQL_PORT, database=MYSQL_DB,
        user=MYSQL_USER, password=MYSQL_PASSWORD,
    )
    cur = conn.cursor()

    # Cek kolom yang sudah ada
    cur.execute("SHOW COLUMNS FROM ref_library")
    existing_cols = {row[0] for row in cur.fetchall()}

    migrations = []
    if "cleaned_text_front" not in existing_cols:
        migrations.append("ALTER TABLE ref_library ADD COLUMN cleaned_text_front MEDIUMTEXT NULL AFTER cleaned_text")
    if "lines_json_front" not in existing_cols:
        # NOTE: MEDIUMTEXT tidak boleh punya DEFAULT di MySQL <8.0.13 → pakai NULL.
        migrations.append("ALTER TABLE ref_library ADD COLUMN lines_json_front MEDIUMTEXT NULL AFTER cleaned_text_front")
    if "cleaned_text_back" not in existing_cols:
        migrations.append("ALTER TABLE ref_library ADD COLUMN cleaned_text_back MEDIUMTEXT NULL AFTER lines_json_front")
    if "lines_json_back" not in existing_cols:
        # NOTE: sama seperti di atas — NULL tanpa DEFAULT.
        migrations.append("ALTER TABLE ref_library ADD COLUMN lines_json_back MEDIUMTEXT NULL AFTER cleaned_text_back")
    if "has_back" not in existing_cols:
        migrations.append("ALTER TABLE ref_library ADD COLUMN has_back TINYINT(1) NOT NULL DEFAULT 0 AFTER lines_json_back")

    # ref_blobs
    cur.execute("SHOW COLUMNS FROM ref_blobs")
    blob_cols = {row[0] for row in cur.fetchall()}
    if "image_back" not in blob_cols:
        migrations.append("ALTER TABLE ref_blobs ADD COLUMN image_back MEDIUMBLOB NULL AFTER image")
    if "thumb_back" not in blob_cols:
        migrations.append("ALTER TABLE ref_blobs ADD COLUMN thumb_back BLOB NULL AFTER thumb")

    if not migrations:
        print("[MIGRATE] Semua kolom sudah ada — skip.")
        cur.close()
        conn.close()
        return

    for sql in migrations:
        print(f"[MIGRATE] {sql[:80]}...")
        cur.execute(sql)

    # Migrasi data: reference lama -> front (asumsi gambar lama = front)
    cur.execute("""
        UPDATE ref_library
        SET cleaned_text_front = cleaned_text,
            lines_json_front = lines_json
        WHERE cleaned_text_front IS NULL
          AND cleaned_text IS NOT NULL
          AND cleaned_text != ''
    """)
    migrated = cur.rowcount

    conn.commit()
    cur.close()
    conn.close()
    print(f"[MIGRATE] Done. {len(migrations)} ALTER(s), {migrated} row(s) front-migrated.")


if __name__ == "__main__":
    run()
