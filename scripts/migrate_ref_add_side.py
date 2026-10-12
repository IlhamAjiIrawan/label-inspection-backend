"""
scripts/migrate_ref_add_side.py — Tambah kolom 'side' pada ref_library.

FIX_REFERENCE_OCR_ACUAN_PER_LABEL_PLAN (2026-10-07) Step F2d.

Jalankan: python scripts/migrate_ref_add_side.py

Idempoten: kolom yang sudah ada di-skip.
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

    cur.execute("SHOW COLUMNS FROM ref_library")
    existing_cols = {row[0] for row in cur.fetchall()}

    migrations = []
    if "side" not in existing_cols:
        migrations.append(
            "ALTER TABLE ref_library ADD COLUMN side VARCHAR(10) NOT NULL DEFAULT '' AFTER template_id"
        )

    if not migrations:
        print("[MIGRATE] Kolom 'side' sudah ada — skip.")
        cur.close()
        conn.close()
        return

    for sql in migrations:
        print(f"[MIGRATE] {sql[:80]}...")
        cur.execute(sql)

    conn.commit()
    cur.close()
    conn.close()
    print(f"[MIGRATE] Done. {len(migrations)} ALTER(s).")


if __name__ == "__main__":
    run()
