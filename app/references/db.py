"""
app/references/db.py — SQLite CRUD untuk Reference Library.

- 1 koneksi pendek per fungsi (aman Windows lock) + timeout=10 + WAL.
- Semua query pakai placeholder "?" (anti SQL injection dari ?q=).
- Lihat plans/PLAN_REFERENCE_LIBRARY_FASTAPI.md §4.4 + §5-T3.
"""

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Optional

from app.config import REF_DB_PATH

JAKARTA_TZ = timezone(timedelta(hours=7))

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS refs (
  ref_id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  template_id TEXT DEFAULT '',
  side TEXT DEFAULT '',
  source TEXT DEFAULT '',
  user_id TEXT DEFAULT '',
  stored_filename TEXT NOT NULL,
  thumb_filename TEXT NOT NULL,
  sha256 TEXT NOT NULL UNIQUE,
  text_hash TEXT DEFAULT '',
  cleaned_text TEXT DEFAULT '',
  lines_json TEXT DEFAULT '[]',
  engine TEXT DEFAULT '',
  ocr_status TEXT DEFAULT 'pending',
  is_empty INTEGER DEFAULT 0,
  error TEXT DEFAULT '',
  file_size_bytes INTEGER DEFAULT 0,
  processing_time_seconds REAL DEFAULT 0,
  usage_count INTEGER DEFAULT 0,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  last_used_at TEXT DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_refs_template ON refs(template_id);
CREATE INDEX IF NOT EXISTS idx_refs_created ON refs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_refs_used ON refs(last_used_at DESC);
CREATE INDEX IF NOT EXISTS idx_refs_text_hash ON refs(text_hash);
"""


def _connect() -> sqlite3.Connection:
    """Buat koneksi pendek per operasi. Caller wajib close di finally."""
    REF_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(REF_DB_PATH), timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def _now_iso() -> str:
    return datetime.now(JAKARTA_TZ).isoformat()


def _row_to_dict(row: sqlite3.Row) -> dict:
    """Konversi Row -> dict + parse lines_json -> lines. Jangan bocorkan lines_json mentah."""
    d = dict(row)
    raw_lines = d.pop("lines_json", "[]")
    try:
        parsed = json.loads(raw_lines) if raw_lines else []
        d["lines"] = parsed if isinstance(parsed, list) else []
    except (json.JSONDecodeError, TypeError):
        d["lines"] = []
    # Normalisasi tipe
    d["is_empty"] = bool(d.get("is_empty", 0))
    d["usage_count"] = int(d.get("usage_count", 0) or 0)
    d["file_size_bytes"] = int(d.get("file_size_bytes", 0) or 0)
    try:
        d["processing_time_seconds"] = float(d.get("processing_time_seconds", 0) or 0)
    except (TypeError, ValueError):
        d["processing_time_seconds"] = 0.0
    # FIX_REFERENCE_OCR_ACUAN_PER_LABEL (2026-10-07): default side/front/back
    # agar router yang akses row.get("side"/"cleaned_text_front") tidak KeyError
    # pada DB sqlite lama yang belum migrasi.
    if d.get("side") is None:
        d["side"] = ""
    if "cleaned_text_front" not in d:
        d["cleaned_text_front"] = ""
    if "cleaned_text_back" not in d:
        d["cleaned_text_back"] = ""
    if "lines_front" not in d:
        d["lines_front"] = []
    if "lines_back" not in d:
        d["lines_back"] = []
    if "has_back" not in d:
        d["has_back"] = False
    return d


def init_db() -> None:
    """Buat tabel + index + WAL. Idempotent, aman dipanggil tiap startup."""
    conn = _connect()
    try:
        conn.executescript(SCHEMA_SQL)
        # FIX_REFERENCE_OCR_ACUAN_PER_LABEL (2026-10-07): migrasi kolom side
        # untuk file sqlite lama yang dibuat sebelum kolom side ada.
        try:
            cur = conn.execute("PRAGMA table_info(refs)")
            _cols = {r[1] for r in cur.fetchall()}
            if "side" not in _cols:
                conn.execute("ALTER TABLE refs ADD COLUMN side TEXT DEFAULT ''")
        except Exception:
            pass
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
        except Exception:
            pass
        conn.commit()
    finally:
        conn.close()


def insert_ref(row: dict) -> None:
    """Insert 1 reference. row wajib berisi ref_id, name, stored_filename, thumb_filename, sha256, created_at, updated_at."""
    conn = _connect()
    try:
        # FIX_REFERENCE_OCR_ACUAN_PER_LABEL (2026-10-07): sertakan side,
        # fallback ke INSERT lama bila file sqlite belum migrasi.
        _clean_side = (row.get("side", "") or "").strip().lower()
        try:
            conn.execute(
                """
                INSERT INTO refs (
                  ref_id, name, template_id, side, source, user_id,
                  stored_filename, thumb_filename, sha256, text_hash,
                  cleaned_text, lines_json, engine, ocr_status, is_empty, error,
                  file_size_bytes, processing_time_seconds, usage_count,
                  created_at, updated_at, last_used_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    row.get("ref_id", ""),
                    row.get("name", ""),
                    row.get("template_id", "") or "",
                    _clean_side,
                    row.get("source", "") or "",
                    str(row.get("user_id", "") or ""),
                    row.get("stored_filename", ""),
                    row.get("thumb_filename", ""),
                    row.get("sha256", ""),
                    row.get("text_hash", "") or "",
                    row.get("cleaned_text", "") or "",
                    row.get("lines_json", "[]") or "[]",
                    row.get("engine", "") or "",
                    row.get("ocr_status", "pending") or "pending",
                    1 if row.get("is_empty") else 0,
                    row.get("error", "") or "",
                    int(row.get("file_size_bytes", 0) or 0),
                    float(row.get("processing_time_seconds", 0) or 0),
                    int(row.get("usage_count", 0) or 0),
                    row.get("created_at", _now_iso()),
                    row.get("updated_at", _now_iso()),
                    row.get("last_used_at", "") or "",
                ),
            )
        except Exception as e_new:
            # DB lama tanpa kolom side → ulangi tanpa side
            if "no such column" not in str(e_new).lower() and "side" not in str(e_new).lower():
                raise
            conn.execute(
                """
                INSERT INTO refs (
                  ref_id, name, template_id, source, user_id,
                  stored_filename, thumb_filename, sha256, text_hash,
                  cleaned_text, lines_json, engine, ocr_status, is_empty, error,
                  file_size_bytes, processing_time_seconds, usage_count,
                  created_at, updated_at, last_used_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    row.get("ref_id", ""),
                    row.get("name", ""),
                    row.get("template_id", "") or "",
                    row.get("source", "") or "",
                    str(row.get("user_id", "") or ""),
                    row.get("stored_filename", ""),
                    row.get("thumb_filename", ""),
                    row.get("sha256", ""),
                    row.get("text_hash", "") or "",
                    row.get("cleaned_text", "") or "",
                    row.get("lines_json", "[]") or "[]",
                    row.get("engine", "") or "",
                    row.get("ocr_status", "pending") or "pending",
                    1 if row.get("is_empty") else 0,
                    row.get("error", "") or "",
                    int(row.get("file_size_bytes", 0) or 0),
                    float(row.get("processing_time_seconds", 0) or 0),
                    int(row.get("usage_count", 0) or 0),
                    row.get("created_at", _now_iso()),
                    row.get("updated_at", _now_iso()),
                    row.get("last_used_at", "") or "",
                ),
            )
        conn.commit()
    finally:
        conn.close()


def get_ref(ref_id: str) -> Optional[dict]:
    """Ambil 1 reference by id. Return None jika tidak ada."""
    conn = _connect()
    try:
        cur = conn.execute("SELECT * FROM refs WHERE ref_id = ?", (ref_id,))
        row = cur.fetchone()
        if row is None:
            return None
        return _row_to_dict(row)
    finally:
        conn.close()


def find_by_sha256(sha: str) -> Optional[dict]:
    """Cari reference dengan SHA256 gambar persis (dedup lapis 1)."""
    if not sha:
        return None
    conn = _connect()
    try:
        cur = conn.execute("SELECT * FROM refs WHERE sha256 = ? LIMIT 1", (sha,))
        row = cur.fetchone()
        if row is None:
            return None
        return _row_to_dict(row)
    finally:
        conn.close()


def list_refs(
    q: str = "",
    template_id: str = "",
    ocr_status: str = "",
    sort: str = "recent_used",
    page: int = 1,
    per_page: int = 20,
) -> tuple[list[dict], int]:
    """
    List ringan untuk dropdown Laravel (tanpa lines full, hanya excerpt).

    Returns:
        (items, total_count). items berisi: ref_id, name, template_id,
        ocr_status, excerpt, usage_count, created_at, last_used_at, sha256.
    """
    # Clamp defense-in-depth
    try:
        page = int(page)
    except (TypeError, ValueError):
        page = 1
    try:
        per_page = int(per_page)
    except (TypeError, ValueError):
        per_page = 20
    if page < 1:
        page = 1
    if per_page < 1:
        per_page = 20
    if per_page > 50:
        per_page = 50

    where_parts: list[str] = []
    params: list = []

    qq = (q or "").strip()
    if qq:
        where_parts.append("(name LIKE ? ESCAPE '\\' OR cleaned_text LIKE ? ESCAPE '\\')")
        # Escape % _ \ agar search literal aman
        esc = qq.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        like = f"%{esc}%"
        params.extend([like, like])

    if template_id:
        where_parts.append("template_id = ?")
        params.append(template_id)

    if ocr_status in ("pending", "success", "failed"):
        where_parts.append("ocr_status = ?")
        params.append(ocr_status)

    where_sql = ("WHERE " + " AND ".join(where_parts)) if where_parts else ""

    if sort == "newest":
        order_sql = "ORDER BY created_at DESC"
    elif sort == "name":
        order_sql = "ORDER BY name COLLATE NOCASE ASC, created_at DESC"
    else:  # recent_used (default): last_used kosong dianggap paling lama
        order_sql = "ORDER BY CASE WHEN last_used_at = '' THEN '' ELSE last_used_at END DESC, created_at DESC"

    offset = (page - 1) * per_page

    conn = _connect()
    try:
        cur = conn.execute(f"SELECT COUNT(*) AS c FROM refs {where_sql}", params)
        total = int(cur.fetchone()["c"] or 0)

        cur = conn.execute(
            f"""
            SELECT ref_id, name, template_id, ocr_status, cleaned_text,
                   usage_count, created_at, last_used_at, sha256
            FROM refs {where_sql} {order_sql} LIMIT ? OFFSET ?
            """,
            (*params, per_page, offset),
        )
        items: list[dict] = []
        for r in cur.fetchall():
            ct = r["cleaned_text"] or ""
            excerpt = " ".join(ct.split())[:120]
            items.append(
                {
                    "ref_id": r["ref_id"],
                    "name": r["name"],
                    "template_id": r["template_id"] or "",
                    "ocr_status": r["ocr_status"] or "pending",
                    "excerpt": excerpt,
                    "usage_count": int(r["usage_count"] or 0),
                    "created_at": r["created_at"] or "",
                    "last_used_at": r["last_used_at"] or "",
                    "sha256": r["sha256"] or "",
                }
            )
        return items, total
    finally:
        conn.close()


def update_ref(ref_id: str, fields: dict) -> bool:
    """
    Update parsial kolom yang diizinkan. Return True jika ada row terupdate.

    Kolom yang boleh diubah: name, template_id, source, user_id, text_hash,
    cleaned_text, lines_json, engine, ocr_status, is_empty, error,
    file_size_bytes, processing_time_seconds, updated_at.
    """
    allowed = {
        "name",
        "template_id",
        "side",
        "source",
        "user_id",
        "text_hash",
        "cleaned_text",
        "lines_json",
        "engine",
        "ocr_status",
        "is_empty",
        "error",
        "file_size_bytes",
        "processing_time_seconds",
        "updated_at",
    }
    sets: list[str] = []
    params: list = []
    for k, v in (fields or {}).items():
        if k not in allowed:
            continue
        if k == "is_empty":
            sets.append("is_empty = ?")
            params.append(1 if v else 0)
        else:
            sets.append(f"{k} = ?")
            params.append(v)
    if not sets:
        return False
    # Selalu refresh updated_at kecuali caller sudah set
    if "updated_at" not in (fields or {}):
        sets.append("updated_at = ?")
        params.append(_now_iso())
    params.append(ref_id)

    conn = _connect()
    try:
        try:
            cur = conn.execute(f"UPDATE refs SET {', '.join(sets)} WHERE ref_id = ?", params)
        except Exception as e_upd:
            # DB lama tanpa kolom side → ulangi tanpa side
            if "no such column" not in str(e_upd).lower() or "side" not in ",".join(sets).lower():
                raise
            filt_sets: list[str] = []
            filt_params: list = []
            for k, v in (fields or {}).items():
                if k not in allowed or k == "side":
                    continue
                if k == "is_empty":
                    filt_sets.append("is_empty = ?")
                    filt_params.append(1 if v else 0)
                else:
                    filt_sets.append(f"{k} = ?")
                    filt_params.append(v)
            if "updated_at" not in (fields or {}):
                filt_sets.append("updated_at = ?")
                filt_params.append(_now_iso())
            if not filt_sets:
                return False
            filt_params.append(ref_id)
            cur = conn.execute(f"UPDATE refs SET {', '.join(filt_sets)} WHERE ref_id = ?", filt_params)
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def delete_ref(ref_id: str) -> Optional[dict]:
    """Hapus 1 row. Return row lama (untuk hapus file) atau None."""
    old = get_ref(ref_id)
    if old is None:
        return None
    conn = _connect()
    try:
        conn.execute("DELETE FROM refs WHERE ref_id = ?", (ref_id,))
        conn.commit()
    finally:
        conn.close()
    return old


def count_and_bytes() -> tuple[int, int]:
    """Return (jumlah_rows, total_file_size_bytes). Untuk cek quota."""
    conn = _connect()
    try:
        cur = conn.execute("SELECT COUNT(*) AS c, COALESCE(SUM(file_size_bytes),0) AS b FROM refs")
        row = cur.fetchone()
        return int(row["c"] or 0), int(row["b"] or 0)
    finally:
        conn.close()


def touch_used(ref_id: str) -> None:
    """Naikkan usage_count + refresh last_used_at. Best-effort, jangan raise."""
    try:
        conn = _connect()
        try:
            conn.execute(
                "UPDATE refs SET usage_count = usage_count + 1, last_used_at = ? WHERE ref_id = ?",
                (_now_iso(), ref_id),
            )
            conn.commit()
        finally:
            conn.close()
    except Exception:
        pass
