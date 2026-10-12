"""
app/db_mysql.py — MySQL CRUD untuk Reference Library (PLAN_MYSQL_BACKED_REFERENCE_STORAGE F2).

- Nama fungsi & signature PERSIS seperti app/references/db.py agar router hanya ganti import.
- Pool global 1x saat import (pool_size + pool_recycle dari app.config).
- Semua query pakai placeholder "%s" (JANGAN f-string untuk value).
- get_ref / list_refs TIDAK PERNAH SELECT kolom blob (aturan keras §2.2).
- Blob hanya via get_blob() (untuk serve image/thumb).
- Error mapping: 2006/2013/2055 → 1x retry koneksi baru; 1062 → DuplicateRefError.
"""

import json
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional, TypeVar

import mysql.connector
from mysql.connector import Error as MySQLError
from mysql.connector.pooling import MySQLConnectionPool

from app.config import (
    MYSQL_DB,
    MYSQL_HOST,
    MYSQL_PASSWORD,
    MYSQL_POOL_SIZE,
    MYSQL_PORT,
    MYSQL_USER,
)

logger = logging.getLogger(__name__)

JAKARTA_TZ = timezone(timedelta(hours=7))

# Error yang boleh 1x retry dengan koneksi baru (koneksi putus / gone away)
RETRY_ERRNOS = {2006, 2013, 2055}
# 1062 = duplicate entry (UNIQUE sha256) → duplikat, bukan 500
DUPLICATE_ERRNO = 1062


class DuplicateRefError(Exception):
    """Dilempar insert_ref saat sha256 sudah ada (race dengan dedup lapis 1)."""

    def __init__(self, ref_id: str):
        super().__init__(f"duplicate sha256, existing ref_id={ref_id}")
        self.ref_id = ref_id


class DBUnavailableError(Exception):
    """Dilempar saat MySQL tidak bisa dihubungi (router balas 503)."""


SCHEMA_SQL = [
    """
    CREATE TABLE IF NOT EXISTS ref_library (
      ref_id               VARCHAR(64)  NOT NULL PRIMARY KEY,
      name                 VARCHAR(80)  NOT NULL,
      template_id          VARCHAR(32)  NOT NULL DEFAULT '',
      side                 VARCHAR(10)  NOT NULL DEFAULT '',
      source               VARCHAR(64)  NOT NULL DEFAULT '',
      user_ref             VARCHAR(64)  NOT NULL DEFAULT '',
      sha256               CHAR(64)     NOT NULL,
      text_hash            CHAR(64)     NOT NULL DEFAULT '',
      cleaned_text         MEDIUMTEXT   NULL,
      lines_json           MEDIUMTEXT   NOT NULL,
      cleaned_text_front   MEDIUMTEXT   NULL,
      lines_json_front     MEDIUMTEXT   NULL,
      cleaned_text_back    MEDIUMTEXT   NULL,
      lines_json_back      MEDIUMTEXT   NULL,
      has_back             TINYINT(1)   NOT NULL DEFAULT 0,
      engine               VARCHAR(32)  NOT NULL DEFAULT '',
      ocr_status           VARCHAR(20)  NOT NULL DEFAULT 'pending',
      is_empty             TINYINT(1)   NOT NULL DEFAULT 0,
      error                VARCHAR(500) NOT NULL DEFAULT '',
      file_size_bytes      INT UNSIGNED NOT NULL DEFAULT 0,
      processing_time_seconds FLOAT     NOT NULL DEFAULT 0,
      usage_count          INT UNSIGNED NOT NULL DEFAULT 0,
      created_at           VARCHAR(32)  NOT NULL,
      updated_at           VARCHAR(32)  NOT NULL,
      last_used_at         VARCHAR(32)  NOT NULL DEFAULT '',
      UNIQUE KEY uq_ref_library_sha256 (sha256),
      KEY idx_ref_library_template (template_id),
      KEY idx_ref_library_created (created_at DESC),
      KEY idx_ref_library_used (last_used_at DESC),
      KEY idx_ref_library_text_hash (text_hash)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,
    """
    CREATE TABLE IF NOT EXISTS ref_blobs (
      ref_id  VARCHAR(64) NOT NULL PRIMARY KEY,
      image   MEDIUMBLOB  NOT NULL,
      thumb   BLOB        NOT NULL,
      image_back MEDIUMBLOB NULL,
      thumb_back BLOB NULL,
      CONSTRAINT fk_ref_blobs_ref FOREIGN KEY (ref_id)
        REFERENCES ref_library (ref_id) ON DELETE CASCADE
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,
]

# Kolom ringan (TANPA blob) — dipakai get_ref agar konsisten & cepat
# REVIEW_FLOW_COMPARE_FRONT_BACK_PLAN (2026-10-05): + kolom front/back.
# FIX_REFERENCE_OCR_ACUAN_PER_LABEL_PLAN (2026-10-07): + kolom side.
LIGHT_COLS = (
    "ref_id, name, template_id, side, source, user_ref, sha256, text_hash, "
    "cleaned_text, lines_json, "
    "cleaned_text_front, lines_json_front, cleaned_text_back, lines_json_back, has_back, "
    "engine, ocr_status, is_empty, error, "
    "file_size_bytes, processing_time_seconds, usage_count, "
    "created_at, updated_at, last_used_at"
)

# Fallback bila DB belum dimigrasi (kolom front/back belum ada) → SELECT lama tetap jalan.
LEGACY_LIGHT_COLS = (
    "ref_id, name, template_id, source, user_ref, sha256, text_hash, "
    "cleaned_text, lines_json, engine, ocr_status, is_empty, error, "
    "file_size_bytes, processing_time_seconds, usage_count, "
    "created_at, updated_at, last_used_at"
)

# Kolom baru front/back + side (dipakai untuk fallback 1054 unknown column).
_DUAL_SIDE_COLS = {
    "cleaned_text_front", "lines_json_front",
    "cleaned_text_back", "lines_json_back", "has_back",
    "side",
}


def _is_unknown_column(e: Exception) -> bool:
    """True bila error MySQL 1054 (unknown column) — DB belum migrasi dual-side."""
    return getattr(e, "errno", None) == 1054

_pool: Optional[MySQLConnectionPool] = None
_pool_created_at: float = 0.0
# Recycle manual tiap 3600 dtk (< wait_timeout default 28800) agar koneksi
# basi tidak menumpuk. (mysql-connector 8.4 pooling tak kenal arg pool_recycle,
# jadi dibuat ulang pool-nya; koneksi lama mati alami. Retry 2006/2013 tetap ada.)
POOL_RECYCLE_SECONDS = 3600


def _get_pool() -> MySQLConnectionPool:
    """Pool global, dibuat 1x + recycle berkala (lazy agar import tidak gagal saat MySQL mati)."""
    global _pool, _pool_created_at
    now = time.time()
    if _pool is None or (now - _pool_created_at) > POOL_RECYCLE_SECONDS:
        _pool = MySQLConnectionPool(
            pool_name="ref_library",
            pool_size=max(1, MYSQL_POOL_SIZE),
            connection_timeout=5,
            host=MYSQL_HOST,
            port=MYSQL_PORT,
            database=MYSQL_DB,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            charset="utf8mb4",
            collation="utf8mb4_unicode_ci",
            autocommit=False,
        )
        _pool_created_at = now
    return _pool


T = TypeVar("T")


def _with_conn(fn: Callable) -> Callable:
    """Decorator: pinjam koneksi pool → panggil fn(conn, ...) → close di finally."""

    def wrapper(*args, **kwargs):
        try:
            conn = _get_pool().get_connection()
        except MySQLError as e:
            logger.warning(f"[MYSQL] pool get_connection gagal: {e}")
            raise DBUnavailableError(str(e)) from e
        try:
            return fn(conn, *args, **kwargs)
        finally:
            try:
                conn.close()  # kembali ke pool, bukan tutup TCP
            except Exception:
                pass

    return wrapper


def _is_retryable(e: MySQLError) -> bool:
    return getattr(e, "errno", None) in RETRY_ERRNOS


def _now_iso() -> str:
    return datetime.now(JAKARTA_TZ).isoformat()


def _row_to_dict(row: dict) -> dict:
    """Konversi dict-row MySQL -> bentuk SAMA seperti db SQLite (kunci user_id, lines)."""
    d = dict(row)
    raw_lines = d.pop("lines_json", "[]")
    try:
        parsed = json.loads(raw_lines) if raw_lines else []
        d["lines"] = parsed if isinstance(parsed, list) else []
    except (json.JSONDecodeError, TypeError):
        d["lines"] = []
    # REVIEW_FLOW_COMPARE_FRONT_BACK_PLAN (2026-10-05): parse lines per sisi.
    for _lj_key, _l_key in (("lines_json_front", "lines_front"), ("lines_json_back", "lines_back")):
        if _lj_key in d:
            _raw = d.pop(_lj_key, "[]")
            # MEDIUMTEXT bisa return bytearray/bytes
            if isinstance(_raw, (bytearray, bytes)):
                try:
                    _raw = bytes(_raw).decode("utf-8", errors="replace")
                except Exception:
                    _raw = "[]"
            try:
                _parsed = json.loads(_raw) if _raw else []
                d[_l_key] = _parsed if isinstance(_parsed, list) else []
            except (json.JSONDecodeError, TypeError):
                d[_l_key] = []
    # Kompat: SQLite pakai kunci user_id; MySQL pakai kolom user_ref
    if "user_ref" in d:
        d["user_id"] = d.pop("user_ref")
    d["is_empty"] = bool(d.get("is_empty", 0))
    d["usage_count"] = int(d.get("usage_count", 0) or 0)
    d["file_size_bytes"] = int(d.get("file_size_bytes", 0) or 0)
    # has_back → bool
    if "has_back" in d:
        try:
            d["has_back"] = bool(int(d.get("has_back", 0) or 0))
        except (TypeError, ValueError):
            d["has_back"] = bool(d.get("has_back", False))
    try:
        d["processing_time_seconds"] = float(d.get("processing_time_seconds", 0) or 0)
    except (TypeError, ValueError):
        d["processing_time_seconds"] = 0.0
    # MySQL bisa return None untuk kolom teks kosong
    for k in ("template_id", "side", "source", "text_hash", "engine", "ocr_status",
              "error", "created_at", "updated_at", "last_used_at", "name", "sha256"):
        if d.get(k) is None:
            d[k] = ""
    if d.get("cleaned_text") is None:
        d["cleaned_text"] = ""
    # Kolom front/back None → "" agar konsisten; MEDIUMTEXT bisa bytearray
    for k in ("cleaned_text_front", "cleaned_text_back"):
        if k in d:
            if d.get(k) is None:
                d[k] = ""
            elif isinstance(d[k], (bytearray, bytes)):
                try:
                    d[k] = bytes(d[k]).decode("utf-8", errors="replace")
                except Exception:
                    d[k] = ""
    return d


def ping() -> float:
    """Ping DB untuk /health. Return latency ms. Raise DBUnavailableError jika gagal."""
    start = time.time()
    try:
        conn = _get_pool().get_connection()
    except MySQLError as e:
        raise DBUnavailableError(str(e)) from e
    try:
        cur = conn.cursor()
        try:
            cur.execute("SELECT 1")
            cur.fetchone()
        finally:
            cur.close()
    finally:
        try:
            conn.close()
        except Exception:
            pass
    return round((time.time() - start) * 1000, 2)


def init_db() -> None:
    """Buat tabel + index. Idempotent, aman dipanggil tiap startup."""
    try:
        conn = _get_pool().get_connection()
    except MySQLError as e:
        raise DBUnavailableError(str(e)) from e
    try:
        cur = conn.cursor()
        try:
            for stmt in SCHEMA_SQL:
                cur.execute(stmt)
            conn.commit()
        finally:
            cur.close()
    finally:
        try:
            conn.close()
        except Exception:
            pass


@_with_conn
def _insert_once(conn, row: dict) -> None:
    """1x percobaan INSERT transaksi (row + blob). Dipanggil via retry wrapper."""
    cur = conn.cursor()
    try:
        # REVIEW_FLOW_COMPARE_FRONT_BACK_PLAN: INSERT mencakup kolom front/back.
        # Fallback ke INSERT lama bila DB belum migrasi (1054 unknown column).
        try:
            cur.execute(
                """
                INSERT INTO ref_library (
                  ref_id, name, template_id, side, source, user_ref, sha256, text_hash,
                  cleaned_text, lines_json,
                  cleaned_text_front, lines_json_front, cleaned_text_back, lines_json_back, has_back,
                  engine, ocr_status, is_empty, error,
                  file_size_bytes, processing_time_seconds, usage_count,
                  created_at, updated_at, last_used_at
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """,
                (
                    row.get("ref_id", ""),
                    row.get("name", ""),
                    row.get("template_id", "") or "",
                    (row.get("side", "") or "").strip().lower(),
                    row.get("source", "") or "",
                    str(row.get("user_id", "") or row.get("user_ref", "") or ""),
                    row.get("sha256", ""),
                    row.get("text_hash", "") or "",
                    row.get("cleaned_text", "") or "",
                    row.get("lines_json", "[]") or "[]",
                    row.get("cleaned_text_front", "") or "",
                    row.get("lines_json_front", "[]") or "[]",
                    row.get("cleaned_text_back", "") or "",
                    row.get("lines_json_back", "[]") or "[]",
                    1 if row.get("has_back") else 0,
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
        except MySQLError as e_new:
            if not _is_unknown_column(e_new):
                raise
            # DB lama: ulangi dengan kolom lama saja
            logger.warning("[MYSQL] insert fallback legacy cols (DB belum migrasi dual-side)")
            cur.execute(
                """
                INSERT INTO ref_library (
                  ref_id, name, template_id, source, user_ref, sha256, text_hash,
                  cleaned_text, lines_json, engine, ocr_status, is_empty, error,
                  file_size_bytes, processing_time_seconds, usage_count,
                  created_at, updated_at, last_used_at
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """,
                (
                    row.get("ref_id", ""),
                    row.get("name", ""),
                    row.get("template_id", "") or "",
                    row.get("source", "") or "",
                    str(row.get("user_id", "") or row.get("user_ref", "") or ""),
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
        cur.execute(
            "INSERT INTO ref_blobs (ref_id, image, thumb) VALUES (%s,%s,%s)",
            (
                row.get("ref_id", ""),
                row.get("image_bytes", b"") or b"",
                row.get("thumb_bytes", b"") or b"",
            ),
        )
        conn.commit()
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        raise
    finally:
        cur.close()


def insert_ref(row: dict) -> None:
    """
    Insert 1 reference (row + blob, 1 transaksi).
    row: ref_id, name, sha256, image_bytes, thumb_bytes, ... (sisanya opsional).
    Raise DuplicateRefError jika sha256 sudah ada (bukan 500).
    Raise DBUnavailableError jika DB mati.
    """
    try:
        _insert_once(row)
    except MySQLError as e:
        if getattr(e, "errno", None) == DUPLICATE_ERRNO:
            # Race dengan dedup lapis 1 → ambil existing untuk respons duplicate:true
            existing = find_by_sha256(row.get("sha256", ""))
            raise DuplicateRefError(existing["ref_id"] if existing else "") from e
        if _is_retryable(e):
            logger.warning(f"[MYSQL] insert retryable ({e}), 1x retry...")
            try:
                _insert_once(row)
                return
            except MySQLError as e2:
                if getattr(e2, "errno", None) == DUPLICATE_ERRNO:
                    existing = find_by_sha256(row.get("sha256", ""))
                    raise DuplicateRefError(existing["ref_id"] if existing else "") from e2
                logger.warning(f"[MYSQL] insert gagal setelah retry: {e2}")
                raise DBUnavailableError(str(e2)) from e2
        logger.warning(f"[MYSQL] insert gagal: {e}")
        raise DBUnavailableError(str(e)) from e


@_with_conn
def _get_impl(conn, ref_id: str) -> Optional[dict]:
    cur = conn.cursor(dictionary=True)
    try:
        try:
            cur.execute(f"SELECT {LIGHT_COLS} FROM ref_library WHERE ref_id = %s", (ref_id,))
        except MySQLError as e_sel:
            if not _is_unknown_column(e_sel):
                raise
            # DB belum migrasi → fallback kolom lama
            cur.execute(f"SELECT {LEGACY_LIGHT_COLS} FROM ref_library WHERE ref_id = %s", (ref_id,))
        row = cur.fetchone()
        if row is None:
            return None
        return _row_to_dict(row)
    finally:
        cur.close()


def get_ref(ref_id: str) -> Optional[dict]:
    """Ambil 1 reference by id (TANPA blob). Return None jika tidak ada."""
    try:
        return _get_impl(ref_id)
    except MySQLError as e:
        if _is_retryable(e):
            try:
                return _get_impl(ref_id)
            except MySQLError as e2:
                raise DBUnavailableError(str(e2)) from e2
        raise DBUnavailableError(str(e)) from e


@_with_conn
def _find_sha_impl(conn, sha: str) -> Optional[dict]:
    cur = conn.cursor(dictionary=True)
    try:
        try:
            cur.execute(f"SELECT {LIGHT_COLS} FROM ref_library WHERE sha256 = %s LIMIT 1", (sha,))
        except MySQLError as e_sel:
            if not _is_unknown_column(e_sel):
                raise
            cur.execute(f"SELECT {LEGACY_LIGHT_COLS} FROM ref_library WHERE sha256 = %s LIMIT 1", (sha,))
        row = cur.fetchone()
        if row is None:
            return None
        return _row_to_dict(row)
    finally:
        cur.close()


def find_by_sha256(sha: str) -> Optional[dict]:
    """Cari reference dengan SHA256 gambar persis (dedup lapis 1)."""
    if not sha:
        return None
    try:
        return _find_sha_impl(sha)
    except MySQLError as e:
        if _is_retryable(e):
            try:
                return _find_sha_impl(sha)
            except MySQLError as e2:
                raise DBUnavailableError(str(e2)) from e2
        raise DBUnavailableError(str(e)) from e


@_with_conn
def _get_blob_impl(conn, ref_id: str) -> Optional[dict]:
    """Ambil bytes image+thumb (HANYA untuk serve /image|/thumb — bukan list)."""
    cur = conn.cursor(dictionary=True)
    try:
        cur.execute("SELECT image, thumb FROM ref_blobs WHERE ref_id = %s", (ref_id,))
        row = cur.fetchone()
        if row is None:
            return None
        img = row.get("image") or b""
        th = row.get("thumb") or b""
        return {"image": bytes(img), "thumb": bytes(th)}
    finally:
        cur.close()


def get_blob(ref_id: str) -> Optional[dict]:
    """Return {"image": bytes, "thumb": bytes} atau None."""
    try:
        return _get_blob_impl(ref_id)
    except MySQLError as e:
        if _is_retryable(e):
            try:
                return _get_blob_impl(ref_id)
            except MySQLError as e2:
                raise DBUnavailableError(str(e2)) from e2
        raise DBUnavailableError(str(e)) from e


@_with_conn
def _list_impl(conn, where_sql: str, params: list, order_sql: str,
               per_page: int, offset: int) -> tuple[list[dict], int]:
    cur = conn.cursor(dictionary=True)
    try:
        cur.execute(f"SELECT COUNT(*) AS c FROM ref_library {where_sql}", params)
        total = int((cur.fetchone() or {}).get("c") or 0)
        cur.execute(
            f"""
            SELECT ref_id, name, template_id, ocr_status, cleaned_text,
                   usage_count, created_at, last_used_at, sha256
            FROM ref_library {where_sql} {order_sql} LIMIT %s OFFSET %s
            """,
            (*params, per_page, offset),
        )
        items: list[dict] = []
        for r in cur.fetchall():
            ct = r.get("cleaned_text") or ""
            # MEDIUMTEXT bisa return bytearray di beberapa versi driver
            if isinstance(ct, (bytearray, bytes)):
                ct = bytes(ct).decode("utf-8", errors="replace")
            excerpt = " ".join(ct.split())[:120]
            items.append(
                {
                    "ref_id": r.get("ref_id", ""),
                    "name": r.get("name", "") or "",
                    "template_id": r.get("template_id", "") or "",
                    "ocr_status": r.get("ocr_status", "") or "pending",
                    "excerpt": excerpt,
                    "usage_count": int(r.get("usage_count", 0) or 0),
                    "created_at": r.get("created_at", "") or "",
                    "last_used_at": r.get("last_used_at", "") or "",
                    "sha256": r.get("sha256", "") or "",
                }
            )
        return items, total
    finally:
        cur.close()


def list_refs(
    q: str = "",
    template_id: str = "",
    ocr_status: str = "",
    sort: str = "recent_used",
    page: int = 1,
    per_page: int = 20,
) -> tuple[list[dict], int]:
    """
    List ringan untuk dropdown Laravel (tanpa lines full, hanya excerpt; TANPA blob).
    Return (items, total_count). Bentuk items SAMA seperti db SQLite.
    """
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
        where_parts.append("(name LIKE %s ESCAPE '\\\\' OR cleaned_text LIKE %s ESCAPE '\\\\')")
        esc = qq.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        like = f"%{esc}%"
        params.extend([like, like])

    if template_id:
        where_parts.append("template_id = %s")
        params.append(template_id)

    if ocr_status in ("pending", "success", "failed"):
        where_parts.append("ocr_status = %s")
        params.append(ocr_status)

    where_sql = ("WHERE " + " AND ".join(where_parts)) if where_parts else ""

    if sort == "newest":
        order_sql = "ORDER BY created_at DESC"
    elif sort == "name":
        order_sql = "ORDER BY name ASC, created_at DESC"
    else:  # recent_used (default)
        order_sql = "ORDER BY CASE WHEN last_used_at = '' THEN '' ELSE last_used_at END DESC, created_at DESC"

    offset = (page - 1) * per_page

    try:
        return _list_impl(where_sql, params, order_sql, per_page, offset)
    except MySQLError as e:
        if _is_retryable(e):
            try:
                return _list_impl(where_sql, params, order_sql, per_page, offset)
            except MySQLError as e2:
                raise DBUnavailableError(str(e2)) from e2
        raise DBUnavailableError(str(e)) from e


@_with_conn
def _update_impl(conn, sets_sql: str, params: list) -> bool:
    cur = conn.cursor()
    try:
        cur.execute(f"UPDATE ref_library SET {sets_sql} WHERE ref_id = %s", params)
        conn.commit()
        return cur.rowcount > 0
    finally:
        cur.close()


def update_ref(ref_id: str, fields: dict) -> bool:
    """
    Update parsial kolom yang diizinkan (SAMA seperti SQLite).
    REVIEW_FLOW_COMPARE_FRONT_BACK_PLAN: + kolom front/back/has_back.
    Return True jika ada row terupdate.
    """
    allowed = {
        "name",
        "template_id",
        "side",
        "source",
        "user_id",  # dipetakan ke kolom user_ref
        "text_hash",
        "cleaned_text",
        "lines_json",
        "cleaned_text_front",
        "lines_json_front",
        "cleaned_text_back",
        "lines_json_back",
        "has_back",
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
        col = "user_ref" if k == "user_id" else k
        if k == "is_empty":
            sets.append("is_empty = %s")
            params.append(1 if v else 0)
        elif k == "has_back":
            sets.append("has_back = %s")
            params.append(1 if v else 0)
        else:
            sets.append(f"{col} = %s")
            params.append(v)
    if not sets:
        return False
    if "updated_at" not in (fields or {}):
        sets.append("updated_at = %s")
        params.append(_now_iso())
    params.append(ref_id)

    try:
        return _update_impl(", ".join(sets), params)
    except MySQLError as e:
        # DB belum migrasi → coba lagi tanpa kolom baru
        if _is_unknown_column(e):
            filt_sets: list[str] = []
            filt_params: list = []
            # params terakhir = ref_id; sets sejajar params[:-1]
            for s, p in zip(sets, params[:-1]):
                _col = s.split("=")[0].strip().strip("`")
                if _col in _DUAL_SIDE_COLS:
                    continue
                filt_sets.append(s)
                filt_params.append(p)
            if not filt_sets:
                return False
            filt_params.append(ref_id)
            try:
                return _update_impl(", ".join(filt_sets), filt_params)
            except MySQLError as e2:
                raise DBUnavailableError(str(e2)) from e2
        if _is_retryable(e):
            try:
                return _update_impl(", ".join(sets), params)
            except MySQLError as e2:
                raise DBUnavailableError(str(e2)) from e2
        raise DBUnavailableError(str(e)) from e


@_with_conn
def _delete_impl(conn, ref_id: str) -> None:
    cur = conn.cursor()
    try:
        # Blob ikut terhapus via ON DELETE CASCADE (1 query, atomic)
        cur.execute("DELETE FROM ref_library WHERE ref_id = %s", (ref_id,))
        conn.commit()
    finally:
        cur.close()


def delete_ref(ref_id: str) -> Optional[dict]:
    """Hapus 1 row (+blob cascade). Return row lama atau None."""
    old = get_ref(ref_id)
    if old is None:
        return None
    try:
        _delete_impl(ref_id)
    except MySQLError as e:
        if _is_retryable(e):
            try:
                _delete_impl(ref_id)
            except MySQLError as e2:
                raise DBUnavailableError(str(e2)) from e2
        else:
            raise DBUnavailableError(str(e)) from e
    return old


@_with_conn
def _count_impl(conn) -> tuple[int, int]:
    cur = conn.cursor(dictionary=True)
    try:
        cur.execute("SELECT COUNT(*) AS c, COALESCE(SUM(file_size_bytes),0) AS b FROM ref_library")
        row = cur.fetchone() or {}
        return int(row.get("c") or 0), int(row.get("b") or 0)
    finally:
        cur.close()


def count_and_bytes() -> tuple[int, int]:
    """Return (jumlah_rows, total_file_size_bytes). Untuk cek quota."""
    try:
        return _count_impl()
    except MySQLError as e:
        if _is_retryable(e):
            try:
                return _count_impl()
            except MySQLError as e2:
                raise DBUnavailableError(str(e2)) from e2
        raise DBUnavailableError(str(e)) from e


@_with_conn
def _touch_impl(conn, ref_id: str) -> None:
    cur = conn.cursor()
    try:
        cur.execute(
            "UPDATE ref_library SET usage_count = usage_count + 1, last_used_at = %s WHERE ref_id = %s",
            (_now_iso(), ref_id),
        )
        conn.commit()
    finally:
        cur.close()


def touch_used(ref_id: str) -> None:
    """Naikkan usage_count + refresh last_used_at. Best-effort, jangan raise."""
    try:
        _touch_impl(ref_id)
    except Exception:
        pass
