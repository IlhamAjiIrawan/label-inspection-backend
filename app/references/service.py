"""
app/references/service.py — Helper murni untuk Reference Library.

Tanpa FastAPI / DB. Hanya: hash, normalisasi teks, kompres + thumbnail.
Lihat plans/PLAN_REFERENCE_LIBRARY_FASTAPI.md §5-T2.
"""

import hashlib
import io
import re
from pathlib import Path

from PIL import Image, ImageOps


def sha256_bytes(data: bytes) -> str:
    """SHA256 hex dari bytes gambar asli (untuk dedup lapis 1)."""
    return hashlib.sha256(data).hexdigest()


def normalize_text(s: str) -> str:
    """
    Normalisasi teks OCR untuk text_hash yang stabil.

    - lower, strip tiap baris, buang baris kosong
    - collapse whitespace berlebih per baris
    - join dengan "\\n"
    """
    if not s or not isinstance(s, str):
        return ""
    lines = []
    for raw_line in s.splitlines():
        # collapse whitespace: "  TEH   BOTOL  " -> "teh botol"
        collapsed = re.sub(r"\s+", " ", raw_line.strip()).strip().lower()
        if collapsed:
            lines.append(collapsed)
    return "\n".join(lines)


def text_hash(normalized: str) -> str:
    """
    SHA256 dari teks ternormalisasi. Return "" jika kosong / "none".

    "none" adalah kontrak VALID tidak ada teks (bukan error), tapi tidak
    berguna untuk dedup teks — jadi kembalikan "" agar tidak dikelompokkan.
    """
    if not normalized:
        return ""
    if normalized.strip().lower() == "none":
        return ""
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def compress_and_save(src_bytes: bytes, dest_path: Path, max_dim: int, quality: int) -> int:
    """
    Kompres bytes gambar upload -> JPEG di dest_path.

    - Buka via PIL, exif_transpose (strip orientasi), convert RGB
    - Resize sisi terpanjang max_dim (LANCZOS), hanya jika lebih besar
    - Simpan JPEG quality; selalu JPEG (abaikan ext client)
    - Return ukuran file hasil (bytes)

    Raises:
        ValueError: jika bytes bukan gambar valid.
        OSError: jika gagal tulis disk.
    """
    try:
        img = Image.open(io.BytesIO(src_bytes))
    except Exception as e:
        raise ValueError(f"Invalid image bytes: {e}") from e

    try:
        # Strip EXIF orientasi + paksa RGB (JPEG tidak support RGBA/P)
        try:
            img = ImageOps.exif_transpose(img)
        except Exception:
            pass
        if img.mode in ("RGBA", "LA", "PA", "P"):
            # Composite ke putih agar transparan tidak jadi hitam
            bg = Image.new("RGB", img.size, (255, 255, 255))
            try:
                bg.paste(img, mask=img.split()[-1])
                img = bg
            except Exception:
                img = img.convert("RGB")
        elif img.mode != "RGB":
            img = img.convert("RGB")

        w, h = img.size
        longest = max(w, h)
        if longest > max_dim and longest > 0:
            # thumbnail() preserve aspect ratio, in-place
            img.thumbnail((max_dim, max_dim), Image.LANCZOS)

        dest_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(str(dest_path), format="JPEG", quality=int(quality), optimize=True)
    finally:
        try:
            img.close()
        except Exception:
            pass

    try:
        return dest_path.stat().st_size
    except OSError:
        return 0


def compress_to_bytes(src_bytes: bytes, max_dim: int, quality: int) -> bytes:
    """
    Varian memory dari compress_and_save (PLAN_MYSQL_BACKED_REFERENCE_STORAGE F3).

    Logika SAMA (exif, RGB, resize LANCZOS, JPEG q) — hanya output bytes
    (via BytesIO) agar FastAPI tidak tulis disk. Fungsi lama TIDAK diubah
    (tetap dipakai mode sqlite selama cutover).
    """
    try:
        img = Image.open(io.BytesIO(src_bytes))
    except Exception as e:
        raise ValueError(f"Invalid image bytes: {e}") from e

    try:
        try:
            img = ImageOps.exif_transpose(img)
        except Exception:
            pass
        if img.mode in ("RGBA", "LA", "PA", "P"):
            bg = Image.new("RGB", img.size, (255, 255, 255))
            try:
                bg.paste(img, mask=img.split()[-1])
                img = bg
            except Exception:
                img = img.convert("RGB")
        elif img.mode != "RGB":
            img = img.convert("RGB")

        w, h = img.size
        longest = max(w, h)
        if longest > max_dim and longest > 0:
            img.thumbnail((max_dim, max_dim), Image.LANCZOS)

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=int(quality), optimize=True)
        return buf.getvalue()
    finally:
        try:
            img.close()
        except Exception:
            pass


def make_thumb_bytes(jpeg_bytes: bytes, max_dim: int, quality: int) -> bytes:
    """
    Varian memory dari make_thumb (F3). Input JPEG hasil compress_to_bytes,
    output thumbnail JPEG bytes. Fungsi lama TIDAK diubah.
    """
    try:
        img = Image.open(io.BytesIO(jpeg_bytes))
    except Exception as e:
        raise ValueError(f"Invalid image bytes: {e}") from e
    try:
        try:
            img = ImageOps.exif_transpose(img)
        except Exception:
            pass
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.thumbnail((max_dim, max_dim), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=int(quality), optimize=True)
        return buf.getvalue()
    finally:
        try:
            img.close()
        except Exception:
            pass


def make_thumb(src_path: Path, thumb_path: Path, max_dim: int, quality: int) -> int:
    """
    Buat thumbnail JPEG kecil dari file hasil kompres.

    - Buka src_path, thumbnail ke max_dim, simpan JPEG quality
    - Return ukuran file thumb (bytes)
    """
    img = Image.open(str(src_path))
    try:
        try:
            img = ImageOps.exif_transpose(img)
        except Exception:
            pass
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.thumbnail((max_dim, max_dim), Image.LANCZOS)
        thumb_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(str(thumb_path), format="JPEG", quality=int(quality), optimize=True)
    finally:
        try:
            img.close()
        except Exception:
            pass

    try:
        return thumb_path.stat().st_size
    except OSError:
        return 0
