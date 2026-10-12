"""
app/compare/gemini_client.py — Client untuk Gemini 1.5 Flash (Visual Comparison).

Menggunakan google-genai SDK resmi untuk:
1. Mengirim gambar multimodal (reference + check images) dalam 1 request
2. Menerima response JSON dari Gemini
3. Handle error dan timeout

Dependencies: google-genai>=1.0.0
"""

import json
import logging
import time
from pathlib import Path

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)


def create_gemini_client(api_key: str) -> genai.Client:
    """
    Buat instance Gemini client.

    Args:
        api_key: API key dari Google AI Studio.

    Returns:
        genai.Client: Client yang siap digunakan.

    Raises:
        ValueError: Jika api_key kosong.
    """
    if not api_key or not api_key.strip():
        raise ValueError("GEMINI_API_KEY tidak boleh kosong. Set di .env atau environment variable.")

    client = genai.Client(api_key=api_key)
    return client


def load_image_as_part(image_path: str) -> types.Part:
    """
    Load gambar dari disk dan convert ke format Part untuk Gemini multimodal input.

    Args:
        image_path: Path absolut ke file gambar (PNG/JPEG).

    Returns:
        types.Part: Part object yang berisi inline image data.

    Raises:
        FileNotFoundError: Jika file tidak ditemukan.
        ValueError: Jika format file tidak didukung.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"File gambar tidak ditemukan: {image_path}")

    # Tentukan MIME type berdasarkan ekstensi
    ext = path.suffix.lower()
    mime_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
    }
    mime_type = mime_map.get(ext)
    if not mime_type:
        raise ValueError(f"Format file tidak didukung: {ext}. Hanya PNG dan JPEG.")

    # Baca file bytes
    image_bytes = path.read_bytes()

    # Buat Part dengan inline data
    part = types.Part.from_bytes(
        data=image_bytes,
        mime_type=mime_type,
    )
    return part


def call_gemini_comparison(
    client: genai.Client,
    model: str,
    prompt: str,
    reference_image_path: str,
    check_image_paths: list[str],
    timeout: float = 30.0,
    retries: int = 3,
) -> dict:
    """
    Kirim prompt + gambar multimodal ke Gemini dan parse response JSON.

    Mengirim 1 request dengan semua gambar sekaligus (1 reference + N check).
    Ini lebih efisien daripada mengirim 1 request per gambar.

    Robustness (temuan uji 2026-09-10): Gemini kadang mengembalikan JSON cacat
    atau 503 saat spike demand — keduanya transien, maka generate+parse di-retry
    hingga `retries` kali. Error config (404 model tidak ada) TIDAK di-retry.

    Args:
        client: Gemini client instance.
        model: Model ID (e.g. "gemini-3.8-flash").
        prompt: Prompt instruksi comparison (dari prompt.py).
        reference_image_path: Path ke gambar reference.
        check_image_paths: List path ke gambar-gambar check.
        timeout: Timeout dalam detik.
        retries: Maksimal attempt generate+parse (default 3).

    Returns:
        dict: Parsed JSON response dari Gemini.
              Format: {"results": [{slot, file_name, is_match, confidence_score, notes}, ...]}

    Raises:
        TimeoutError: Jika request melebihi timeout.
        ValueError: Jika response bukan JSON valid (setelah retries habis).
        Exception: Error lain dari Gemini API.
    """
    start = time.time()

    # Build multimodal content: [prompt_text, ref_image, check_1, check_2, ...]
    contents = []

    # 1. Tambahkan prompt teks
    contents.append(prompt)

    # 2. Tambahkan gambar reference (PERTAMA)
    ref_part = load_image_as_part(reference_image_path)
    contents.append(ref_part)

    # 3. Tambahkan gambar check (berurutan)
    for check_path in check_image_paths:
        check_part = load_image_as_part(check_path)
        contents.append(check_part)

    logger.info(
        f"[COMPARE] Sending {1 + len(check_image_paths)} images to Gemini ({model}), "
        f"timeout={timeout}s, retries={retries}"
    )

    last_error: Exception | None = None
    # NOTE: Jangan pakai client.models.generate_content langsung — SDK google-genai
    # (>=1.x) me-log warning AFC untuk path itu:
    #   "Direct use of automatic function calling (AFC) in Models.generate_content
    #    is not recommended. Instead, we recommend to use AFC in Chat.send_message."
    # Rekomendasi resmi: pakai Chat.send_message. Kita juga tidak memakai function
    # calling/tools sama sekali, jadi parameter automatic_function_calling TIDAK
    # diteruskan — tanpa tools AFC memang tidak berjalan. Meneruskan
    # AutomaticFunctionCallingConfig(disable=True) justru memicu warning SDK
    # ("disable is True but maximum_remote_calls is positive") karena default
    # maximum_remote_calls=10 ikut terkirim.
    gen_config = types.GenerateContentConfig(
        temperature=0.1,            # Low temperature untuk konsistensi
        max_output_tokens=2048,     # Cukup untuk 5 slot JSON
        response_mime_type="application/json",  # Force JSON response
    )
    for attempt in range(1, retries + 1):
        # 4. Kirim ke Gemini via Chat API (fresh chat per attempt agar
        #    retry tidak menumpuk history di chat yang sama)
        try:
            chat = client.chats.create(model=model, config=gen_config)
            response = chat.send_message(message=contents)
        except Exception as e:
            elapsed = time.time() - start
            # 404 model-tidak-ada = error config → jangan retry, gagal cepat
            if "404" in str(e) and "NOT_FOUND" in str(e):
                logger.error(f"[COMPARE] Gemini API config error (no-retry): {e}")
                raise
            last_error = e
            logger.warning(f"[COMPARE] Gemini API error attempt {attempt}/{retries} after {elapsed:.2f}s: {e}")
            if attempt < retries:
                time.sleep(2 * attempt)
                continue
            logger.error(f"[COMPARE] Gemini API error after {elapsed:.2f}s: {e}", exc_info=True)
            raise

        elapsed = time.time() - start
        logger.info(f"[COMPARE] Gemini response received in {elapsed:.2f}s (attempt {attempt})")

        # 5. Parse response
        if not response.text:
            last_error = ValueError("Gemini mengembalikan response kosong (empty text)")
            logger.warning(f"[COMPARE] Empty response attempt {attempt}/{retries}")
            if attempt < retries:
                time.sleep(2 * attempt)
                continue
            raise last_error

        raw_text = response.text.strip()

        # Gemini kadang wrap JSON dalam ```json ... ``` — strip jika ada
        if raw_text.startswith("```"):
            # Hapus opening ``` dan closing ```
            lines = raw_text.splitlines()
            # Hapus baris pertama (```json) dan terakhir (```)
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            raw_text = "\n".join(lines)

        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError as e:
            last_error = ValueError(f"Response Gemini bukan JSON valid: {e}")
            logger.warning(f"[COMPARE] Parse gagal attempt {attempt}/{retries}: {e} | raw[:300]={raw_text[:300]!r}")
            if attempt < retries:
                time.sleep(2 * attempt)
                continue
            logger.error(f"[COMPARE] Failed to parse Gemini JSON after {retries}x: {raw_text[:500]}")
            raise last_error from e

        if attempt > 1:
            logger.info(f"[COMPARE] Sukses pada attempt {attempt}/{retries}")
        return parsed

    # Seharusnya tidak sampai sini; pengaman jika loop selesai tanpa return/raise
    assert last_error is not None
    raise last_error
