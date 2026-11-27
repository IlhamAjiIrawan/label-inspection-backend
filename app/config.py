"""
app/config.py — Konstanta konfigurasi global.

Disimpan terpisah agar junior programmer / agent murah mudah menemukan
nilai yang perlu diubah tanpa menyentuh logika bisnis.

Semua nilai di sini CPU-only, tanpa GPU.
"""

import os
from pathlib import Path

# Load .env jika ada (opsional, tidak error jika python-dotenv belum install)
try:
    from dotenv import load_dotenv

    # Cari .env di root project (parent dari app/)
    _env_path = Path(__file__).resolve().parent.parent / ".env"
    if _env_path.exists():
        load_dotenv(dotenv_path=_env_path, override=False)
except ImportError:
    # Fallback parser mandiri jika python-dotenv belum terpasang
    _env_path = Path(__file__).resolve().parent.parent / ".env"
    if _env_path.exists():
        try:
            with open(_env_path, "r", encoding="utf-8") as _f:
                for _line in _f:
                    _line = _line.strip()
                    if not _line or _line.startswith("#") or "=" not in _line:
                        continue
                    _k, _v = _line.split("=", 1)
                    _k = _k.strip()
                    _v = _v.strip().strip("'\"")
                    if _k not in os.environ:
                        os.environ[_k] = _v
        except Exception:
            pass

# --- Upload & Storage ---
UPLOAD_DIR = Path(__file__).resolve().parent.parent / "uploads"
OCR_DIR = UPLOAD_DIR / "ocr"
METADATA_FILE = UPLOAD_DIR / "metadata.json"

# Pastikan folder ada saat import (idempotent)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OCR_DIR.mkdir(parents=True, exist_ok=True)

# --- Keamanan Upload ---
MAX_UPLOAD_SIZE = 5 * 1024 * 1024  # 5 MB dalam bytes (5.242.880)

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg"}

# Magic bytes untuk validasi konten file (bukan hanya ekstensi)
# PNG:  89 50 4E 47 0D 0A 1A 0A  (8 bytes)
# JPEG: FF D8 FF                (3 bytes)
MAGIC_BYTES = {
    "png": b"\x89PNG\r\n\x1a\n",
    "jpeg": b"\xff\xd8\xff",
}

# Rate limiting: maksimal 1 gambar per 60 detik (POST upload) — LEGACY fallback.
# FIX 429 dual-side (2026-09-15): digantikan burst sliding-window di bawah.
# RATE_LIMIT_SECONDS dipertahankan agar kode/test lama tidak rusak.
RATE_LIMIT_SECONDS = 60

# --- Rate limit POST capture burst (FIX 429 dual-side front+back, 2026-09-15) ---
# Dulu: 1 gambar/menit hardcode (RATE_LIMIT_SECONDS=60) → BACK selalu 429.
# Sekarang: burst sliding-window, default 5 req / 60s (via env, BUTUH RESTART).
# Alasan 5: preview front+back (2) + forward save front+back (2) + 1 retake cadangan.
# Spam 6+/menit masih 429 + Retry-After (anti-abuse tetap jalan).
POST_RATE_LIMIT_REQUESTS = int(os.getenv("POST_RATE_LIMIT_REQUESTS", "60"))
POST_RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("POST_RATE_LIMIT_WINDOW_SECONDS", "60"))

# --- Rate limiting GET (polling) & DELETE (cleanup) — Ideal handling 429 (11. Penanganan 429) ---
# POST burst 5/min (di atas) agar front+back <60 detik lolos.
# GET (polling OCR/status/files/image) longgar: default 120 req / 60s.
# Dulu 60/60 → dual polling (2 sisi × 10 poll) + Choose thumbs 1 bucket bisa mepet.
# Sekarang default 120/60. Polling ideal tetap 1 URL/poll (jangan 6 URL/poll).
# Jika melebihi, FastAPI return 429 dengan Retry-After + X-RateLimit headers (di-handle di app/security.py).
GET_RATE_LIMIT_REQUESTS = int(os.getenv("GET_RATE_LIMIT_REQUESTS", "120"))
GET_RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("GET_RATE_LIMIT_WINDOW_SECONDS", "60"))
# DELETE untuk retakeCleanup Laravel (best effort): batasi 30/min cegah abuse
DELETE_RATE_LIMIT_REQUESTS = int(os.getenv("DELETE_RATE_LIMIT_REQUESTS", "30"))
DELETE_RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("DELETE_RATE_LIMIT_WINDOW_SECONDS", "60"))

# --- Filesystem External (10.4) ---
# Laravel ↔ FastAPI mapping:
#   uploads/label-ref-*.png        -> input capture (POST /api/test/status)
#   uploads/ocr/label-*.png.json   -> hasil OCR (GET /api/test/ocr/{filename})
#   ocr/latest?filename=           -> query spesifik (GET /api/test/ocr/latest?filename=xxx)
# Flag ini hanya dokumentasi/env toggle; logic latest?filename sudah di main.py
OCR_LATEST_ALLOW_FILENAME_QUERY = os.getenv("OCR_LATEST_ALLOW_FILENAME_QUERY", "true").lower() == "true"

# --- OCR Engine ---
TESSERACT_LANG = "eng"
TESSERACT_PSM = "6"  # Page Segmentation Mode 6 = single uniform block
TESSERACT_CONFIG = f"--psm {TESSERACT_PSM}"

# ---------------------------------------------------------------------------
# OCR Engine Upgrade (2026-09-11 / OCR_ENGINE_UPGRADE_PADDLE_PLAN)
# "paddle" = RapidOCR ONNX (PP-OCR deteksi+recognisi, CPU) | "tesseract" = lama
# ---------------------------------------------------------------------------
OCR_ENGINE = os.getenv("OCR_ENGINE", "paddle").strip().lower()  # paddle | vitstr | surya | tesseract
OCR_DET_THRESHOLD = float(os.getenv("OCR_DET_THRESHOLD", "0.5"))
OCR_REC_THRESHOLD = float(os.getenv("OCR_REC_THRESHOLD", "0.5"))
OCR_MIN_TEXT_LEN = int(os.getenv("OCR_MIN_TEXT_LEN", "1"))
OCR_MODEL_DIR = os.getenv("OCR_MODEL_DIR", "").strip() or None

# ---------------------------------------------------------------------------
# OCR Tuning RapidOCR (2026-09-12 / PADDLEOCR_NATIVE_PLAN §6.0 FALLBACK opsi B+C)
# §6.0 VERDICT RED (paddlepaddle tanpa wheel py3.14) → akurasi via model PP-OCR
# yang lebih besar + threshold, tetap runtime ONNX CPU (tanpa dependensi baru).
# - REC_MODEL medium = PP-OCRv6_rec_medium, DEFAULT sejak 2026-09-12 (~73MB
#   auto-download sekali lalu offline, ~4-5s/gambar, terukur memperbaiki
#   HANDPHONE -> KANDPHONE + multilingual 50 bahasa incl. angka/tanda baca).
# - REC_MODEL small = perilaku lama (~1s/gambar, fallback bila CPU keberatan).
# Revert: set ENV RAPIDOCR_REC_MODEL=small + restart.
# Dibaca saat engine dibangun (lazy singleton) → perubahan BUTUH RESTART.
# Validasi ringan di sini; validasi penuh (enum rapidocr) di paddle_engine.
# ---------------------------------------------------------------------------
RAPIDOCR_REC_MODEL = os.getenv("RAPIDOCR_REC_MODEL", "small").strip().lower() or "small"  # medium | small
RAPIDOCR_REC_VERSION = os.getenv("RAPIDOCR_REC_VERSION", "PP-OCRv6").strip() or "PP-OCRv6"  # PP-OCRv6 | PP-OCRv5
RAPIDOCR_REC_LANG = os.getenv("RAPIDOCR_REC_LANG", "en").strip().lower() or "en"  # en | ch | latin ...
RAPIDOCR_DET_VERSION = os.getenv("RAPIDOCR_DET_VERSION", "PP-OCRv6").strip() or "PP-OCRv6"
RAPIDOCR_DET_MODEL = os.getenv("RAPIDOCR_DET_MODEL", "small").strip().lower() or "small"  # tiny | small | medium
RAPIDOCR_BOX_THRESH = float(os.getenv("RAPIDOCR_BOX_THRESH", "0.5"))
RAPIDOCR_TEXT_SCORE = float(os.getenv("RAPIDOCR_TEXT_SCORE", "0.5"))
RAPIDOCR_UNCLIP_RATIO = float(os.getenv("RAPIDOCR_UNCLIP_RATIO", "1.6"))

# ---------------------------------------------------------------------------
# OCR Optimization Plan (2026-10-02 / OCR_OPTIMIZATION_PLAN R1-R10)
# Semua opt-in via .env; default = perilaku lama (nol regresi).
# 6-core target: OCR_WORKERS=2, INTRA auto = core//workers = 3.
# ---------------------------------------------------------------------------
def _cpu_count_physical_default() -> int:
    try:
        import os as _os

        n = _os.cpu_count() or 4
        return max(1, int(n))
    except Exception:
        return 4


_CPU_COUNT = _cpu_count_physical_default()

OCR_WARMUP = os.getenv("OCR_WARMUP", "true").lower() == "true"
# 0 = nonaktif (perilaku lama: gambar original penuh)
OCR_MAX_SIDE_PX = int(os.getenv("OCR_MAX_SIDE_PX", "1600"))
OCR_DET_LIMIT_SIDE = int(os.getenv("OCR_DET_LIMIT_SIDE", "960"))
OCR_LONGPOLL_MAX_SECONDS = float(os.getenv("OCR_LONGPOLL_MAX_SECONDS", "10"))
CALLBACK_WORKERS = int(os.getenv("CALLBACK_WORKERS", "2"))
OCR_WORKERS = max(1, int(os.getenv("OCR_WORKERS", "2")))
OCR_QUEUE_MAX = max(1, int(os.getenv("OCR_QUEUE_MAX", "20")))
_ort_intra_raw = os.getenv("OCR_ORT_INTRA_THREADS", "0").strip()
try:
    _ort_intra_val = int(_ort_intra_raw or "0")
except Exception:
    _ort_intra_val = 0
if _ort_intra_val <= 0:
    OCR_ORT_INTRA_THREADS = max(1, _CPU_COUNT // max(1, OCR_WORKERS))
else:
    OCR_ORT_INTRA_THREADS = max(1, _ort_intra_val)
OCR_ORT_INTER_THREADS = int(os.getenv("OCR_ORT_INTER_THREADS", "1"))
OCR_CASCADE_ENABLED = os.getenv("OCR_CASCADE_ENABLED", "false").lower() == "true"
OCR_CASCADE_CONF = float(os.getenv("OCR_CASCADE_CONF", "0.90"))
OCR_CASCADE_REC_MODEL = os.getenv("OCR_CASCADE_REC_MODEL", "medium").strip().lower() or "medium"
OCR_USE_CLS = os.getenv("OCR_USE_CLS", "true").lower() == "true"
OCR_TESSERACT_TIMEOUT_SECONDS = float(os.getenv("OCR_TESSERACT_TIMEOUT_SECONDS", "10"))
OCR_TESSERACT_MAX_SIDE = int(os.getenv("OCR_TESSERACT_MAX_SIDE", "2500"))
OCR_TIMINGS_ENABLED = os.getenv("OCR_TIMINGS_ENABLED", "true").lower() == "true"

# ---------------------------------------------------------------------------
# Compare Similarity Teks OCR (2026-09-12 / COMPARE_TEXT_SIMILARITY_PLAN)
# reference vs slot dihitung di FastAPI (stdlib difflib, <5ms), verdict ikut
# callback slot yang sudah ada (tanpa HTTP tambahan).
# SIMILARITY_ENABLED=false = kill switch (callback tetap jalan seperti dulu).
# ---------------------------------------------------------------------------
SIMILARITY_ENABLED = os.getenv("SIMILARITY_ENABLED", "true").lower() == "true"
SIMILARITY_PASS_THRESHOLD = float(os.getenv("SIMILARITY_PASS_THRESHOLD", "0.85"))

# ---------------------------------------------------------------------------
# Toleransi tanda baca/spasi pada compare (2026-10-09)
# - CANONICAL_PUNCT=true (default): perbedaan titik/koma/strip/spasi diabaikan
#   pada skor + diff ("HELLO." == "HELLO", "PT." == "PT"). Kill switch: false
#   = kembali ke perilaku lama (tanpa toleransi). BUTUH RESTART.
# - KEEP_DECIMAL=true (default): titik/koma yang diapit digit ("1.5",
#   "805.011") tetap dibedakan (lindungi nilai angka/kode). false = agresif.
# ---------------------------------------------------------------------------
SIMILARITY_CANONICAL_PUNCT = os.getenv("SIMILARITY_CANONICAL_PUNCT", "true").lower() == "true"
SIMILARITY_KEEP_DECIMAL = os.getenv("SIMILARITY_KEEP_DECIMAL", "true").lower() == "true"

# ---------------------------------------------------------------------------
# Engine compare teks (2026-10-09 / PLAN_MIGRASI_DIFFLIB_KE_RAPIDFUZZ)
# difflib = perilaku lama (stdlib) | rapidfuzz = baru (C++, default).
# Rollback instan: ubah .env SIMILARITY_ENGINE=difflib + restart. BUTUH RESTART.
# ---------------------------------------------------------------------------
SIMILARITY_ENGINE = os.getenv("SIMILARITY_ENGINE", "rapidfuzz").strip().lower() or "rapidfuzz"  # rapidfuzz | difflib

# ---------------------------------------------------------------------------
# Content-gate beda konten (2026-10-09): jatuhkan PASS→FAIL bila terlalu banyak
# kata tak-terjelaskan (beda label/ukuran, bukan typo OCR). Skor tetap apa adanya.
# - CONTENT_GATE=false = nonaktif (verdict murni dari skor). BUTUH RESTART.
# - MAX_ORPHAN_FRACTION: batas fraksi kata yatim tak-terjelaskan (default 0.08).
# - ORPHAN_SIM_MIN: kemiripan minimum agar kata yatim dianggap pasangan
#   noise-OCR (default 0.6). Naikkan bila terlalu toleran, turunkan bila
#   terlalu galak. BUTUH RESTART setelah diubah.
# ---------------------------------------------------------------------------
SIMILARITY_CONTENT_GATE = os.getenv("SIMILARITY_CONTENT_GATE", "true").lower() == "true"
SIMILARITY_MAX_ORPHAN_FRACTION = float(os.getenv("SIMILARITY_MAX_ORPHAN_FRACTION", "0.08") or 0.08)
SIMILARITY_ORPHAN_SIM_MIN = float(os.getenv("SIMILARITY_ORPHAN_SIM_MIN", "0.6") or 0.6)

# Kandidat path Tesseract di Windows (dicoba berurutan)
TESSERACT_CANDIDATE_PATHS = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    r"C:\Users\Aji\AppData\Local\Tesseract-OCR\tesseract.exe",
]

# Jumlah maksimum hasil OCR yang disimpan di memory (LRU).
# FIX timeout Label 1/2 (2026-10-06): batas polling dipanjangkan ke 120s (frontend)
# + 150s (Laravel). Agar "Cek Lagi" (re-attach filename lama) tetap ketemu tanpa
# capture ulang, memory harus menampung >1 sesi split (4 label + margin).
# Default 20 (naik dari 10 hardcoded). BUTUH RESTART setelah diubah.
# LRU cap di save() tetap berlaku; RETAIN_COUNT=5 tetap jadi batas cleanup per POST.
OCR_MAX_MEMORY_RESULTS = max(5, int(os.getenv("OCR_MAX_MEMORY_RESULTS", "20") or 20))

# Jendela "fresh pending" untuk redirect polling race (detik).
# FIX timeout Label 1/2 (2026-10-06): frontend batas 120s, Laravel 150s.
# Pending-race di main.py return 200 pending + poll_url yang benar bila ada job
# pending lain yang masih fresh (bukan 404). Default 180 agar menutup 120s + margin.
OCR_PENDING_FRESH_SECONDS = float(os.getenv("OCR_PENDING_FRESH_SECONDS", "180") or 180)

# --- Transient capture tanpa disk (2026-09-15 / PLAN_MYSQL_BACKED_REFERENCE_STORAGE F7) ---
# False = ocr_store.save() HANYA memory (tanpa uploads/ocr/*.json).
# True  = perilaku lama (tulis JSON disk, untuk debug offline).
OCR_DISK_PERSIST = os.getenv("OCR_DISK_PERSIST", "false").lower() == "true"

# Preprocessing
OCR_UPSCALE_FACTOR = 2

# ---------------------------------------------------------------------------
# Callback ke Laravel (PUSH cleaned_text) — Fix Retake 401 (2026-09-06)
# Sinkron 3 variabel WAJIB sama dengan Laravel .env (lihat .env.example):
#   OCR_CALLBACK_API_KEY / LARAVEL_API_KEY = YOUR_API_KEY_HERE (isi via ENV)
#   OCR_CALLBACK_SECRET  / LARAVEL_HMAC_SECRET = YOUR_HMAC_SECRET_HERE (isi via ENV)
#   OCR_CALLBACK_URL     / LARAVEL_CALLBACK_URL = http://127.0.0.1:8000/api/ocr/callback
# Fallback berurutan: LARAVEL_* -> OCR_CALLBACK_* -> FASTAPI_* (untuk transisi)
# ---------------------------------------------------------------------------
LARAVEL_CALLBACK_URL = (
    os.getenv("LARAVEL_CALLBACK_URL")
    or os.getenv("OCR_CALLBACK_URL")
    or "http://127.0.0.1:8000/api/ocr/callback"
)

def _clean_env_val(val: str = None) -> str:
    if not val:
        return ""
    val = val.strip()
    if val in ("", "YOUR_API_KEY_HERE", "YOUR_CHECK_API_KEY", "YOUR_HMAC_SECRET_HERE"):
        return ""
    return val


# API Key untuk header X-API-Key (wajib diisi di .env untuk production)
# Prioritas: LARAVEL_API_KEY -> OCR_CALLBACK_API_KEY -> FASTAPI_API_KEY
LARAVEL_API_KEY = (
    _clean_env_val(os.getenv("LARAVEL_API_KEY"))
    or _clean_env_val(os.getenv("OCR_CALLBACK_API_KEY"))
    or _clean_env_val(os.getenv("FASTAPI_API_KEY"))
    or ""
)

# Secret untuk HMAC-SHA256 header X-Signature (min 32 char random)
# Prioritas: LARAVEL_HMAC_SECRET -> OCR_CALLBACK_SECRET -> LARAVEL_CALLBACK_SECRET
LARAVEL_HMAC_SECRET = (
    _clean_env_val(os.getenv("LARAVEL_HMAC_SECRET"))
    or _clean_env_val(os.getenv("OCR_CALLBACK_SECRET"))
    or _clean_env_val(os.getenv("LARAVEL_CALLBACK_SECRET"))
    or ""
)

# Alias untuk backward compat — biar code yang pakai OCR_CALLBACK_* tetap jalan
OCR_CALLBACK_API_KEY = LARAVEL_API_KEY
OCR_CALLBACK_SECRET = LARAVEL_HMAC_SECRET
OCR_CALLBACK_URL = LARAVEL_CALLBACK_URL

# Enable/disable callback (bisa dimatikan untuk testing tanpa Laravel)
CALLBACK_ENABLED = os.getenv("CALLBACK_ENABLED", "true").lower() == "true"

# Timeout HTTP callback (detik). Connect timeout fixed 2s di client.
CALLBACK_TIMEOUT_SECONDS = float(os.getenv("CALLBACK_TIMEOUT_SECONDS", "5.0"))

# Jumlah retry jika Laravel 5xx/timeout/429
CALLBACK_RETRIES = int(os.getenv("CALLBACK_RETRIES", "3"))

# Base untuk exponential backoff: attempt 1=2s, 2=4s, 3=8s (+ jitter 0-1s)
CALLBACK_RETRY_BACKOFF_BASE = float(os.getenv("CALLBACK_RETRY_BACKOFF_BASE", "2.0"))

# Jika true, kirim raw_text juga (lebih boros bandwidth). Default false.
CALLBACK_INCLUDE_RAW_TEXT = os.getenv("CALLBACK_INCLUDE_RAW_TEXT", "false").lower() == "true"

# Jika false, hanya kirim yang status success. Jika true, kirim juga failed (untuk alert).
CALLBACK_SEND_ON_FAILED = os.getenv("CALLBACK_SEND_ON_FAILED", "false").lower() == "true"

# File log callback (JSON Lines) dan dead letter
CALLBACK_LOG_FILE = UPLOAD_DIR / "callback_log.jsonl"
CALLBACK_DEAD_LETTER_FILE = UPLOAD_DIR / "callback_dead_letter.jsonl"

# Allowlist URL callback untuk cegah SSRF — hanya host ini yang diizinkan
# Kosong = tidak ada allowlist (hanya validasi skema http/https). Isi contoh: "127.0.0.1,localhost,yourdomain.com"
CALLBACK_ALLOWED_HOSTS_RAW = os.getenv("CALLBACK_ALLOWED_HOSTS", "127.0.0.1,localhost")
CALLBACK_ALLOWED_HOSTS = {h.strip().lower() for h in CALLBACK_ALLOWED_HOSTS_RAW.split(",") if h.strip()}

# --- Retake Cleanup ---
# Jika True: setiap POST baru hapus file lama di uploads/ dan ocr/ (hanya simpan terbaru) — hemat storage & cegah stale
CLEANUP_RETAKE = os.getenv("CLEANUP_RETAKE", "true").lower() == "true"
# Jumlah file terbaru yang dipertahankan.
# Default 5 (FIX split 2026-09-30): 1 sesi split dual-side = 4 label
# (front-1, front-2, back-1, back-2) + 1 margin. Nilai 1 MERUSAK alur split:
# POST Label 2 menghapus metadata+memory Label 1 (false-empty Label 1).
# Memory LRU (OCR_MAX_MEMORY_RESULTS=10) tetap jadi batas absolut.
RETAIN_COUNT = int(os.getenv("RETAIN_COUNT", "5"))  # 5 = 1 sesi split (4) + 1 margin; 1 = hanya terbaru (single-flow saja)

# ---------------------------------------------------------------------------
# Check Batch — 5 Images (Laravel -> FastAPI) — PLAN_CHECK_5_IMAGES
# ---------------------------------------------------------------------------
CHECK_UPLOAD_ROOT = Path(os.getenv("CHECK_UPLOAD_ROOT", "uploads/check"))
# Resolve terhadap project root (parent dari app/)
if not CHECK_UPLOAD_ROOT.is_absolute():
    CHECK_UPLOAD_ROOT = (Path(__file__).resolve().parent.parent / CHECK_UPLOAD_ROOT).resolve()

# API Key khusus Check — fallback berurutan: CHECK_API_KEY -> LARAVEL_API_KEY -> OCR_CALLBACK_API_KEY -> FASTAPI_API_KEY
CHECK_API_KEY = (
    _clean_env_val(os.getenv("CHECK_API_KEY"))
    or _clean_env_val(os.getenv("LARAVEL_API_KEY"))
    or _clean_env_val(os.getenv("OCR_CALLBACK_API_KEY"))
    or _clean_env_val(os.getenv("FASTAPI_API_KEY"))
    or ""
)

CHECK_MAX_FILE_SIZE = int(os.getenv("CHECK_MAX_FILE_SIZE", str(MAX_UPLOAD_SIZE)))  # 5MB per file
CHECK_ALLOWED_MIMES = {m.strip().lower() for m in os.getenv("CHECK_ALLOWED_MIMES", "image/png,image/jpeg").split(",") if m.strip()}

# ---------------------------------------------------------------------------
# Check OCR Batch — Process 5 Labels (PLAN_CHECK_OCR_5_LABEL_PROCESS)
# ---------------------------------------------------------------------------
CHECK_OCR_ENABLED = os.getenv("CHECK_OCR_ENABLED", "true").lower() == "true"
CHECK_OCR_MAX_WORKERS = int(os.getenv("CHECK_OCR_MAX_WORKERS", "2"))  # 1 serial, 2 paralel
CHECK_OCR_TIMEOUT_SECONDS = float(os.getenv("CHECK_OCR_TIMEOUT_SECONDS", "15.0"))
CHECK_PROCESS_RATE_LIMIT_REQUESTS = int(os.getenv("CHECK_PROCESS_RATE_LIMIT_REQUESTS", "30"))
CHECK_PROCESS_RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("CHECK_PROCESS_RATE_LIMIT_WINDOW_SECONDS", "60"))
CHECK_OCR_CALLBACK_ENABLED = os.getenv("CHECK_OCR_CALLBACK_ENABLED", "false").lower() == "true"
# Progressive callback per-slot (PLAN_PROGRESSIVE_COMPARE_AND_ASYNC_CALLBACK_PER_LABEL).
# true = kirim callback tiap slot selesai via background thread + summary di akhir.
# false = perilaku lama (borongan sinkron di akhir batch). Rollback tanpa ubah kode.
CHECK_OCR_PROGRESSIVE_CALLBACK = os.getenv("CHECK_OCR_PROGRESSIVE_CALLBACK", "true").lower() == "true"
# Callback URL khusus check — fallback ke LARAVEL_CALLBACK_URL dengan suffix /check/ocr/callback
_raw_check_callback = os.getenv("CHECK_OCR_CALLBACK_URL")
if _raw_check_callback:
    CHECK_OCR_CALLBACK_URL = _raw_check_callback
else:
    # Fallback: ganti /api/ocr/callback -> /api/check/ocr/callback jika ada
    if LARAVEL_CALLBACK_URL and "/api/ocr/callback" in LARAVEL_CALLBACK_URL:
        CHECK_OCR_CALLBACK_URL = LARAVEL_CALLBACK_URL.replace("/api/ocr/callback", "/api/check/ocr/callback")
    else:
        CHECK_OCR_CALLBACK_URL = LARAVEL_CALLBACK_URL or "http://127.0.0.1:8000/api/check/ocr/callback"

# ---------------------------------------------------------------------------
# Visual Comparison — Gemini 1.5 Flash (PLAN_VISUAL_COMPARISON_GEMINI)
# [OCR-REENABLE 2026-09-11 / PLAN_REVERT_TO_OCR] — Mode OCR aktif kembali.
#     Compare Gemini nonaktif via .env (bukan hapus kode).
# ---------------------------------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
GEMINI_COMPARE_TIMEOUT = float(os.getenv("GEMINI_COMPARE_TIMEOUT", "30.0"))
GEMINI_COMPARE_ENABLED = os.getenv("GEMINI_COMPARE_ENABLED", "false").lower() == "true"

# ---------------------------------------------------------------------------
# Visual Compare Mode — callback hasil compare ke Laravel (PLAN_VISUAL_COMPARE_MODE)
# [OCR-REENABLE 2026-09-11 / PLAN_REVERT_TO_OCR] — callback compare OFF sementara.
# ---------------------------------------------------------------------------
# URL webhook Laravel penerima hasil compare (1 POST per batch).
# Route ini BELUM ADA di Laravel (lihat snippet di plans/VISUAL_COMPARE_MODE_PLAN.md §7).
# Selama belum ada, callback 404 → hanya di-log, compare tetap sukses.
CHECK_COMPARE_CALLBACK_URL = os.getenv(
    "CHECK_COMPARE_CALLBACK_URL",
    "http://127.0.0.1:8000/api/check/compare/callback",
)
CHECK_COMPARE_CALLBACK_ENABLED = os.getenv("CHECK_COMPARE_CALLBACK_ENABLED", "false").lower() == "true"

# ---------------------------------------------------------------------------
# Check Storage Efficiency — Retain 1 Batch Only (PLAN_CHECK_STORAGE_EFFICIENCY)
# ---------------------------------------------------------------------------
CHECK_RETAIN_BATCHES = int(os.getenv("CHECK_RETAIN_BATCHES", "1"))  # 1 = hanya terbaru, 3 = simpan 3 terbaru untuk audit
CHECK_CLEANUP_ON_CAPTURE = os.getenv("CHECK_CLEANUP_ON_CAPTURE", "true").lower() == "true"
CHECK_LATEST_BATCH_FILE = CHECK_UPLOAD_ROOT / "latest_batch.txt"

# --- Reference MySQL backend (2026-09-15 / PLAN_MYSQL_BACKED_REFERENCE_STORAGE) ---
# REF_STORE=sqlite (lama, file+SQLite) | mysql (baru, stateless DB utama).
# Cutover aman via flag + restart; rollback = kembalikan ke sqlite.
REF_STORE = os.getenv("REF_STORE", "sqlite").strip().lower()
MYSQL_HOST = os.getenv("MYSQL_HOST", "127.0.0.1")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_DB = os.getenv("MYSQL_DB", "label_inspection")
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
MYSQL_POOL_SIZE = int(os.getenv("MYSQL_POOL_SIZE", "5"))

# Pastikan folder check ada saat import (idempotent)
CHECK_UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
# Buat .gitkeep jika belum ada
_gitkeep = CHECK_UPLOAD_ROOT / ".gitkeep"
if not _gitkeep.exists():
    try:
        _gitkeep.touch(exist_ok=True)
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Reference Library — Label Acuan Permanen (PLAN_REFERENCE_LIBRARY_FASTAPI)
# Storage terisolasi di uploads/references/, KEBAL terhadap CLEANUP_RETAKE
# dan _cleanup_old_batches (yang hanya menyentuh uploads/*.png dan check/).
# ---------------------------------------------------------------------------
REF_ROOT = (Path(__file__).resolve().parent.parent / "uploads" / "references")
REF_ROOT.mkdir(parents=True, exist_ok=True)
REF_DB_PATH = REF_ROOT / "references.db"
REF_MAX_IMAGE_DIM = int(os.getenv("REF_MAX_IMAGE_DIM", "1280"))  # px sisi terpanjang
REF_THUMB_DIM = int(os.getenv("REF_THUMB_DIM", "320"))  # px thumbnail
REF_JPG_QUALITY = int(os.getenv("REF_JPG_QUALITY", "82"))
REF_THUMB_QUALITY = int(os.getenv("REF_THUMB_QUALITY", "70"))
REF_MAX_COUNT = int(os.getenv("REF_MAX_COUNT", "2000"))
REF_MAX_BYTES_MB = float(os.getenv("REF_MAX_BYTES_MB", "2048"))  # soft quota
# Rate limit push Reference Library (POST /api/references & /from-capture).
# Default 60 req / 60s agar simpan multi-lembar (mis. 4 lembar × 4 label = 16 push/set) tidak kena 429.
REF_RATE_LIMIT_REQUESTS = int(os.getenv("REF_RATE_LIMIT_REQUESTS", "60"))
REF_RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("REF_RATE_LIMIT_WINDOW_SECONDS", "60"))

# ---------------------------------------------------------------------------
# Deteksi Label Kosong (Blank Label + Anti Bleed-Through)
# ---------------------------------------------------------------------------
BLANK_LABEL_DETECT_ENABLED = os.getenv("BLANK_LABEL_DETECT_ENABLED", "true").lower() == "true"
BLANK_INK_DELTA = float(os.getenv("BLANK_INK_DELTA", "48"))
BLANK_MIN_INK_RATIO = float(os.getenv("BLANK_MIN_INK_RATIO", "0.0012"))
BLANK_SOBEL_MIN = float(os.getenv("BLANK_SOBEL_MIN", "45"))
BLANK_BORDER_TRIM_RATIO = float(os.getenv("BLANK_BORDER_TRIM_RATIO", "0.06"))
