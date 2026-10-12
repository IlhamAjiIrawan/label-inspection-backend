# Label Inspection Backend (FastAPI)

Service FastAPI untuk **OCR label + callback ke Laravel**, **Check Batch 4–5 gambar**, **compare similarity teks**, dan **Reference Library label acuan**. Berjalan CPU-only di port `9000`, terintegrasi dengan Laravel (`:8000`) sebagai upstream.

> Dokumen planning (`plans/*.md`, `SETUP.md`) **tidak di-commit** ke repo — semuanya dipindahkan sebagai **GitHub Issues** berlabel `documentation` (39 issues). Lihat tab Issues untuk riwayat desain.

---

## 1. Fitur utama

| Area | Keterangan |
|---|---|
| OCR pipeline 10 tahap | Upload → validasi → simpan → preprocessing → OCR → parsing → store → GET → callback → reconcile |
| Engine OCR | `paddle` (RapidOCR ONNX PP-OCR, default) dengan fallback `tesseract` via `OCR_ENGINE` |
| Callback PUSH ke Laravel | `cleaned_text` + HMAC-SHA256, retry 3x exponential, dead-letter, allowlist anti-SSRF |
| Check Batch | `POST /api/check/captures` terima 4–5 gambar/batch + `POST /api/check/process` OCR batch + callback progresif per-slot |
| Compare similarity | Reference vs slot (`rapidfuzz` default, fallback `difflib`), threshold 0.85, toleransi tanda baca, content-gate |
| Reference Library | Label acuan permanen (`uploads/references/`), backend `sqlite` atau `mysql` via `REF_STORE` |
| Rate limiting | POST burst 5/60s, GET 120/60s, DELETE 30/60s, process 10/60s — semua 429 menyertakan `Retry-After` + `X-RateLimit-*` |
| Keamanan upload | Maks 5 MB, ekstensi + magic bytes, filename di-generate internal, anti path-traversal |

---

## 2. Arsitektur & alur

```
Laravel (:8000)  ──POST image──>  FastAPI (:9000) ──BackgroundTasks──> OCR ──PUSH──> Laravel callback
     │                                   │                                              │
     │  GET /api/test/ocr/{fn} (poll)    │  uploads/*.png + metadata.json               │ HMAC verify
     │  DELETE (retakeCleanup)           │  ocr_store memory LRU + disk (opsional)      │ simpan hasil
```

### 2.1 Pipeline OCR single capture (10 tahap, `main.py`)

1. **Validasi** — ukuran ≤ 5 MB, ekstensi `png/jpg/jpeg`, magic bytes PNG/JPEG (`app/security.py:validate_image_file`).
2. **Simpan** — nama file di-generate `label-{YYYYMMDD-HHMMSS}-{uuid8}[-front-1].{ext}`, tulis `uploads/`, update `metadata.json` atomik.
3. **Deteksi** — pakai generated filename (bukan input client).
4. **Preprocessing** — grayscale → CLAHE → autocontrast → upscale 2x → sharpen → medianBlur, fallback adaptiveThreshold (CPU-only, `app/ocr/preprocess.py`).
5. **OCR Engine** — RapidOCR ONNX default (`app/ocr/paddle_engine.py`); Tesseract fallback (`app/ocr/engine.py`, `--psm 6`, fallback `11,3` bila blank).
6. **Parsing** — strip, hapus baris kosong, pecah per baris; kontrak `'none'` = teks valid bila gambar memang tanpa teks (`app/ocr/parser.py`).
7. **Store** — memory LRU (`OCR_MAX_MEMORY_RESULTS`, default 20) + disk JSON **hanya** bila `OCR_DISK_PERSIST=true` (`app/ocr/store.py`).
8. **GET** — hasil via `/api/test/ocr/latest` / `/api/test/ocr/{filename}` + flag `is_empty`/warning; long-poll `?wait=` s.d. `OCR_LONGPOLL_MAX_SECONDS`.
9. **Callback PUSH** — kirim `cleaned_text` ke Laravel via `httpx` + HMAC + retry (`app/callback/`).
10. **Reconcile** — Laravel bisa fallback `GET /ocr/latest` + trigger manual `POST /callback/retry`.

### 2.2 Check Batch (4–5 gambar, `routers/check.py`)

1. `POST /api/check/captures` — validasi `X-API-Key == CHECK_API_KEY`, `batch_code` regex `^[A-Za-z0-9\-_]+$`, 4–5 file, slot unik 1..5, MIME + magic + 5 MB/file, idempoten bila `batch_code` sudah ada.
2. `POST /api/check/process` (alias `/api/check/ocr/process`) — OCR batch (serial default `CHECK_OCR_MAX_WORKERS=1`).
3. Callback progresif per-slot + summary `batch_completed` bila `CHECK_OCR_PROGRESSIVE_CALLBACK=true` (default), atau borongan sinkron bila `false`.
4. `POST /api/check/compare` — compare slot vs reference; `GET /api/check/compare/{batch}` baca hasil.

### 2.3 Reference Library (`routers/references.py`, `app/references/`)

Scan-sekali-pilih-ulang: `POST /api/references` (kompres + OCR background) → `GET /api/references` (list + search) → `GET /{ref_id}` (detail/polling) → `/{ref_id}/image|thumb` → `DELETE /{ref_id}`. Storage terisolasi di `uploads/references/`, **kebal** terhadap `CLEANUP_RETAKE`. Backend dipilih via `REF_STORE=sqlite|mysql`.

---

## 3. Prasyarat

| Kebutuhan | Detail |
|---|---|
| Python | **3.14.x** (terverifikasi 3.14.3). `opencv-python>=4.10` wajib di NumPy 2.x — jangan turunkan ke 4.8 |
| OS | Windows 10/11 atau Linux. Path Tesseract default Windows di `app/config.py:TESSERACT_CANDIDATE_PATHS` |
| Tesseract | Opsional (hanya bila `OCR_ENGINE=tesseract` atau fallback R10). Installer UB Mannheim, centang `eng` |
| MySQL | Opsional, hanya bila `REF_STORE=mysql` (default `sqlite`, tanpa dependensi tambahan) |
| Laravel upstream | Default `http://127.0.0.1:8000` (callback + check). Bisa dimatikan via `CALLBACK_ENABLED=false` untuk testing mandiri |

---

## 4. Instalasi

```powershell
# 1. Clone
git clone https://github.com/ajidwi02/label-inspection-backend.git
cd label-inspection-backend

# 2. Virtual env + dependensi
python -m venv .venv
.\.venv\Scripts\Activate.ps1        # Linux: source .venv/bin/activate
pip install -r requirements.txt

# 3. (Opsional) Tesseract bila pakai engine tesseract
"C:\Program Files\Tesseract-OCR\tesseract.exe" --version

# 4. Konfigurasi environment
copy .env.example .env              # Linux: cp .env.example .env
# Edit .env — isi API key/HMAC, sinkron dengan Laravel (lihat §5)

# 5. Jalan
python main.py
# atau: uvicorn main:app --host 0.0.0.0 --port 9000
```

> **PENTING:** tetap `uvicorn --workers 1`. Store OCR, rate-limit, dan antrean ada di memori per-proses; konkurensi diatur via `OCR_WORKERS` (ThreadPool), bukan multi-proses. `OCR_WORKERS=2` + `OCR_ORT_INTRA_THREADS=auto` ideal untuk 6 core.

Verifikasi:

```powershell
curl.exe -s http://127.0.0.1:9000/health
curl.exe -s http://127.0.0.1:9000/
```

---

## 5. Konfigurasi (`.env`)

Salin `.env.example` → `.env`. **Jangan commit `.env`** (sudah di `.gitignore`). Semua perubahan ENV **butuh restart** service.

### 5.1 Sinkronisasi wajib dengan Laravel

Tiga variabel ini **harus identik** dengan `.env` Laravel (sisi Laravel: `OCR_CALLBACK_API_KEY`, `LARAVEL_HMAC_SECRET`, `LARAVEL_CALLBACK_URL`):

| FastAPI | Laravel | Isi |
|---|---|---|
| `LARAVEL_API_KEY` (fallback `OCR_CALLBACK_API_KEY` → `FASTAPI_API_KEY`) | `OCR_CALLBACK_API_KEY` | 32 char random |
| `LARAVEL_HMAC_SECRET` (fallback `OCR_CALLBACK_SECRET`) | `LARAVEL_HMAC_SECRET` | 64 char hex |
| `LARAVEL_CALLBACK_URL` (fallback `OCR_CALLBACK_URL`) | `LARAVEL_CALLBACK_URL` | `http://127.0.0.1:8000/api/ocr/callback` |

Generate baru (PowerShell):

```powershell
-join ((48..57)+(65..90)+(97..122) | Get-Random -Count 32 | ForEach-Object {[char]$_})
-join ((48..57)+(97..102) | Get-Random -Count 64 | ForEach-Object {[char]$_})
```

### 5.2 OCR engine & tuning RapidOCR

| Variabel | Default | Keterangan |
|---|---|---|
| `OCR_ENGINE` | `paddle` | `paddle` (RapidOCR ONNX) \| `tesseract` (fallback lama) |
| `OCR_DET_THRESHOLD` / `OCR_REC_THRESHOLD` | `0.5` / `0.5` | Ambang deteksi/recognition |
| `OCR_MIN_TEXT_LEN` | `1` | Panjang teks minimum |
| `OCR_MODEL_DIR` | kosong | Direktori model kustom (kosong = auto-download) |
| `RAPIDOCR_REC_MODEL` | `medium` | `medium` akurat ~4–5 s (fix H→K) \| `small` cepat ~1 s |
| `RAPIDOCR_REC_VERSION` / `RAPIDOCR_DET_VERSION` | `PP-OCRv6` | Versi model PP-OCR |
| `RAPIDOCR_REC_LANG` | `en` | Bahasa recognition |
| `RAPIDOCR_DET_MODEL` | `small` | `tiny` \| `small` \| `medium` |
| `RAPIDOCR_BOX_THRESH` / `RAPIDOCR_TEXT_SCORE` | `0.5` | Threshold box & skor teks |
| `RAPIDOCR_UNCLIP_RATIO` | `1.6` | Unclip ratio deteksi |

### 5.3 Optimasi OCR R1–R10 (target 6-core)

| Variabel | Default | Keterangan |
|---|---|---|
| `OCR_WARMUP` | `true` | Warm-up engine saat startup agar request pertama ≈ steady-state |
| `OCR_MAX_SIDE_PX` | `1600` | Pre-resize sisi terpanjang (`0` = nonaktif). Turunkan 1280 → 960 bila `critical_lines` tetap lolos |
| `OCR_DET_LIMIT_SIDE` | `960` | `Det.limit_side_len` RapidOCR |
| `OCR_LONGPOLL_MAX_SECONDS` | `10` | Batas `GET /ocr/{fn}?wait=` |
| `OCR_MAX_MEMORY_RESULTS` | `20` | Kapasitas LRU memory (dukung "Cek Lagi" tanpa capture ulang) |
| `OCR_PENDING_FRESH_SECONDS` | `180` | Jendela redirect polling race (menutup frontend 120 s + Laravel 150 s) |
| `OCR_WORKERS` / `OCR_QUEUE_MAX` | `2` / `20` | Thread OCR + batas antrean (penuh → `503 + Retry-After: 3`) |
| `OCR_ORT_INTRA_THREADS` | `0` (= auto `core // workers`) | Thread ONNX intra-op; `OCR_ORT_INTER_THREADS` default `1` |
| `OCR_CASCADE_ENABLED` / `OCR_CASCADE_CONF` / `OCR_CASCADE_REC_MODEL` | `false` / `0.90` / `medium` | Cascade small→medium selektif |
| `OCR_USE_CLS` | `true` | Angle classifier (`false` lebih cepat tapi label terbalik gagal) |
| `OCR_TESSERACT_TIMEOUT_SECONDS` / `OCR_TESSERACT_MAX_SIDE` | `10` / `2500` | Fallback Tesseract |
| `OCR_TIMINGS_ENABLED` | `true` | Sertakan `timings/image_scale/ocr_config` di hasil |
| `CALLBACK_WORKERS` | `2` | Pool callback terpisah (jangan samakan dengan `OCR_WORKERS`) |
| `OCR_DISK_PERSIST` | `false` | `false` = memory saja (persistensi di Laravel); `true` = tulis JSON disk untuk debug offline |

### 5.4 Compare similarity teks

| Variabel | Default | Keterangan |
|---|---|---|
| `SIMILARITY_ENABLED` | `true` | Kill switch (`false` = callback jalan tanpa verdict) |
| `SIMILARITY_PASS_THRESHOLD` | `0.85` | Skor ≥ threshold = PASS |
| `SIMILARITY_CANONICAL_PUNCT` | `true` | Abaikan titik/koma/strip/spasi pada skor+diff |
| `SIMILARITY_KEEP_DECIMAL` | `true` | Lindungi desimal (`1.5` vs `15` tetap beda) |
| `SIMILARITY_ENGINE` | `rapidfuzz` | `rapidfuzz` (C++, default) \| `difflib` (rollback) |
| `SIMILARITY_CONTENT_GATE` | `true` | Jatuhkan PASS→FAIL bila kata tak-terjelaskan > batas |
| `SIMILARITY_MAX_ORPHAN_FRACTION` | `0.08` | Batas fraksi kata yatim |
| `SIMILARITY_ORPHAN_SIM_MIN` | `0.6` | Kemiripan minimum agar yatim dianggap noise-OCR |

### 5.5 Callback ke Laravel

| Variabel | Default | Keterangan |
|---|---|---|
| `CALLBACK_ENABLED` | `true` | `false` untuk testing tanpa Laravel |
| `CALLBACK_TIMEOUT_SECONDS` | `5.0` | Timeout HTTP (connect fixed 2 s) |
| `CALLBACK_RETRIES` / `CALLBACK_RETRY_BACKOFF_BASE` | `3` / `2.0` | Retry 5xx/timeout/429: 2/4/8 s + jitter |
| `CALLBACK_INCLUDE_RAW_TEXT` | `false` | `true` = kirim `raw_text` juga (boros) |
| `CALLBACK_SEND_ON_FAILED` | `false` | `true` = kirim juga saat OCR failed (alert) |
| `CALLBACK_ALLOWED_HOSTS` | `127.0.0.1,localhost` | Allowlist anti-SSRF (kosongkan = disable) |

Header callback: `X-API-Key`, `X-Timestamp` (TTL 5 mnt), `X-Signature: sha256=<hex>` dengan message `"{timestamp}.{canonical_json}"` (`app/callback/signer.py`). Payload: `event` (`ocr.completed`/`ocr.failed`), `filename`, `original_filename`, `cleaned_text`, `lines`, `status`, `engine`, `is_empty`, `processed_at`, `processing_time_seconds`, plus metadata (`source`, `template_*`, `check_session_id`, `label_slot`, `reference_id`). Gagal 3x → `uploads/callback_dead_letter.jsonl`; tiap attempt → `uploads/callback_log.jsonl` (rotate 10 MB).

### 5.6 Retake cleanup & rate limit

| Variabel | Default | Keterangan |
|---|---|---|
| `CLEANUP_RETAKE` | `true` | Tiap POST baru hapus file lama (hemat + anti-stale) |
| `RETAIN_COUNT` | `5` | 5 = 1 sesi split (4 label) + 1 margin. Jangan `1` untuk alur split (merusak Label 1) |
| `POST_RATE_LIMIT_REQUESTS` / `POST_RATE_LIMIT_WINDOW_SECONDS` | `5` / `60` | Burst capture (preview 2 + forward 2 + 1 retake) |
| `GET_RATE_LIMIT_REQUESTS` / `GET_RATE_LIMIT_WINDOW_SECONDS` | `120` / `60` | Polling longgar (dual polling + thumbs aman) |
| `DELETE_RATE_LIMIT_REQUESTS` / `DELETE_RATE_LIMIT_WINDOW_SECONDS` | `30` / `60` | Cleanup best-effort |
| `OCR_LATEST_ALLOW_FILENAME_QUERY` | `true` | `GET /ocr/latest?filename=` spesifik anti-stale |

### 5.7 Check Batch

| Variabel | Default | Keterangan |
|---|---|---|
| `CHECK_API_KEY` | (fallback ke `LARAVEL_API_KEY` …) | Key khusus check, sinkron dengan Laravel `CHECK_EXTERNAL_API_KEY` |
| `CHECK_UPLOAD_ROOT` | `uploads/check` | Root batch |
| `CHECK_MAX_FILE_SIZE` | `5242880` | 5 MB/file |
| `CHECK_ALLOWED_MIMES` | `image/png,image/jpeg` | MIME diizinkan |
| `CHECK_OCR_ENABLED` / `CHECK_OCR_MAX_WORKERS` / `CHECK_OCR_TIMEOUT_SECONDS` | `true` / `1` / `15` | OCR batch (1 serial, 2 paralel) |
| `CHECK_PROCESS_RATE_LIMIT_REQUESTS` / `CHECK_PROCESS_RATE_LIMIT_WINDOW_SECONDS` | `10` / `60` | Rate limit trigger process |
| `CHECK_OCR_CALLBACK_ENABLED` / `CHECK_OCR_CALLBACK_URL` | `true` / `http://127.0.0.1:8000/api/check/ocr/callback` | Callback check (fallback otomatis dari URL utama) |
| `CHECK_OCR_PROGRESSIVE_CALLBACK` | `true` | Streaming per-slot + summary (`false` = borongan, rollback) |
| `CHECK_RETAIN_BATCHES` / `CHECK_CLEANUP_ON_CAPTURE` | `1` / `true` | Retensi batch (hemat storage) |
| `CHECK_COMPARE_CALLBACK_URL` / `CHECK_COMPARE_CALLBACK_ENABLED` | `.../api/check/compare/callback` / `false` | Callback hasil compare (route Laravel mungkin belum ada → 404 hanya di-log) |

Sinkron Laravel: `CHECK_EXTERNAL_URL=http://127.0.0.1:9000/api/check/captures`.

### 5.8 Gemini & Reference backend

| Variabel | Default | Keterangan |
|---|---|---|
| `GEMINI_API_KEY` / `GEMINI_MODEL` / `GEMINI_COMPARE_TIMEOUT` / `GEMINI_COMPARE_ENABLED` | kosong / `gemini-3.8-flash` / `30` / `false` | Visual compare **nonaktif** (mode OCR aktif). Jangan isi key di repo |
| `REF_STORE` | `sqlite` | `sqlite` (file+SQLite) \| `mysql` (stateless, cutover via flag + restart) |
| `MYSQL_HOST` / `MYSQL_PORT` / `MYSQL_DB` / `MYSQL_USER` / `MYSQL_PASSWORD` / `MYSQL_POOL_SIZE` | `127.0.0.1` / `3306` / `label_inspection` / `root` / kosong / `5` | Password **hanya** di `.env`, tidak di `.env.example` |
| `REF_MAX_IMAGE_DIM` / `REF_THUMB_DIM` / `REF_JPG_QUALITY` / `REF_THUMB_QUALITY` / `REF_MAX_COUNT` / `REF_MAX_BYTES_MB` | `1280` / `320` / `82` / `70` / `2000` / `2048` | Kompresi & kuota library |

---

## 6. Menjalankan & endpoint

Base URL: `http://127.0.0.1:9000`. Semua 429 mengembalikan `retry_after_seconds` + header `Retry-After`, `X-RateLimit-Limit/Remaining/Reset`.

### 6.1 Single capture + OCR (`main.py`)

| Method | Path | Fungsi | Batas |
|---|---|---|---|
| `POST` | `/api/test/status` | Upload gambar + form → trigger OCR background | 5/60s |
| `GET` | `/api/test/status` | Status terakhir (`?filename=` spesifik) | 120/60s |
| `GET` | `/api/test/image/{filename}` | Download file (basename saja) | 120/60s |
| `GET` | `/api/test/files` | List file + `ocr_status` | 120/60s |
| `GET` | `/api/test/ocr/latest` | OCR terbaru (`?filename=`, anti-stale) | 120/60s |
| `GET` | `/api/test/ocr/{filename}` | OCR per file; `?wait=8` long-poll | 120/60s |
| `POST` | `/api/test/ocr/run` | Trigger manual file terbaru | 5/60s |
| `DELETE` | `/api/test/ocr/{filename}` | Hapus image + JSON + metadata + memory (`X-API-Key`) | 30/60s |
| `DELETE` | `/api/test/status?filename={fn}` | Alias hapus via query | 30/60s |
| `DELETE` | `/api/test/status/{filename}` | Alias hapus via path | 30/60s |
| `POST` | `/api/test/ocr/{filename}/callback/retry` | Replay callback | 5/60s |
| `GET` | `/api/test/callback/logs?limit=20` | Log callback terakhir | 120/60s |
| `GET` | `/api/test/callback/dead-letter?limit=20` | Dead letter (gagal 3x) | 120/60s |
| `GET` | `/api/test/result/{filename}`, `/api/ocr/*` | Alias kompatibilitas | sama |
| `GET` | `/health` | Status engine, callback, antrean, rate limit (tanpa secret) | — |
| `GET` | `/` | `{"message":"External Service is running"}` | — |

Form `POST /api/test/status` (multipart, semua wajib kecuali 3 terakhir opsional): `image` (PNG/JPEG ≤ 5 MB), `source`, `timestamp`, `template_id`, `template_name`, `template_width`, `template_height`, `filename`, `resolution_w`, `resolution_h`, `filesize`, `check_session_id?`, `label_slot?` (`1`/`2`), `reference_id?`.

```powershell
curl.exe -s -X POST http://127.0.0.1:9000/api/test/status `
  -F image=@label.png -F source=label-inspection -F timestamp=2026-10-12T10:00:00+07:00 `
  -F template_id=TL -F template_name=Test -F template_width=100 -F template_height=50 `
  -F filename=label.png -F resolution_w=600 -F resolution_h=400 -F filesize=12671
# -> {"success":true,"filename":"label-20261012-100000-a1b2c3d4.png","ocr_status":"pending"}
curl.exe -s "http://127.0.0.1:9000/api/test/ocr/label-20261012-100000-a1b2c3d4.png?wait=8"
```

**Polling ideal:** simpan `filename` dari response POST, lalu poll **1 URL spesifik** (`GET /ocr/{saved}` tiap 3 s × 10). Jangan fetch 6 URL/poll. Hormati `Retry-After` dengan backoff 2/4/8 s + jitter.

### 6.2 Check Batch (`/api/check`)

| Method | Path | Fungsi |
|---|---|---|
| `POST` | `/api/check/captures` | Terima 4–5 gambar (`images[]` + `batch_code` + `slot_numbers`) |
| `POST` | `/api/check/process` (alias `/api/check/ocr/process`) | Trigger OCR batch |
| `GET` | `/api/check/ocr/{batch_code}` | Hasil OCR batch |
| `GET` | `/api/check/ocr/{batch_code}/slot/{n}` | Hasil per slot |
| `GET` | `/api/check/latest` | Batch terbaru |
| `POST` | `/api/check/compare` | Compare batch vs reference |
| `GET` | `/api/check/compare/{batch_code}` | Hasil compare |

### 6.3 Reference Library (`/api/references`)

| Method | Path | Fungsi |
|---|---|---|
| `POST` | `/api/references` | Simpan acuan baru (rate limit sendiri 10/mnt) |
| `GET` | `/api/references` | List paginasi + search |
| `GET` | `/api/references/stats` | Statistik |
| `GET` | `/api/references/health` | Ping DB |
| `POST` | `/api/references/from-capture` | Promosikan file `uploads/` lama |
| `GET` | `/api/references/{ref_id}` | Detail + polling OCR |
| `GET` | `/api/references/{ref_id}/image` | Gambar asli (cache 1 hari) |
| `GET` | `/api/references/{ref_id}/thumb` | Thumbnail (cache 1 hari) |
| `DELETE` | `/api/references/{ref_id}` | Hapus |

---

## 7. Struktur folder

```
.
├── main.py                 # app + pipeline OCR + endpoint /api/test/* + /health
├── app/
│   ├── config.py           # semua ENV + default (satu tempat)
│   ├── security.py         # validasi, sanitasi, rate-limit sliding window
│   ├── storage.py          # metadata.json atomik, cleanup retake, resolve filename
│   ├── db_mysql.py         # backend MySQL reference
│   ├── callback/           # signer (HMAC) + payload + client (retry/dead-letter)
│   ├── ocr/                # preprocess, paddle_engine, engine (tesseract),
│   │                       # dispatch, executor (R6), store (LRU+long-poll), parser
│   ├── compare/            # visual compare Gemini (nonaktif default)
│   ├── compare_text/       # similarity rapidfuzz/difflib + content-gate
│   └── references/         # service + db SQLite reference library
├── routers/
│   ├── check.py            # /api/check/* (captures/process/compare)
│   └── references.py       # /api/references/*
├── scripts/                # bench_ocr, make_bench_images, migrate_* (sqlite→mysql, dual-side)
├── tests/ocr_bench/        # dataset benchmark lokal (tidak di-commit)
├── uploads/                # runtime (hanya .gitkeep yang di-commit)
│   ├── ocr/                # hasil OCR (bila OCR_DISK_PERSIST=true)
│   ├── check/              # batch check
│   └── references/         # library permanen (kebal cleanup)
├── test_*.py               # test manual: compare, similarity, paddle
├── requirements.txt
├── .env.example            # template (tanpa secret asli)
└── README.md               # file ini (satu-satunya .md yang di-commit)
```

Kebijakan git: `*.md` (kecuali `README.md`), `.env`, `*.bak`, `*.sql`, `baseline.json`, `*.zip`, `*.db`, dan seluruh isi runtime `uploads/` **di-ignore**. Lihat `.gitignore`.

---

## 8. Testing & benchmark

```powershell
$env:PYTHONPATH='.'
python test_similarity.py     # verdict PASS/FAIL/SKIP + assert <5 ms untuk 5 pasang
python test_compare.py
python test_paddle_ocr.py
python test_native_compare.py
python -m compileall app main.py
```

Benchmark Fase 0 (observability R1–R10):

```powershell
$env:PYTHONPATH='.'
python scripts/make_bench_images.py
python scripts/bench_ocr.py --dir tests/ocr_bench --repeat 3 --out baseline.json --batch5
python scripts/bench_ocr.py --dir tests/ocr_bench --repeat 3 --compare baseline.json
```

Migrasi reference:

```powershell
python scripts/migrate_library_sqlite_to_mysql.py
python scripts/migrate_ref_dual_side.py
python scripts/migrate_ref_add_side.py
```

---

## 9. Keamanan

- Validasi berlapis: 5 MB (`app/config.py`), ekstensi + magic bytes (`app/security.py`), filename generate internal + `is_safe_path()` + `sanitize_filename_for_lookup()`.
- DELETE verifikasi `X-API-Key == LARAVEL_API_KEY` (timing-safe); tanpa key terkonfigurasi = mode dev longgar (jangan di production).
- Callback: HMAC-SHA256 + `X-Timestamp` TTL 5 mnt + allowlist host anti-SSRF + timeout + retry + dead-letter; 401 tidak di-retry (ada hint log sinkronisasi key).
- Secret hanya via `.env` (tidak pernah di-commit); `.env.example` berisi placeholder. Bila key pernah bocor di histori/Issues, **rotasi segera**.
- Tidak ada `eval`/`exec` input user; tidak ada traceback internal ke client.
- `uploads/` + `sql/` dump + `*.bak` tidak di-commit (data runtime/dump bisa sensitif).

---

## 10. Troubleshooting

| Gejala | Penyebab umum | Solusi |
|---|---|---|
| `tesseract_available:false` di `/health` | Path Tesseract salah / `eng` tidak terinstal | Reinstall UB Mannheim + `eng`; cek `TESSERACT_CANDIDATE_PATHS` |
| Semua OCR gagal `rapidocr/onnxruntime belum install` | `opencv-python` 4.8 di NumPy 2.x | `pip install "opencv-python>=4.10"` (jangan 4.8) |
| `ModuleNotFoundError` | venv tidak aktif / deps belum install | Aktifkan venv → `pip install -r requirements.txt` |
| Port 9000 dipakai | Proses lama masih jalan | `netstat -ano \| findstr :9000` → `taskkill /PID <PID> /F` |
| `404 No status data` / OCR kosong | Belum ada POST sukses / memory reset setelah restart | POST dulu; `OCR_DISK_PERSIST=true` bila butuh persistensi disk |
| Callback `401` | Key/HMAC tidak sinkron Laravel ↔ FastAPI | Samakan 3 variabel §5.1; cek `callback_log.jsonl` + `/health.callback_401_count` |
| `429` POST | Spam >5/mnt | Tunggu `Retry-After`; alur normal front+back tidak kena |
| `429` GET | Poll 6 URL/poll atau paralel berlebih | Poll 1 URL spesifik, 3 s × 10; backoff 2/4/8 s + jitter |
| `503 OCR queue full` | `OCR_WORKERS`/`OCR_QUEUE_MAX` jenuh batch 5 | Retry setelah 3 s; naikkan worker sesuai core (tetap `--workers 1`) |
| Label terbalik gagal | `OCR_USE_CLS=false` | Set `true` + restart |
| Frontend 120 s / Laravel 150 s timeout | Memory LRU kecil / pending basi | `OCR_MAX_MEMORY_RESULTS=20`, `OCR_PENDING_FRESH_SECONDS=180` + restart |
| `paddleocr` tidak bisa install di Py 3.14 | Belum ada wheel `paddlepaddle` | Jangan tambah `paddleocr` (verdict RED); pakai `RAPIDOCR_REC_MODEL=medium` |
| `*.md` hilang setelah clone | By design → Issues | Baca tab GitHub Issues (`documentation`) |

---

## 11. Catatan operasional

- **Satu-satunya config surface adalah `.env`** (`app/config.py` hanya default). Semua flag butuh **restart**.
- Jangan jalankan `uvicorn --workers >1`.
- Log runtime: stdout + `uploads/callback_log.jsonl` + `uploads/callback_dead_letter.jsonl` + `GET /api/test/callback/logs|dead-letter`.
- `GET /health` adalah dashboard tanpa secret: engine, tuning aktif, antrean, rate limit, 401/empty/failed counters.
- Backup yang perlu: `.env` (rahasia, di luar git) + `uploads/references/` atau MySQL `label_inspection` bila pakai `REF_STORE=mysql`.
