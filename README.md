# Label Inspection Backend (FastAPI)

Service FastAPI untuk **Auto-Detect & Auto-Deskew Label Berbasis OpenCV (Opsi C)**, **Multi-Model OCR + Callback ke Laravel**, **Check Batch 4 Slot Paralel**, **Dual Similarity Engine (RapidFuzz & Gemini Flash AI)**, dan **Reference Library Label Acuan**. Berjalan secara efisien di CPU pada port `9000`, terintegrasi penuh dengan Laravel (`:8000`) sebagai upstream.

> Dokumen planning (`plans/*.md`, `SETUP.md`) **tidak di-commit** ke repo — semuanya dipindahkan sebagai **GitHub Issues** berlabel `documentation`. Lihat tab Issues untuk riwayat desain.

---

## 1. Fitur Utama

| Area | Keterangan |
|---|---|
| **OpenCV Auto-Detect & Deskew (Opsi C)** | Deteksi pita label otomatis (`POST /api/references/detect-regions`), Canny edge + morphological text fusion, *Short Axis Expansion* (`factor=1.18`, 25–35px margin), *4-Point Perspective Transform (Deskew)*, pemotongan pita continuous 2-in-1 tepat 50:50 (*Sobel Midpoint Cut*), dan *Inverse Mapping* ($M^{-1}$) untuk preview kotak hijau (*annotated bounding box*) |
| **OCR Pipeline 10 Tahap** | Upload → validasi magic bytes → simpan → preprocessing → OCR (Cascade Small→Medium) → parsing → store → GET → callback → reconcile |
| **Multi-Model OCR Engine** | Selectable via `OCR_ENGINE`: `paddle` (RapidOCR ONNX, default produksi), `vitstr` (ViTSTR-Tiny Transformer), `surya` (Surya OCR SegFormer/Transformer), dan `tesseract` (fallback) |
| **Arsitektur Model Cascade** | Eksekusi super cepat dengan model `small` (~1.0 dtk/gambar); otomatis eskalasi (*cascade*) ke model `medium` hanya bila skor confidence `< 0.88` (`OCR_CASCADE_ENABLED=true`) |
| **Check Batch Paralel** | `POST /api/check/captures` terima 4–5 gambar/batch + `POST /api/check/process` eksekusi OCR paralel multi-worker (`CHECK_OCR_MAX_WORKERS=2`) + callback progresif instan per-slot via `as_completed()` (~4–6 detik total 4 slot) |
| **Reference Library & Prefill OCR** | Label acuan permanen (`uploads/references/`), backend `sqlite` atau `mysql` via `REF_STORE`. Mendukung **Prefill OCR** (`cleaned_text`, `lines_json`, `engine`) dari tahap preview sehingga penyimpanan acuan instan (<0.5 dtk) tanpa duplikasi background re-OCR |
| **Dual Compare Similarity** | 1) **RapidFuzz** (Levenshtein + Line F1, threshold 0.85, toleransi tanda baca, content-gate) <br/>2) **Gemini Flash AI** (Visual & Semantic comparison: klasifikasi *Critical Diffs* vs *Minor Diffs*) |
| **Callback PUSH ke Laravel** | `cleaned_text` + HMAC-SHA256, retry 3x exponential, dead-letter, allowlist anti-SSRF |
| **Rate Limiting Ramah Paralel** | POST burst `20/60s` (mendukung dual-slot paralel front+back tanpa 429), GET `120/60s`, DELETE `30/60s`, process `10/60s` — semua 429 menyertakan `Retry-After` + `X-RateLimit-*` |
| **Keamanan Upload** | Maks 5 MB, sinkronisasi ketat ekstensi file + magic bytes (PNG/JPEG), filename di-generate internal, anti path-traversal |

---

## 2. Arsitektur & Alur

```
Laravel (:8000) ──POST detect-regions──> FastAPI (:9000) [OpenCV Deskew + 50:50 Midpoint Cut + Box Hijau]
     │
     ├──POST image (parallel slot 1&2)─> FastAPI (:9000) ──BackgroundTasks──> OCR (Small/Cascade) ──PUSH──> Laravel callback
     │  GET /api/test/ocr/{fn} (poll)    uploads/*.png + metadata.json                                      HMAC verify
     │
     ├──POST /api/references (prefill)─> Simpan instan ke Reference Library (ocr_status="success", lewati re-OCR)
     │
     └──POST /api/check/captures + process ──> ThreadPoolExecutor (2 Workers) ──Progressive Callback per Slot──> Laravel
```

### 2.1 Modul Visi Komputer: OpenCV Auto-Detect & Deskew (`app/vision/label_detector.py`)

Digunakan pada tahap **Tambah Reference** maupun **Check Label** (`POST /api/references/detect-regions`):
1. **Grayscale & Canny Edge Detection (30, 100):** Mengabaikan 4% margin tepi luar frame kamera.
2. **Morphological Rect Close:** Menggabungkan baris-baris teks menjadi kontur lempengan pita label utuh.
3. **Short Axis Expansion (`expand_rect_short_axis`, factor=1.18):** Memperlebar sumbu pendek pita secara simetris sebesar 18% dari titik pusatnya untuk memberi ruang napas (*breathing room* 25–35 px) di sisi kiri & kanan agar huruf tepi (seperti `"PT:"`, `"FR:"`) tidak terpotong.
4. **4-Point Perspective Transform (Deskew):** Meluruskan pita label yang miring menjadi tegak lurus.
5. **Klasifikasi & Sobel Midpoint Cut:**
   - **Kasus A (Pita Continuous 2-in-1):** Profil energi Sobel menentukan batas atas/bawah pita, lalu membelah tepat 50:50 (`mid = (y_min + y_max) // 2`) dengan mempertahankan lebar penuh sumbu pendek (`keep_full_short_axis=True`).
   - **Kasus B (2 Label Terpisah):** Memisahkan dua kontur label independen menjadi Slot 1 & Slot 2.
   - **Kasus C (Fallback):** Adaptive center crops jika kontur tidak terbaca jelas.
6. **Inverse Perspective Mapping ($M^{-1}$):** Memetakan kembali titik potong ke gambar asli untuk menggambar kotak hijau (*annotated preview*) dan mengembalikan gambar crop dalam format PNG Base64 murni (`ext="png"`).

### 2.2 Pipeline OCR Single Capture (10 Tahap, `main.py`)

1. **Validasi** — ukuran ≤ 5 MB, ekstensi `png/jpg/jpeg`, verifikasi ketat magic bytes PNG (`89 50 4E 47`) / JPEG (`FF D8 FF`) (`app/security.py:validate_image_file`).
2. **Simpan** — nama file di-generate `label-{YYYYMMDD-HHMMSS}-{uuid8}[-front-1].{ext}`, tulis `uploads/`, update `metadata.json` atomik.
3. **Deteksi** — pakai generated filename (bukan input client).
4. **Preprocessing** — grayscale → CLAHE → autocontrast → bicubic upscale 2x → sharpen → medianBlur 3x3 (`app/ocr/preprocess.py`).
5. **OCR Engine** — RapidOCR ONNX default (`app/ocr/paddle_engine.py`) dengan model `small` + *cascade* otomatis ke `medium` bila confidence `< 0.88`; atau ViTSTR / Surya / Tesseract sesuai `OCR_ENGINE`.
6. **Parsing** — strip, hapus baris kosong, pecah per baris; kontrak `'none'` = teks valid bila gambar memang tanpa teks (`app/ocr/parser.py`).
7. **Store** — memory LRU (`OCR_MAX_MEMORY_RESULTS`, default 20) + disk JSON bila `OCR_DISK_PERSIST=true` (`app/ocr/store.py`).
8. **GET** — hasil via `/api/test/ocr/latest` / `/api/test/ocr/{filename}` + flag `is_empty`/warning; long-poll `?wait=` s.d. `OCR_LONGPOLL_MAX_SECONDS`.
9. **Callback PUSH** — kirim `cleaned_text` ke Laravel via `httpx` + HMAC + retry (`app/callback/`).
10. **Reconcile** — Laravel bisa fallback `GET /ocr/latest` + trigger manual `POST /callback/retry`.

### 2.3 Check Batch Paralel (4–5 Gambar, `routers/check.py`)

1. `POST /api/check/captures` — validasi `X-API-Key == CHECK_API_KEY`, `batch_code` regex `^[A-Za-z0-9\-_]+$`, 4–5 file, slot unik 1..5, MIME + magic bytes + 5 MB/file, idempoten bila `batch_code` sudah ada.
2. `POST /api/check/process` (alias `/api/check/ocr/process`) — memproses seluruh slot secara **paralel** menggunakan `ThreadPoolExecutor(max_workers=CHECK_OCR_MAX_WORKERS)` (default `2` workers).
3. Callback progresif dikirim ke Laravel begitu masing-masing thread slot selesai (`as_completed`), memperbarui `batch.json` secara real-time, lalu diakhiri summary `batch_completed`.
4. `POST /api/check/compare` — compare slot vs reference (RapidFuzz & opsional Gemini Flash AI); `GET /api/check/compare/{batch}` baca hasil.

### 2.4 Reference Library dengan Prefill OCR (`routers/references.py`, `app/references/`)

- **Deteksi Region (Opsi C):** `POST /api/references/detect-regions` menghasilkan `annotated_preview` dan `slots[]` (Slot 1 & Slot 2).
- **Simpan Instan (Tanpa Triple-OCR):** `POST /api/references` menerima form opsional `cleaned_text`, `lines_json`, dan `engine` dari hasil OCR preview. Jika teks sudah tersedia, sistem langsung menyimpan ke database dengan `ocr_status="success"` dan **melewatkan antrean `_ocr_reference_background`**. Jika belum ada teks, background OCR dijalankan seperti biasa.
- Storage terisolasi di `uploads/references/`, **kebal** terhadap `CLEANUP_RETAKE`. Backend dipilih via `REF_STORE=sqlite|mysql`.

---

## 3. Prasyarat

| Kebutuhan | Detail |
|---|---|
| Python | **Python 3.11 / 3.13 / 3.14** (terverifikasi pada Laragon Python 3.13 & venv). `opencv-python>=4.10` wajib di NumPy 2.x — jangan turunkan ke 4.8 |
| OS | Windows 10/11 atau Linux. Path Tesseract default Windows di `app/config.py:TESSERACT_CANDIDATE_PATHS` |
| Tesseract | Opsional (hanya bila `OCR_ENGINE=tesseract` atau fallback R10). Installer UB Mannheim, centang `eng` |
| MySQL | Direkomendasikan (`REF_STORE=mysql` pada database `label_inspection`), atau `sqlite` untuk mode standalone |
| Laravel upstream | Default `http://127.0.0.1:8000` (callback + check). Bisa dimatikan via `CALLBACK_ENABLED=false` untuk testing mandiri |

---

## 4. Instalasi

### 4.1 Instalasi Cepat (Default Engine - RapidOCR)

```powershell
# 1. Clone
git clone https://github.com/IlhamAjiIrawan/label-inspection-backend.git
cd label-inspection-backend

# 2. Virtual env + dependensi
python -m venv .venv
.\.venv\Scripts\Activate.ps1        # Linux: source .venv/bin/activate
pip install -r requirements.txt

# 3. Konfigurasi environment
copy .env.example .env              # Linux: cp .env.example .env
# Edit .env — isi API key/HMAC, sinkron dengan Laravel (lihat §5)

# 4. Jalankan server
python main.py
# Atau menggunakan Laragon Python 3.13 secara langsung:
# C:\laragon\bin\python\python-3.13\python.exe main.py
```

> **PENTING:** tetap `uvicorn --workers 1`. Store OCR, rate-limit, dan antrean ada di memori per-proses; konkurensi diatur via `OCR_WORKERS=2` dan `CHECK_OCR_MAX_WORKERS=2` (ThreadPool dengan Thread-Local ONNX Session), bukan multi-proses.

Verifikasi kesehatan service:

```powershell
curl.exe -s http://127.0.0.1:9000/health
curl.exe -s http://127.0.0.1:9000/
```

### 4.2 Manajemen Multi Virtual Environment (Switching Model OCR)

Backend ini mendukung beberapa engine OCR (*Multi-Model*). Untuk menghindari konflik dependensi antara PyTorch, ONNX Runtime, dan SegFormer/Transformers, lingkungan Python dapat dipisahkan ke dalam virtual environment masing-masing:

| Model / Engine | Nilai `OCR_ENGINE` | Virtual Environment | Karakteristik Utama |
|---|---|---|---|
| **RapidOCR ONNX** *(Default Produksi)* | `paddle` | `.venv` | Ringan, CPU-friendly (~1.0 dtk/slot dengan model `small` + cascade `medium`), PP-OCRv4/v6 |
| **ViTSTR-Tiny** | `vitstr` | `.venv-vitstr` | Vision Transformer (STR Hub PyTorch), pengenalan kata ultra-cepat |
| **Surya OCR** | `surya` | `.venv-surya` | SegFormer + Transformer, akurasi sangat tinggi untuk 90+ bahasa & simbol |
| **Tesseract** | `tesseract` | `.venv` | Fallback engine legacy via binary Tesseract OCR |

#### Cara Mengganti Virtual Environment & Model:

1. **Nonaktifkan venv aktif** (jika ada): `deactivate`
2. **Aktifkan Virtual Environment tujuan**:
   - RapidOCR / Tesseract: `.\.venv\Scripts\Activate.ps1`
   - ViTSTR: `.\.venv-vitstr\Scripts\Activate.ps1`
   - Surya OCR: `.\.venv-surya\Scripts\Activate.ps1`
3. **Ubah variabel `OCR_ENGINE` di file `.env`**:
   ```ini
   # Pilihan: paddle | vitstr | surya | tesseract
   OCR_ENGINE=paddle
   ```
4. **Jalankan kembali FastAPI server**: `python main.py`

---

## 5. Konfigurasi (`.env`)

Salin `.env.example` → `.env`. **Jangan commit `.env`** (sudah di `.gitignore`). Semua perubahan ENV **butuh restart** service.

### 5.1 Sinkronisasi Wajib dengan Laravel

Tiga variabel ini **harus identik** dengan `.env` Laravel:

| FastAPI | Laravel | Isi |
|---|---|---|
| `LARAVEL_API_KEY` (fallback `OCR_CALLBACK_API_KEY` → `FASTAPI_API_KEY`) | `OCR_CALLBACK_API_KEY` | 32 char random |
| `LARAVEL_HMAC_SECRET` (fallback `OCR_CALLBACK_SECRET`) | `LARAVEL_HMAC_SECRET` | 64 char hex |
| `LARAVEL_CALLBACK_URL` (fallback `OCR_CALLBACK_URL`) | `LARAVEL_CALLBACK_URL` | `http://127.0.0.1:8000/api/ocr/callback` |

### 5.2 OCR Engine & Tuning RapidOCR (Optimasi Kecepatan Produksi)

| Variabel | Default Produksi | Keterangan |
|---|---|---|
| `OCR_ENGINE` | `paddle` | Pilihan model: `paddle` (RapidOCR ONNX default) \| `vitstr` \| `surya` \| `tesseract` |
| `RAPIDOCR_REC_MODEL` | `small` | `small` super cepat (~1 dtk/slot) \| `medium` (~3–4 dtk/slot) |
| `OCR_CASCADE_ENABLED` | `true` | Aktifkan eskalasi otomatis `small` → `medium` jika confidence di bawah ambang |
| `OCR_CASCADE_CONF` / `OCR_CASCADE_REC_MODEL` | `0.88` / `medium` | Ambang confidence cascade & model tujuan eskalasi |
| `RAPIDOCR_REC_VERSION` / `RAPIDOCR_DET_VERSION` | `PP-OCRv6` | Versi model PP-OCR |
| `RAPIDOCR_REC_LANG` | `en` | Bahasa recognition |
| `RAPIDOCR_DET_MODEL` | `small` | `tiny` \| `small` \| `medium` |
| `RAPIDOCR_BOX_THRESH` / `RAPIDOCR_TEXT_SCORE` | `0.5` | Threshold box & skor teks |
| `RAPIDOCR_UNCLIP_RATIO` | `1.6` | Unclip ratio deteksi |

### 5.3 Optimasi Runtime & Konkurensi (R1–R10)

| Variabel | Default Produksi | Keterangan |
|---|---|---|
| `OCR_WARMUP` | `true` | Warm-up engine saat startup agar request pertama langsung cepat |
| `OCR_MAX_SIDE_PX` | `1280` | Pre-resize sisi terpanjang gambar sebelum masuk pipeline OCR |
| `OCR_DET_LIMIT_SIDE` | `736` | `Det.limit_side_len` DBNet RapidOCR (memangkas beban deteksi tanpa kehilangan baris teks) |
| `OCR_WORKERS` / `OCR_QUEUE_MAX` | `2` / `20` | Worker thread OCR preview + kapasitas antrean |
| `OCR_ORT_INTRA_THREADS` | `0` (= auto `core // workers`) | Thread ONNX intra-op; `OCR_ORT_INTER_THREADS` default `1` |
| `OCR_USE_CLS` | `true` | Angle classifier (`true` agar tahan terhadap label miring/terbalik) |
| `OCR_LONGPOLL_MAX_SECONDS` | `10` | Batas `GET /ocr/{fn}?wait=` |
| `OCR_MAX_MEMORY_RESULTS` | `20` | Kapasitas LRU memory |
| `OCR_PENDING_FRESH_SECONDS` | `180` | Jendela redirect polling race |
| `CALLBACK_WORKERS` | `2` | ThreadPool pengiriman webhook callback |
| `OCR_DISK_PERSIST` | `false` | `false` = memory LRU saja; `true` = tulis JSON ke disk untuk debug |

### 5.4 Retake Cleanup & Rate Limit (Anti-429 Paralel)

| Variabel | Default Produksi | Keterangan |
|---|---|---|
| `CLEANUP_RETAKE` | `true` | Tiap POST baru membersihkan file sementara lama |
| `RETAIN_COUNT` | `5` | Menyimpan 5 file terakhir (1 sesi split 4 label + 1 margin) |
| `POST_RATE_LIMIT_REQUESTS` / `POST_RATE_LIMIT_WINDOW_SECONDS` | `20` / `60` | Dinaikkan ke **20 req/mnt** untuk mendukung auto-detect + dual-slot OCR paralel tanpa error 429 |
| `GET_RATE_LIMIT_REQUESTS` / `GET_RATE_LIMIT_WINDOW_SECONDS` | `120` / `60` | Kuota polling longgar |
| `DELETE_RATE_LIMIT_REQUESTS` / `DELETE_RATE_LIMIT_WINDOW_SECONDS` | `30` / `60` | Cleanup best-effort |

### 5.5 Check Batch & Multi-Worker

| Variabel | Default Produksi | Keterangan |
|---|---|---|
| `CHECK_API_KEY` | (fallback ke `LARAVEL_API_KEY`) | Key khusus check, sinkron dengan Laravel `CHECK_EXTERNAL_API_KEY` |
| `CHECK_OCR_ENABLED` | `true` | Aktifkan OCR pada batch check |
| `CHECK_OCR_MAX_WORKERS` | `2` | **Eksekusi paralel 2 worker threads** untuk memproses 4 slot label dalam ~4–6 detik |
| `CHECK_OCR_TIMEOUT_SECONDS` | `15` | Timeout per slot |
| `CHECK_OCR_PROGRESSIVE_CALLBACK` | `true` | Kirim hasil tiap slot seketika saat selesai (`as_completed`) + summary akhir |
| `CHECK_RETAIN_BATCHES` / `CHECK_CLEANUP_ON_CAPTURE` | `1` / `true` | Retensi batch otomatis untuk menghemat ruang disk |

### 5.6 Compare Similarity (RapidFuzz & Gemini Flash AI) & Reference Store

| Variabel | Default Produksi | Keterangan |
|---|---|---|
| `SIMILARITY_ENABLED` / `SIMILARITY_ENGINE` | `true` / `rapidfuzz` | Pencocokan teks deterministik super cepat (<5 ms) |
| `SIMILARITY_PASS_THRESHOLD` | `0.85` | Skor ≥ 0.85 = PASS |
| `GEMINI_COMPARE_ENABLED` / `GEMINI_MODEL` | `false` / `gemini-3.8-flash` | Opsional: komparasi visual & semantik multimodal (*Critical vs Minor Diffs*) |
| `REF_STORE` | `mysql` (atau `sqlite`) | Penyimpanan metadata Reference Library (`mysql` berbagi DB `label_inspection`) |

---

## 6. Endpoint Utama

Base URL: `http://127.0.0.1:9000`.

### 6.1 OpenCV Auto-Detect & Reference Library (`/api/references`)

| Method | Path | Fungsi |
|---|---|---|
| `POST` | `/api/references/detect-regions` | **OpenCV Auto-Detect & Deskew (Opsi C)**: menerima full frame kamera, mengembalikan `annotated_preview` (kotak hijau) + `slots` (crop PNG Base64 Label 1 & Label 2) |
| `POST` | `/api/references` | Simpan acuan baru; mendukung form prefill `cleaned_text`, `lines_json`, `engine` untuk bypass background OCR (`ocr_status="success"` instan) |
| `GET` | `/api/references` | List paginasi + pencarian acuan |
| `GET` | `/api/references/stats` & `/health` | Statistik library & pengecekan koneksi DB |
| `GET` | `/api/references/{ref_id}` | Detail acuan + hasil OCR |
| `GET` | `/api/references/{ref_id}/image` & `/thumb` | Unduh gambar asli / thumbnail acuan |
| `DELETE` | `/api/references/{ref_id}` | Hapus acuan dari library |

### 6.2 Single Capture + OCR Preview (`main.py`)

| Method | Path | Fungsi | Batas |
|---|---|---|---|
| `POST` | `/api/test/status` | Upload gambar slot + form (`template_id` mendukung `S,M,L,XL,TL,AUTO`) → trigger OCR background | 20/60s |
| `GET` | `/api/test/ocr/{filename}` | Ambil hasil OCR per file (`?wait=0` non-blocking atau `?wait=8` long-poll) | 120/60s |
| `GET` | `/api/test/ocr/latest` | Hasil OCR terbaru (`?filename=` spesifik anti-stale) | 120/60s |
| `DELETE` | `/api/test/ocr/{filename}` | Hapus image + JSON + metadata + memory (`X-API-Key`) | 30/60s |
| `GET` | `/health` | Status engine, konfigurasi cascade, antrean, rate limit, dan statistik callback | — |

### 6.3 Check Batch (`/api/check`)

| Method | Path | Fungsi |
|---|---|---|
| `POST` | `/api/check/captures` | Terima 4–5 gambar slot hasil Auto-Detect (`images[]` + `batch_code` + `slot_numbers`) |
| `POST` | `/api/check/process` | Trigger OCR batch paralel (`ThreadPoolExecutor` 2 workers + progressive webhook callback) |
| `GET` | `/api/check/ocr/{batch_code}` | Hasil OCR seluruh slot dalam batch |
| `POST` | `/api/check/compare` | Compare batch terhadap reference |
| `GET` | `/api/check/compare/{batch_code}` | Hasil compare batch |

---

## 7. Struktur Folder

```
.
├── main.py                 # Entrypoint FastAPI + pipeline OCR + endpoint /api/test/* + /health
├── app/
│   ├── config.py           # Konfigurasi ENV & default terpusat
│   ├── security.py         # Validasi ekstensi & magic bytes, sanitasi path, sliding-window rate limiter
│   ├── storage.py          # metadata.json atomik, cleanup retake, resolve filename
│   ├── db_mysql.py         # Backend MySQL untuk Reference Library
│   ├── vision/
│   │   └── label_detector.py # OpenCV Auto-Detect, Short Axis Expansion (1.18), Deskew, & 50:50 Midpoint Cut
│   ├── callback/           # Signer (HMAC-SHA256) + payload + client (retry/dead-letter)
│   ├── ocr/                # preprocess, paddle_engine (Small+Cascade), vitstr_engine,
│   │                       # surya_engine, engine (tesseract), dispatch, executor, store, parser
│   ├── compare/            # Visual & semantic compare via Gemini Flash AI
│   ├── compare_text/       # Similarity RapidFuzz/difflib + content-gate
│   └── references/         # Service & SQLite/MySQL repository untuk Reference Library
├── routers/
│   ├── check.py            # /api/check/* (captures, parallel process, compare)
│   └── references.py       # /api/references/* (detect-regions, prefill store, library CRUD)
├── scripts/                # bench_ocr, make_bench_images, migrate_*
├── uploads/                # Direktori runtime (di-ignore oleh git)
│   ├── check/              # Batch check produksi
│   └── references/         # Gambar master reference permanen (kebal cleanup)
├── test_*.py               # Suite pengujian unit & integrasi (similarity, compare, OCR engines)
├── requirements.txt
├── .env.example            # Template konfigurasi
└── README.md
```

---

## 8. Testing & Verifikasi Sintaks

```powershell
$env:PYTHONPATH='.'
python -m py_compile main.py app/config.py routers/references.py routers/check.py app/vision/label_detector.py
python test_similarity.py     # Verdict PASS/FAIL/SKIP + assert <5 ms
python test_compare.py
python test_paddle_ocr.py
```

---

## 9. Troubleshooting

| Gejala | Penyebab Umum | Solusi |
|---|---|---|
| `400 File content does not match expected image format` | Ekstensi `.png` tetapi konten berformat JPEG (mismatch magic bytes) | Pastikan encoder citra (`detect-regions` / frontend canvas) menggunakan format `image/png` (`ext="png"`) yang konsisten dengan ekstensi nama file |
| `404 Not Found` saat polling `/api/test/ocr/{filename}` | Upload sebelumnya gagal (misal ditolak 400/429) sehingga file tidak terdaftar di store | Cek response `POST /api/test/status`; pastikan kode `200 OK` dan magic bytes gambar valid |
| `429 Too Many Requests` pada POST preview | `POST_RATE_LIMIT_REQUESTS` masih menggunakan nilai lama (`5`) | Set `POST_RATE_LIMIT_REQUESTS=20` di `.env` lalu restart FastAPI |
| Callback ke Laravel timeout / lambat | `php artisan serve` memblokir koneksi karena `OCR_LONGPOLL_WAIT > 0` | Set `OCR_LONGPOLL_WAIT=0` pada `.env` Laravel agar server single-thread PHP tidak tertahan oleh request polling |
| Semua OCR gagal `rapidocr/onnxruntime belum install` | `opencv-python` 4.8 di NumPy 2.x | `pip install "opencv-python>=4.10"` |
| Callback `401 Unauthorized` | Key/HMAC tidak sinkron Laravel ↔ FastAPI | Samakan 3 variabel pada §5.1 di kedua `.env`, lalu restart service |
