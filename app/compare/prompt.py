"""
app/compare/prompt.py — Template prompt untuk Visual Comparison via Gemini.

Prompt ini menginstruksikan Gemini untuk:
1. Menerima gambar pertama sebagai REFERENCE (acuan)
2. Membandingkan setiap gambar berikutnya (check images) dengan reference
3. Mengembalikan hasil dalam format JSON strict

Pemisahan ini memudahkan tuning prompt tanpa mengubah logic bisnis.
"""

# Prompt utama yang dikirim ke Gemini bersama gambar-gambar
COMPARISON_SYSTEM_PROMPT = """Kamu adalah sistem inspeksi visual label pada lini produksi.

TUGAS:
Bandingkan setiap gambar CHECK (gambar ke-2 sampai terakhir) dengan gambar REFERENCE (gambar pertama).

KRITERIA PERBANDINGAN:
1. **Posisi dan layout** — Apakah posisi elemen-elemen (teks, logo, barcode, gambar) sama?
2. **Komponen visual** — Apakah semua komponen yang ada di reference juga ada di check?
3. **Warna dan kontras** — Apakah ada perbedaan warna signifikan? (Abaikan perbedaan pencahayaan/exposure yang minor)
4. **Teks dan konten** — Apakah teks yang terlihat sama?
5. **Elemen hilang/tambahan** — Apakah ada elemen yang hilang atau bertambah?

ATURAN PENILAIAN:
- `is_match: true` → Gambar check SESUAI dengan reference (perbedaan minor akibat pencahayaan/sudut foto DIABAIKAN)
- `is_match: false` → Ada perbedaan SIGNIFIKAN (elemen hilang, warna salah, posisi bergeser, teks berbeda)
- `confidence_score` → Nilai 0.00 sampai 1.00 yang menunjukkan seberapa yakin kamu dengan hasil penilaian
- `notes` → Penjelasan singkat dalam Bahasa Indonesia tentang hasil perbandingan. Jika tidak match, jelaskan APA yang berbeda.

FORMAT RESPONSE:
Kembalikan HANYA JSON valid tanpa markdown code block, tanpa penjelasan tambahan, dengan struktur:
{
  "results": [
    {
      "slot": <nomor_urut_dimulai_1>,
      "file_name": "<nama_file_check>",
      "is_match": <true/false>,
      "confidence_score": <0.00-1.00>,
      "notes": "<penjelasan singkat>"
    }
  ]
}
"""


def build_comparison_prompt(check_filenames: list[str]) -> str:
    """
    Bangun prompt lengkap dengan informasi nama file untuk setiap gambar check.

    Args:
        check_filenames: List nama file check images (tanpa path, hanya basename).
                         Contoh: ["slot1-check-tl-...png", "slot2-check-tl-...png", ...]

    Returns:
        str: Prompt lengkap yang siap dikirim ke Gemini bersama gambar.
    """
    file_list = "\n".join(
        f"  - Gambar Check {i+1} (slot {i+1}): {fn}"
        for i, fn in enumerate(check_filenames)
    )

    return f"""{COMPARISON_SYSTEM_PROMPT}

INFORMASI GAMBAR YANG DIKIRIM:
- Gambar PERTAMA adalah REFERENCE (acuan)
- Gambar berikutnya adalah CHECK images:
{file_list}

Gunakan file_name yang tertera di atas dalam response JSON kamu.
"""
