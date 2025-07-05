# src/data_pipeline/02_generate_queries.py

import os
import logging
import pandas as pd
import google.generativeai as genai
import argparse
from tqdm import tqdm

# --- Konfigurasi ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Muat kunci API dari environment variable
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logging.error("GEMINI_API_KEY tidak ditemukan di environment variable. Harap tambahkan ke file .env Anda.")
    exit()

genai.configure(api_key=GEMINI_API_KEY)

# Path dasar untuk data yang diproses
PROCESSED_DATA_DIR = "data/02_processed"

# --- Prompt untuk LLM (diadaptasi dari Lampiran 3) ---
QUERY_GENERATION_PROMPT = """
Anda adalah asisten AI untuk membuat dataset query pencarian informasi alami dengan tujuan optimasi search engine BPS. Anda akan menerima sebuah judul dokumen dan membuat 3 variasi query yang unik dan berbeda.

Aturan Utama:
- Variasi & Kreativitas: Gunakan sinonim, parafrase, dan bahasa awam. Hindari kemiripan literal dengan judul.
- Sertakan Tahun: Setiap query harus menyertakan minimal salah satu tahun dari rentang tahun dokumen.
- Keselarasan Informasi: Makna query harus sesuai dengan judul dokumen.
- Format Output: Hasil harus berupa 3 baris teks biasa, di mana setiap baris adalah satu query. Jangan sertakan header, nomor, atau markdown.

Sekarang, proses judul dokumen berikut dan buat 3 query unik:
"{document_title}"
"""

# --- Fungsi ---
def generate_queries_for_title(title: str, model) -> list:
    """Menghasilkan 3 kueri untuk satu judul menggunakan model Gemini."""
    try:
        prompt = QUERY_GENERATION_PROMPT.format(document_title=title)
        response = model.generate_content(prompt)
        queries = [q.strip() for q in response.text.strip().split('\n') if q.strip()]
        return queries
    except Exception as e:
        logging.error(f"Error saat menghasilkan kueri untuk judul '{title}': {e}")
        return []

# --- Eksekusi Utama ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Membangkitkan kueri sintetis dari file korpus menggunakan Gemini.")
    parser.add_argument(
        "-m", "--mfd",
        default="0000",
        type=str,
        help="ID MFD 4 digit dari korpus yang akan digunakan. Default: '0000'."
    )
    args = parser.parse_args()
    mfd_id = args.mfd

    input_path = os.path.join(PROCESSED_DATA_DIR, f"corpus_{mfd_id}.csv")
    output_path = os.path.join(PROCESSED_DATA_DIR, f"generated_queries_{mfd_id}.csv")

    logging.info(f"--- Memulai Pipeline Pembangkitan Kueri untuk MFD: {mfd_id} ---")

    if not os.path.exists(input_path):
        logging.error(f"File input tidak ditemukan: {input_path}. Jalankan skrip 01 terlebih dahulu.")
        exit()

    # Muat korpus
    corpus_df = pd.read_csv(input_path)
    logging.info(f"Berhasil memuat {len(corpus_df)} dokumen dari {input_path}.")
    
    # Inisialisasi model Generatif Gemini
    model = genai.GenerativeModel('gemini-pro')

    all_generated_data = []

    # Iterasi melalui setiap baris di korpus dengan progress bar
    for index, row in tqdm(corpus_df.iterrows(), total=corpus_df.shape[0], desc=f"Menghasilkan Kueri untuk MFD {mfd_id}"):
        title = row['title']
        table_id = row['table_id']
        
        generated_queries = generate_queries_for_title(title, model)
        
        for query in generated_queries:
            all_generated_data.append({
                "query": query,
                "original_title": title,
                "table_id": table_id
            })

    # Simpan hasil ke file CSV
    if all_generated_data:
        output_df = pd.DataFrame(all_generated_data)
        output_df.to_csv(output_path, index=False)
        logging.info(f"Berhasil menghasilkan {len(output_df)} kueri dan menyimpannya di: {output_path}")
    else:
        logging.warning("Tidak ada kueri yang berhasil dihasilkan.")

    logging.info("--- Pipeline Pembangkitan Kueri Selesai ---")