# src/data_pipeline/01_download_data.py

import os
import time
import requests
import json
import pandas as pd
import re
import logging
from typing import List, Dict

# --- Konfigurasi ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Direkomendasikan untuk memuat API Key dari environment variable, bukan hardcode
# Untuk sekarang, kita gunakan placeholder. Ganti dengan kunci API Anda.
API_KEY = os.getenv("BPS_API_KEY", "7a62d6af2de1e80xxxx")
[cite_start]DOMAIN_ID = "0000"  # Domain BPS Pusat [cite: 193]
BASE_URL = 'https://webapi.bps.go.id/v1/api/list/model/statictable/lang/ind/domain/'

# Path untuk menyimpan output
RAW_DATA_DIR = "data/01_raw"
PROCESSED_DATA_PATH = "data/02_processed/corpus.csv"

# --- Fungsi-fungsi ---

def download_all_pages(domain: str) -> None:
    """
    Mengunduh semua halaman data tabel statis dari Web API BPS untuk domain tertentu.
    [cite_start]Logika berasal dari Lampiran 1. [cite: 1311]
    """
    headers = {'Authorization': f'Bearer {API_KEY}'}
    page = 1
    
    while True:
        url = f"{BASE_URL}{domain}/page/{page}/key/{API_KEY}"
        try:
            logging.info(f"Mengunduh halaman {page} untuk domain {domain}...")
            response = requests.get(url, headers=headers)
            response.raise_for_status()  # Akan error jika status code bukan 2xx
            
            data = response.json()
            
            # Cek apakah data['data'][1] ada dan tidak kosong
            if 'data' not in data or not data['data'] or len(data['data']) < 2 or not data['data'][1]:
                logging.info(f"Tidak ada lagi data pada halaman {page}. Proses unduh selesai.")
                break

            # Simpan respons JSON mentah
            os.makedirs(os.path.join(RAW_DATA_DIR, domain), exist_ok=True)
            filename = os.path.join(RAW_DATA_DIR, domain, f"response_page_{page}.json")
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            
            logging.info(f"Berhasil menyimpan {filename}")
            page += 1
            time.sleep(1)  # Jeda 1 detik untuk menghindari rate limiting

        except requests.exceptions.RequestException as e:
            logging.error(f"Error saat mengunduh halaman {page}: {e}")
            break
        except (KeyError, IndexError):
            logging.error(f"Format respons tidak terduga pada halaman {page}. Menghentikan proses.")
            break

def aggregate_and_clean_data(domain: str) -> pd.DataFrame:
    """
    Menggabungkan semua file JSON mentah menjadi satu DataFrame dan membersihkannya.
    """
    all_records = []
    domain_dir = os.path.join(RAW_DATA_DIR, domain)
    
    if not os.path.exists(domain_dir):
        logging.error(f"Direktori data mentah tidak ditemukan: {domain_dir}")
        return pd.DataFrame()

    for filename in sorted(os.listdir(domain_dir)):
        if filename.endswith(".json"):
            filepath = os.path.join(domain_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Data tabel berada di 'data'[1]
                records = data.get('data', [None, []])[1]
                all_records.extend(records)

    if not all_records:
        logging.warning("Tidak ada data yang berhasil diagregasi.")
        return pd.DataFrame()
        
    df = pd.DataFrame(all_records)
    logging.info(f"Total {len(df)} data berhasil diagregasi.")
    
    # [cite_start]--- Proses Pembersihan (dari Lampiran 2) --- [cite: 1312]
    logging.info("Memulai proses pembersihan data...")
    df.dropna(subset=['title'], inplace=True)
    
    # Menghapus tag HTML
    df['title'] = df['title'].str.replace(r'<[^>]*>', '', regex=True)
    # Menghapus karakter non-ASCII
    df['title'] = df['title'].str.encode('ascii', 'ignore').str.decode('ascii')
    # Menormalisasi spasi
    df['title'] = df['title'].str.replace(r'\s+', ' ', regex=True).str.strip()

    logging.info("Pembersihan data selesai.")
    return df

# --- Eksekusi Utama ---
if __name__ == "__main__":
    logging.info("--- Memulai Pipeline Pengumpulan Data ---")
    
    # 1. Unduh semua data mentah
    download_all_pages(domain=DOMAIN_ID)
    
    # 2. Agregasi dan bersihkan data
    cleaned_df = aggregate_and_clean_data(domain=DOMAIN_ID)
    
    # 3. Simpan korpus yang sudah bersih
    if not cleaned_df.empty:
        # Memilih kolom yang relevan sesuai skripsi
        final_columns = ['table_id', 'title', 'subj', 'subj_id', 'url', 'last_update']
        final_df = cleaned_df[[col for col in final_columns if col in cleaned_df.columns]]
        
        # Pastikan direktori output ada
        os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
        final_df.to_csv(PROCESSED_DATA_PATH, index=False)
        logging.info(f"Korpus bersih berhasil disimpan di: {PROCESSED_DATA_PATH}")
    
    logging.info("--- Pipeline Pengumpulan Data Selesai ---")