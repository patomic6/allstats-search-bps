# src/data_pipeline/01_download_data.py

import os
import time
import requests
import json
import pandas as pd
import re
import logging
import argparse
from typing import List, Dict

# --- Konfigurasi ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

API_KEY = os.getenv("BPS_API_KEY", "7a62d6af2de1e80xxxx")
BASE_URL = 'https://webapi.bps.go.id/v1/api/list/model/statictable/lang/ind/domain/'

# Path untuk menyimpan output
RAW_DATA_DIR = "data/01_raw"
PROCESSED_DATA_PATH = "data/02_processed"

# --- Fungsi-fungsi ---
def download_all_pages(mfd: str) -> None:
    """
    Mengunduh semua halaman data tabel statis dari Web API BPS untuk MFD tertentu.
    """
    headers = {'Authorization': f'Bearer {API_KEY}'}
    page = 1
    
    while True:
        url = f"{BASE_URL}{mfd}/page/{page}/key/{API_KEY}"
        try:
            logging.info(f"Mengunduh halaman {page} untuk MFD {mfd}...")
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            data = response.json()
            
            if 'data' not in data or not data['data'] or len(data['data']) < 2 or not data['data'][1]:
                logging.info(f"Tidak ada lagi data pada halaman {page} untuk MFD {mfd}. Proses unduh selesai.")
                break

            os.makedirs(os.path.join(RAW_DATA_DIR, mfd), exist_ok=True)
            filename = os.path.join(RAW_DATA_DIR, mfd, f"response_page_{page}.json")
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            
            logging.info(f"Berhasil menyimpan {filename}")
            page += 1
            time.sleep(1)

        except requests.exceptions.RequestException as e:
            logging.error(f"Error saat mengunduh halaman {page}: {e}")
            break
        except (KeyError, IndexError):
            logging.error(f"Format respons tidak terduga pada halaman {page}. Menghentikan proses.")
            break


def aggregate_and_clean_data(mfd: str) -> pd.DataFrame:
    """
    Menggabungkan semua file JSON mentah menjadi satu DataFrame dan membersihkannya.
    """
    all_records = []
    mfd_dir = os.path.join(RAW_DATA_DIR, mfd)
    
    if not os.path.exists(mfd_dir):
        logging.error(f"Direktori data mentah tidak ditemukan: {mfd_dir}")
        return pd.DataFrame()

    for filename in sorted(os.listdir(mfd_dir)):
        if filename.endswith(".json"):
            filepath = os.path.join(mfd_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                records = data.get('data', [None, []])[1]
                all_records.extend(records)

    if not all_records:
        logging.warning("Tidak ada data yang berhasil diagregasi.")
        return pd.DataFrame()
        
    df = pd.DataFrame(all_records)
    logging.info(f"Total {len(df)} data berhasil diagregasi untuk MFD {mfd}.")
    
    logging.info("Memulai proses pembersihan data...")
    df.dropna(subset=['title'], inplace=True)
    
    df['title'] = df['title'].str.replace(r'<[^>]*>', '', regex=True)
    df['title'] = df['title'].str.encode('ascii', 'ignore').str.decode('ascii')
    df['title'] = df['title'].str.replace(r'\s+', ' ', regex=True).str.strip()

    logging.info("Pembersihan data selesai.")
    return df


# --- Eksekusi Utama ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unduh dan proses data tabel statis dari Web API BPS.")
    parser.add_argument(
        "-m", "--mfd", 
        default="0000", 
        type=str,
        help="ID MFD 4 digit yang akan diunduh. Default: '0000' untuk BPS Pusat."
    )
    args = parser.parse_args()
    mfd_id = args.mfd
    
    logging.info(f"--- Memulai Pipeline Pengumpulan Data untuk MFD: {mfd_id} ---")
    
    download_all_pages(mfd=mfd_id)
    
    cleaned_df = aggregate_and_clean_data(mfd=mfd_id)
    
    if not cleaned_df.empty:
        final_columns = ['table_id', 'title', 'subj', 'subj_id', 'url', 'last_update']
        final_df = cleaned_df[[col for col in final_columns if col in cleaned_df.columns]]
        
        processed_filename = f"corpus_{mfd_id}.csv"
        processed_filepath = os.path.join(PROCESSED_DATA_PATH, processed_filename)
        
        os.makedirs(os.path.dirname(processed_filepath), exist_ok=True)
        final_df.to_csv(processed_filepath, index=False)
        logging.info(f"Korpus bersih berhasil disimpan di: {processed_filepath}")
    
    logging.info(f"--- Pipeline Pengumpulan Data untuk MFD: {mfd_id} Selesai ---")