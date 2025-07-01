# test_main.py
#
# File ini berisi unit test untuk layanan API pencarian semantik (main.py)
# Pengujian ini menggunakan pendekatan White-Box untuk memvalidasi logika internal.
#
# Framework: pytest
# Mocking: unittest.mock

import pytest
from unittest.mock import patch, MagicMock
from fastapi import HTTPException
import elasticsearch

# Asumsikan kode API Anda ada di file bernama 'main.py'
# dan fungsi yang akan diuji telah di-refactor jika perlu.
import main

# ==============================================================================
# 1. Pengujian untuk Fungsi `validate_api_key`
# ==============================================================================

def test_validate_api_key_valid():
    """
    Kasus Uji (TC-WB-001): Memvalidasi bahwa API key yang benar akan diterima.
    Teknik: Branch Coverage (Jalur 'if' tidak terpenuhi).
    """
    valid_key = "akjba9wf4389fndi9j0998hd3hwe9h898aos8h128"
    # Seharusnya fungsi mengembalikan key tanpa error
    assert main.validate_api_key(valid_key) == valid_key

def test_validate_api_key_invalid():
    """
    Kasus Uji (TC-WB-002): Memvalidasi bahwa API key yang salah akan ditolak.
    Teknik: Branch Coverage (Jalur 'if' terpenuhi).
    """
    invalid_key = "kunci_salah"
    # Mengharapkan HTTPException dengan status 403
    with pytest.raises(HTTPException) as excinfo:
        main.validate_api_key(invalid_key)
    assert excinfo.value.status_code == 403

# ==============================================================================
# 2. Pengujian untuk Fungsi `validate_date`
# ==============================================================================

# Catatan: Asumsikan Anda telah memindahkan fungsi `validate_date` keluar dari
# `semantic_search` agar bisa diuji secara independen.

def test_validate_date_valid_format():
    """
    Kasus Uji (TC-WB-003): Memvalidasi format tanggal yang benar.
    Teknik: Statement Coverage.
    """
    assert main.validate_date("2025-06-28") == "2025-06-28"

def test_validate_date_none_input():
    """
    Kasus Uji (TC-WB-004): Memvalidasi penanganan input None.
    Teknik: Branch Coverage (if date_str).
    """
    assert main.validate_date(None) is None

def test_validate_date_invalid_format():
    """
    Kasus Uji (TC-WB-005): Memvalidasi penanganan format tanggal yang salah.
    Teknik: Branch Coverage (jalur except ValueError).
    """
    with pytest.raises(HTTPException) as excinfo:
        main.validate_date("28-06-2025")
    assert excinfo.value.status_code == 400
    assert "Invalid date format" in excinfo.value.detail

# ==============================================================================
# 3. Pengujian Logika Internal `semantic_search`
# ==============================================================================

# Di sini kita menggunakan @patch untuk "mengganti" fungsi eksternal
# dengan objek tiruan (MagicMock) selama pengujian berlangsung.

@patch('main.es.search')
@patch('main.es.count')
@patch('main.model.encode')
def test_semantic_search_uses_knn_for_large_corpus(mock_encode, mock_count, mock_search):
    """
    Kasus Uji (TC-WB-009): Memastikan kueri KNN digunakan untuk data > 10.000.
    Teknik: Branch Coverage (if doc_count > 10000).
    """
    # Setup mock return values
    mock_encode.return_value = [0.1] * 384 # Vektor dummy
    mock_count.return_value = {"count": 15000} # Jumlah dokumen besar
    mock_search.return_value = {"hits": {"hits": []}} # Respons dummy

    # Panggil fungsi yang diuji
    main.semantic_search(keyword="test", api_key=main.API_KEY)

    # Verifikasi
    # Pastikan es.search dipanggil sekali
    assert mock_search.call_count == 1
    # Ambil argumen yang digunakan untuk memanggil es.search
    _, kwargs = mock_search.call_args
    # Dapatkan body query dari argumen
    query_body = kwargs.get("body", {})
    
    # Periksa apakah kueri 'knn' ada di dalam `should` clause
    should_clauses = query_body["query"]["bool"]["should"]
    # Cari klausa yang merupakan function_score dengan query knn
    knn_clause_found = any(
        'knn' in clause.get('function_score', {}).get('query', {})
        for clause in should_clauses
    )
    assert knn_clause_found, "Kueri KNN seharusnya digunakan untuk korpus besar"


@patch('main.es.search')
@patch('main.es.count')
@patch('main.model.encode')
def test_semantic_search_uses_script_score_for_small_corpus(mock_encode, mock_count, mock_search):
    """
    Kasus Uji (TC-WB-010): Memastikan script_score digunakan untuk data <= 10.000.
    Teknik: Branch Coverage (else dari if doc_count > 10000).
    """
    # Setup
    mock_encode.return_value = [0.1] * 384
    mock_count.return_value = {"count": 5000} # Jumlah dokumen kecil
    mock_search.return_value = {"hits": {"hits": []}}

    # Panggil
    main.semantic_search(keyword="test", api_key=main.API_KEY)

    # Verifikasi
    assert mock_search.call_count == 1
    _, kwargs = mock_search.call_args
    query_body = kwargs.get("body", {})
    
    should_clauses = query_body["query"]["bool"]["should"]
    # Cari klausa yang merupakan function_score dengan query script_score
    script_score_clause_found = any(
        'script_score' in clause.get('function_score', {}).get('query', {})
        for clause in should_clauses
    )
    assert script_score_clause_found, "Kueri script_score seharusnya digunakan untuk korpus kecil"

@patch('main.es.search')
@patch('main.es.count')
@patch('main.model.encode')
def test_semantic_search_removes_stopwords_from_bm25(mock_encode, mock_count, mock_search):
    """
    Kasus Uji (TC-WB-006): Memvalidasi penghapusan stopwords dari kueri BM25.
    Teknik: Statement Coverage, Data Flow Testing.
    """
    # Setup
    mock_encode.return_value = [0.1] * 384
    mock_count.return_value = {"count": 100}
    mock_search.return_value = {"hits": {"hits": []}}

    # Panggil dengan keyword yang mengandung stopwords
    main.semantic_search(keyword="statistik inflasi data indonesia", api_key=main.API_KEY)

    # Verifikasi
    assert mock_search.call_count == 1
    _, kwargs = mock_search.call_args
    query_body = kwargs.get("body", {})

    # Ekstrak query bm25
    bm25_query_part = query_body["query"]["bool"]["should"][0]["function_score"]["query"]
    final_keyword = bm25_query_part["match"]["judul"]["query"]

    # Pastikan stopwords telah dihapus
    assert "statistik" not in final_keyword
    assert "data" not in final_keyword
    assert "indonesia" not in final_keyword
    assert final_keyword == "inflasi"