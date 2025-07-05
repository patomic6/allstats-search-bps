# ----------------------------------------
# 1. Import Library
# ----------------------------------------
from fastapi import FastAPI, HTTPException, Query, Security, Depends
from fastapi.security.api_key import APIKeyHeader
from fastapi.responses import HTMLResponse
from elasticsearch import Elasticsearch
from elasticsearch.exceptions import RequestError
from sentence_transformers import SentenceTransformer
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
import numpy as np
import torch
import html
import logging
from datetime import datetime

# ----------------------------------------
# 2. Inisialisasi Aplikasi dan Konfigurasi
# ----------------------------------------

# Inisialisasi aplikasi FastAPI
# root_path digunakan jika aplikasi dijalankan di belakang reverse proxy
app = FastAPI(root_path="/yahya")

# Konfigurasi logging untuk memantau aktivitas dan error
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Koneksi ke instance Elasticsearch yang berjalan secara lokal
es = Elasticsearch("http://127.0.0.1:9200")

# Memuat model Sentence Transformer (SBERT) dari HuggingFace Hub
# Secara otomatis mendeteksi dan menggunakan GPU (CUDA) jika tersedia untuk performa terbaik
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("yahyaabd/allstats-search-mini-v1-1-mnrl").to(device)
logger.info(f"Model SBERT 'yahyaabd/allstats-search-mini-v1-1-mnrl-sts' berhasil dimuat di perangkat: {device}")

# ----------------------------------------
# 3. Konfigurasi Keamanan dan Parameter
# ----------------------------------------

# Kunci API statis untuk mengamankan endpoint
API_KEY = "akjba9wf4389fndi9j0998hd3hwe9h898aos8h128"
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)

# Bobot untuk menggabungkan skor dari BM25 dan pencarian semantik
ALPHA = 0.4  # Bobot untuk skor leksikal (BM25)
BETA = 1 - ALPHA   # Bobot untuk skor semantik (cosine similarity)

# Dictionary untuk memetakan parameter 'content' ke filter query Elasticsearch
CONTENT_FILTERS = {
    "table": [{"term": {"konten": "table"}}],
    "publication": [{"term": {"konten": "publication"}}, {"term": {"jenis": "softcopy"}}],
    "pressrelease": [{"term": {"konten": "pressrelease"}}, {"term": {"jenis": "pressrelease"}}],
    "infographic": [{"term": {"konten": "infographic"}}, {"term": {"jenis": "infographic"}}],
    "news": [{"term": {"konten": "news"}}, {"term": {"jenis": "news"}}],
    "microdata": [{"term": {"konten": "microdata"}}, {"term": {"jenis": "microdata"}}],
    "metadata": [{"term": {"konten": "metadata"}}, {"term": {"jenis": "metadata"}}],
    "kbli2020": [{"term": {"konten": "kbli2020"}}, {"term": {"jenis": "kbli2020"}}],
    "kbli2017": [{"term": {"konten": "kbli2017"}}, {"term": {"jenis": "kbli2017"}}],
    "kbli2015": [{"term": {"konten": "kbli2015"}}, {"term": {"jenis": "kbli2015"}}],
    "kbli2009": [{"term": {"konten": "kbli2009"}}, {"term": {"jenis": "kbli2009"}}],
    "kbki2015": [{"term": {"konten": "kbki2015"}}, {"term": {"jenis": "kbki2015"}}],
    "glosarium": [{"bool": {"should": [{"term": {"konten": "glosarium"}}, {"term": {"jenis": "glosarium"}}], "minimum_should_match": 1}}],
    "all": []
}


# ----------------------------------------
# 4. Model Data Pydantic
# ----------------------------------------
class Document(BaseModel):
    doc_id: str = Field(..., description="ID unik untuk dokumen, akan digunakan sebagai _id di Elasticsearch.")
    judul: str = Field(..., description="Judul dokumen yang akan di-encode menjadi vektor.")
    # Gunakan Dict[str, Any] untuk menampung field lain yang dinamis
    # additional_data: Dict[str, Any] = Field({}, description="Metadata tambahan untuk dokumen.")
    
    # Field-field spesifik sesuai mapping Elasticsearch
    corpus_id: Optional[str] = None
    deskripsi: Optional[str] = None
    id: Optional[str] = None
    jenis: Optional[str] = None
    konten: Optional[str] = None
    last_update: Optional[str] = None
    mfd: Optional[str] = None
    source: Optional[str] = None
    tgl_rilis: Optional[str] = None
    url: Optional[str] = None

# ----------------------------------------
# 4. Fungsi Helper dan Dependensi
# ----------------------------------------
def validate_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key

def validate_date(date_str: Optional[str], role: str = 'start') -> Optional[str]:
    """
    Fungsi helper untuk validasi format tanggal yang lebih fleksibel.
    Menerima format YYYY, YYYY-MM-DD, dll.
    Mengembalikan tanggal dalam format standar YYYY-MM-DD.
    'role' menentukan apakah tanggal adalah 'start' atau 'end' untuk menangani input tahun.
    """
    if date_str:
        # Logika khusus untuk format tahun saja
        if len(date_str) == 4 and date_str.isdigit():
            if role == 'end':
                # Untuk date_to, gunakan hari terakhir tahun itu
                return f"{date_str}-12-31"
            else:
                # Untuk date_from, gunakan hari pertama tahun itu
                return f"{date_str}-01-01"

        # Daftar format lain yang akan dicoba
        supported_formats = [
            "%Y-%m-%d",
            "%d-%m-%Y",
            "%Y/%m/%d",
            "%d/%m/%Y",
        ]

        for fmt in supported_formats:
            try:
                dt_object = datetime.strptime(date_str, fmt)
                return dt_object.strftime("%Y-%m-%d")
            except ValueError:
                continue
        
        raise HTTPException(status_code=400, detail=f"Invalid date format: '{date_str}'. Please use a supported format.")
    
    return None

def min_max_normalize(scores: np.ndarray) -> np.ndarray:
    if scores.size == 0:
        return np.array([])
    min_val, max_val = scores.min(), scores.max()
    if max_val == min_val:
        return np.zeros_like(scores, dtype=float)
    return (scores - min_val) / (max_val - min_val)


# ----------------------------------------
# 4. Fungsi dan Endpoint API
# ----------------------------------------

# Fungsi dependensi untuk memvalidasi API Key pada setiap request
def validate_api_key(api_key: str = Security(api_key_header)):
    """Memeriksa apakah API Key yang diberikan valid."""
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key

@app.get("/", response_class=HTMLResponse, summary="Halaman Utama API")
def read_root():
    """Menampilkan halaman HTML sederhana sebagai konfirmasi bahwa API berjalan."""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Allstats Semantic Search API</title>
        <style>body { font-family: sans-serif; padding: 2em; }</style>
    </head>
    <body>
        <h1>Allstats Semantic Search API</h1>
        <p>Layanan API untuk pencarian semantik BPS. Akses <strong>/docs</strong> untuk dokumentasi interaktif.</p>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# @app.get("/encode/", summary="Mengubah Teks menjadi Vektor Embedding")
# def encode_text(
#     text: str = Query(..., description="Teks yang akan diubah menjadi vektor."),
#     api_key: str = Depends(validate_api_key)
# ):
#     """
#     Endpoint ini menerima sebuah string teks, membersihkannya dari potensi
#     skrip HTML, lalu mengubahnya menjadi vektor embedding 384 dimensi
#     menggunakan model SBERT.
#     """
#     try:
#         sanitized_text = html.escape(text)
#         embeddings = model.encode(sanitized_text).tolist()
#         return {"text": sanitized_text, "embeddings": embeddings}
#     except Exception as e:
#         logger.error(f"Error saat encoding teks: {e}")
#         raise HTTPException(status_code=500, detail=f"Error during text encoding: {e}")

# @app.post("/index-document/", summary="Mengindeks Dokumen Baru dengan Vektor")
# def index_document(
#     document: Document,
#     index_name: str = "datacontent",
#     api_key: str = Depends(validate_api_key)
# ):
#     """
#     Endpoint untuk mengindeks satu dokumen.
#     - Menghasilkan vektor embedding dari field 'judul'.
#     - Menyimpan/memperbarui dokumen di Elasticsearch.
#     """
#     if model is None or es is None:
#         raise HTTPException(status_code=503, detail="Layanan tidak tersedia (Model atau DB Error).")

#     try:
#         # 1. Encode judul menjadi vektor
#         title_vector = model.encode(document.judul).tolist()

#         # 2. Siapkan body dokumen dari model Pydantic
#         # Gunakan .dict() untuk mengubah model menjadi dictionary
#         # exclude_unset=True akan mengabaikan field yang tidak diisi (opsional)
#         doc_body = document.dict(exclude={"doc_id"}, exclude_unset=True)
        
#         # Tambahkan field vektor ke dokumen
#         doc_body['title_embeddings_384'] = title_vector
        
#         # 3. Indeks dokumen ke Elasticsearch
#         # Menggunakan doc_id sebagai ID di Elasticsearch untuk upsert (update jika ada, create jika tidak)
#         response = es.index(index=index_name, id=document.doc_id, document=doc_body)
        
#         logger.info(f"Dokumen {document.doc_id} berhasil diindeks dengan hasil: {response.body.get('result', 'unknown')}")
#         return {"status": "sukses", "doc_id": document.doc_id, "result": response.body.get('result', 'unknown')}

#     except Exception as e:
#         logger.error(f"Gagal mengindeks dokumen {document.doc_id}: {e}", exc_info=True)
#         raise HTTPException(status_code=500, detail=f"Gagal mengindeks dokumen: {str(e)}")

# Endpoint semantic-search
@app.get("/semantic-search/", summary="Pencarian Hybrid (BM25 + Semantik)")
def semantic_search(
    keyword: str = Query(..., description="Kata kunci untuk pencarian."),
    threshold: float = Query(0.4, description="Ambang batas skor kemiripan minimum (0.0 - 1.0).", ge=0.0, le=1.0),
    alpha: float = Query(0.4, description="Weight for BM25 score (0.0 to 1.0)", ge=0.0, le=1.0),
    fuzzy: int = Query(0, description="number of fuzzy search (0 to 3)", ge=0, le=3),
    size: int = Query(10, description="Jumlah hasil per halaman.", ge=1, le=10000),
    from_: int = Query(0, description="Offset untuk paginasi.", ge=0, alias="from"),
    sort: str = Query("relevansi", description="Kriteria pengurutan: 'relevansi' atau 'tanggal'."),
    content: str = Query("datacontent", description="Filter jenis konten (e.g., 'table', 'publication', 'all')."),
    mfd: Optional[str] = Query(None, description="Filter berdasarkan kode MFD (4 digit)."),
    date_from: Optional[str] = Query(None, description="Filter tanggal mulai (YYYY-MM-DD)."),
    date_to: Optional[str] = Query(None, description="Filter tanggal akhir (YYYY-MM-DD)."),
    api_key: str = Depends(validate_api_key)
):
    """
    Endpoint untuk pencarian hybrid (BM25 + Semantik).
    Filter tanggal ditulis ulang untuk memastikan pencocokan dokumen berdasarkan tgl_rilis atau last_update.
    """
    try:
        # Log parameter input
        logger.info(f"Parameters: keyword={keyword}, content={content}, date_from={date_from}, date_to={date_to}, threshold={threshold}, mfd={mfd}")

        sanitized_keyword = html.escape(keyword.lower())
        input_embeddings = model.encode(sanitized_keyword)
        query_vector = input_embeddings.tolist()
        index_name = "glosarium" if content == "glosarium" else "datacontent,glosarium" if content == "all" else "datacontent"
        logger.info(f"Index used: {index_name}")

        # Membangun filter
        content_filters = CONTENT_FILTERS.get(content, []).copy()
        if mfd and mfd.isdigit() and len(mfd) == 4:
            content_filters.append({"term": {"mfd": mfd}})

        # Filter tanggal baru
        if date_from or date_to:
            content_filters.append({
                "bool": {
                    "should": [
                        {
                            "range": {
                                "tgl_rilis": {
                                    "gte": date_from,
                                    "lte": date_to,
                                    "format": "yyyy-MM-dd"  # Memastikan format tanggal
                                }
                            }
                        },
                        {
                            "range": {
                                "last_update": {
                                    "gte": date_from,
                                    "lte": date_to,
                                    "format": "yyyy-MM-dd"  # Memastikan format tanggal
                                }
                            }
                        }
                    ],
                    "minimum_should_match": 1
                }
            })
        logger.info(f"Content filters: {content_filters}")

        # Hitung jumlah dokumen yang lolos filter
        count_query = {"query": {"bool": {"filter": content_filters if content_filters else []}}}
        count_res = es.count(index=index_name, body=count_query)
        doc_count = count_res["count"]
        logger.info(f"Document count after filters: {doc_count}")

        # Logika pembersihan stopwords
        custom_stopwords = {"statistik", "data", "banyak", "jumlah", "total", "informasi", "angka", "penduduk", "menurut", "indonesia"}
        tokens = sanitized_keyword.lower().split()
        tokens = [word for word in tokens if word not in custom_stopwords]
        if not tokens:
            tokens = sanitized_keyword.lower().split()
        sanitized_keyword_bm25 = " ".join(tokens)
        query_length = len(tokens)
        logger.info(f"Sanitized BM25 keyword: {sanitized_keyword_bm25}")

        # Logika dinamis untuk query BM25
        # operator = "AND" if query_length == 2 else "OR"
        # min_should_match = "100%" if query_length <= 2 else "50%"
        # operator = "OR" if query_length <= 3 else "AND"
        # min_should_match = "100%" if query_length <= 3 else "75%" if query_length <= 5 else "25%"

        min_should_match = "100%" if query_length == 2 else "75%"
        operator = "OR" if query_length <= 3 else "AND"
        
        bm25_query = {
            "match": {
                "judul": {
                    "query": sanitized_keyword_bm25,
                    "operator": operator,
                    "fuzziness": fuzzy,
                    "minimum_should_match": min_should_match
                }
            }
        }

        # Logika dinamis untuk query semantik
        if doc_count > 10000:
            search_type = "knn"
            cosine_query = {
                "knn": {
                    "field": "title_embeddings_384",
                    "query_vector": query_vector,
                    "k": min(doc_count, 10000),
                    "num_candidates": min(doc_count, 10000)
                }
            }
        else:
            search_type = "script_score"
            cosine_query = {
                "script_score": {
                    "query": {"bool": {"filter": [{"exists": {"field": "title_embeddings_384"}}]}},
                    "script": {
                        "source": "(cosineSimilarity(params.query_vector, 'title_embeddings_384')+1)/2",
                        "params": {"query_vector": query_vector}
                    }
                }
            }
        logger.info(f"Search type: {search_type}")

        # Membangun query hybrid
        hybrid_query = {
            "query": {
                "bool": {
                    "should": [
                        {
                            "function_score": {
                                "query": bm25_query,
                                "boost": alpha,
                                "script_score": {"script": {"source": "_score"}}
                            }
                        },
                        {
                            "function_score": {
                                "query": cosine_query,
                                "boost": (1 - alpha)
                            }
                        }
                    ]
                }
            },
            "min_score": threshold,
            "from": from_,
            "size": size,
            "track_scores": True,
            "track_total_hits": True,
            "_source": ["judul", "deskripsi", "konten", "jenis", "img", "last_update", "tgl_rilis", "mfd", "url"]
        }

        if content_filters:
            hybrid_query["query"]["bool"]["filter"] = content_filters

        if sort == "relevansi":
            hybrid_query["sort"] = [{"_score": {"order": "desc"}}]
        elif sort == "tanggal":
            hybrid_query["sort"] = [{"last_update": {"order": "desc"}}]
        else:
            raise HTTPException(status_code=400, detail="Invalid sort value. Use 'relevansi' or 'tanggal'.")

        logger.info(f"Elasticsearch query: {hybrid_query}")
        page_res_obj = es.search(index=index_name, body=hybrid_query)
        logger.info(f"Elasticsearch response: {page_res_obj.body}")

        response_dict = page_res_obj.body
        hits = response_dict.get("hits", {}).get("hits", [])
        response_dict["took"] = response_dict.get("took", 0) / 1000

        if not hits:
            logger.warning("No hits returned")
            return response_dict

        # Pasca-pemrosesan skor
        raw_scores = np.array([hit["_score"] for hit in hits])
        normalized_scores = min_max_normalize(raw_scores)

        combined_hits = []
        for i, hit in enumerate(hits):
            hit_copy = hit.copy()
            hit_copy["_score"] = normalized_scores[i]
            combined_hits.append(hit_copy)

        response_dict["hits"]["hits"] = combined_hits
        return response_dict

    except HTTPException:
        raise
    except RequestError as e:
        logger.error(f"Elasticsearch error: {e.info}")
        raise HTTPException(status_code=500, detail=str(e.info))
    except Exception as e:
        logger.error(f"Error during hybrid search: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Unexpected error during search: {e}")
        
# ----------------------------------------
# 5. Menjalankan Aplikasi
# ----------------------------------------
if __name__ == "__main__":
    import uvicorn
    # Menjalankan server Uvicorn pada host 0.0.0.0 port 3700
    uvicorn.run(app, host="0.0.0.0", port=3700)