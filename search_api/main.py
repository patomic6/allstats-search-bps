from fastapi import FastAPI, HTTPException, Query, Security, Depends
from fastapi.security.api_key import APIKeyHeader
from fastapi.responses import HTMLResponse
from elasticsearch import Elasticsearch
import time
import html
import torch
import logging
from typing import Optional
import numpy as np
import json
from elasticsearch.exceptions import RequestError
from sentence_transformers import SentenceTransformer

app = FastAPI(root_path="/yahya") # sesuaikan root_path anda

# Koneksi ke Elasticsearch
es = Elasticsearch("http://127.0.0.1:9200")

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load Sentence Transformer model
# device = "cpu"
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("yahyaabd/allstats-search-mini-v1-1-mnrl-sts").to(device)
print(f"Model loaded on device: {device}")

# API Key yang diizinkan
API_KEY = "##############################" # buat api anda di sini
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)

# Bobot skoring
ALPHA = 0.4  # Bobot untuk BM25
BETA = 1-ALPHA   # Bobot untuk cosine similarity

# Fungsi untuk validasi API Key
def validate_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key

@app.get("/", response_class=HTMLResponse)
def read_root():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>allstats Search SBERT model</title>
    </head>
    <body>
        <h1>Allstats Search</h1>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/encode/")
def encode_text(
    text: str = Query(..., description="Text to encode using SBERT model"),
    api_key: str = Depends(validate_api_key)
):
    try:
        sanitized_text = html.escape(text)
        embeddings = model.encode(sanitized_text).tolist()
        return {"text": sanitized_text, "embeddings": embeddings}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during text encoding: {e}")

# Initialize model once at startup
# model = SentenceTransformer('yahyaabd/allstats-search-mini-v1-1-mnrl')

# Content filter mapping
CONTENT_FILTERS = {
    "table": [{"term": {"konten": "table"}}],
    "publication": [
        {"term": {"konten": "publication"}},
        {"term": {"jenis": "softcopy"}}
    ],
    "pressrelease": [
        {"term": {"konten": "pressrelease"}},
        {"term": {"jenis": "pressrelease"}}
    ],
    "infographic": [
        {"term": {"konten": "infographic"}},
        {"term": {"jenis": "infographic"}}
    ],
    "news": [
        {"term": {"konten": "news"}},
        {"term": {"jenis": "news"}}
    ],
    "microdata": [
        {"term": {"konten": "microdata"}},
        {"term": {"jenis": "microdata"}}
    ],
    "metadata": [
        {"term": {"konten": "metadata"}},
        {"term": {"jenis": "metadata"}}
    ],
    "kbli2020": [
        {"term": {"konten": "kbli2020"}},
        {"term": {"jenis": "kbli2020"}}
    ],
    "kbli2017": [
        {"term": {"konten": "kbli2017"}},
        {"term": {"jenis": "kbli2017"}}
    ],
    "kbli2015": [
        {"term": {"konten": "kbli2015"}},
        {"term": {"jenis": "kbli2015"}}
    ],
    "kbli2009": [
        {"term": {"konten": "kbli2009"}},
        {"term": {"jenis": "kbli2009"}}
    ],
    "kbki2015": [
        {"term": {"konten": "kbki2015"}},
        {"term": {"jenis": "kbki2015"}}
    ],
    "glosarium": [{
        "bool": {
            "should": [
                {"term": {"konten": "glosarium"}},
                {"term": {"jenis": "glosarium"}}
            ],
            "minimum_should_match": 1
        }
    }],
    "all": []
}

@app.get("/cosine-search-only/")
def semantic_search_only(
    keyword: str = Query(..., description="Keyword for semantic search (cosine similarity only)"),
    threshold: float = Query(0.3, description="Minimum similarity score threshold (0.0 to 1.0)", ge=0.0, le=1.0),
    size: int = Query(10, description="Number of results per page", ge=1, le=10000),
    from_: int = Query(0, description="Offset to start results from (for pagination)", ge=0, alias="from"),
    sort: str = Query("relevansi", description="Sort criteria: 'relevansi' (by score) or 'tanggal' (by last_update)"),
    content: str = Query("datacontent", description="Content type: 'glosarium', 'all', 'table', 'publication', etc."),
    mfd: Optional[str] = Query(None, description="Filter by MFD (4-digit code)"),
    date_from: Optional[str] = Query(None, description="Start date filter (YYYY-MM-DD)"),
    date_to: Optional[str] = Query(None, description="End date filter (YYYY-MM-DD)"),
    api_key: str = Depends(validate_api_key)
):
    try:
        sanitized_keyword = html.escape(keyword.lower())
        input_embeddings = model.encode(sanitized_keyword)
        query_vector = input_embeddings.tolist()
        index_name = "glosarium" if content == "glosarium" else "datacontent,glosarium" if content == "all" else "datacontent"

        # Build content filters
        content_filters = CONTENT_FILTERS.get(content, []).copy()
        if mfd and mfd.isdigit() and len(mfd) == 4:
            content_filters.append({"term": {"mfd": mfd}})
        if date_from and date_to:
            content_filters.append({
                "bool": {
                    "should": [
                        {"range": {"tgl_rilis": {"gte": date_from, "lte": date_to}}},
                        {"range": {"last_update": {"gte": date_from, "lte": date_to}}}
                    ],
                    "minimum_should_match": 1
                }
            })

        logger.debug("Applied filters: %s, mfd: %s", json.dumps(content_filters, indent=2), mfd)

        # Count total documents matching filters
        count_query = {
            "query": {
                "bool": {
                    "filter": content_filters if content_filters else []
                }
            }
        }
        count_res = es.count(index=index_name, body=count_query)
        doc_count = count_res["count"]
        logger.debug("Document count for mfd %s: %d", mfd, doc_count)

        # Choose search type based on document count
        if doc_count > 10000:
            search_type = "knn"
            # Set k and num_candidates to doc_count to capture all possible documents
            semantic_query = {
                "knn": {
                    "field": "title_embeddings_384",
                    "query_vector": query_vector,
                    "k": 10000,  # Dynamically set to total document count
                    "num_candidates": 10000  # Ensure all documents are considered
                }
            }
        else:
            search_type = "script_score"
            semantic_query = {
                "script_score": {
                    "query": {
                        "bool": {
                            "filter": [
                                {"exists": {"field": "title_embeddings_384"}}
                            ]
                        }
                    },
                    "script": {
                        "source": "(cosineSimilarity(params.query_vector, 'title_embeddings_384')+1)/2",
                        "params": {"query_vector": query_vector}
                    }
                }
            }

        # Build the Elasticsearch query with min_score to filter by threshold
        query = {
            "query": semantic_query,
            "min_score": threshold,  # Ensure only documents with score >= threshold are returned
            "from": from_,
            "size": size,  # Limit results per page for pagination
            "track_scores": True,
            "track_total_hits": True,  # Track total number of documents meeting threshold
            "_source": ["judul", "deskripsi", "konten", "jenis", "img", "last_update", "tgl_rilis", "mfd", "url"]
        }

        if content_filters:
            query["query"] = {
                "bool": {
                    "must": [semantic_query],
                    "filter": content_filters
                }
            }

        # Apply sorting
        if sort == "relevansi":
            query["sort"] = [{"_score": {"order": "desc"}}]
        elif sort == "tanggal":
            query["sort"] = [{"last_update": {"order": "desc"}}]
        else:
            raise HTTPException(status_code=400, detail="Invalid sort value. Use 'relevansi' or 'tanggal'.")

        logger.debug("Executing Elasticsearch query for keyword: %s, mfd: %s, doc_count: %d, search_type: %s, threshold: %f",
                     sanitized_keyword, mfd, doc_count, search_type, threshold)
        page_res = es.search(index=index_name, body=query)

        hits = page_res["hits"]["hits"]
        # Log the number of hits and their scores to verify threshold filtering
        if hits:
            scores = [hit["_score"] for hit in hits]
            logger.debug("Returned %d hits with scores: %s", len(hits), scores)
            # Verify all returned documents meet the threshold
            if any(score < threshold for score in scores):
                logger.warning("Some documents returned with scores below threshold: %s", scores)

        if not hits:
            logger.info("No documents returned for keyword: %s, mfd: %s, threshold: %f", sanitized_keyword, mfd, threshold)
            return {
                "search_type": search_type,
                "took": page_res["took"] / 1000,
                "timed_out": page_res["timed_out"],
                "_shards": page_res["_shards"],
                "hits": {
                    "total": {"value": page_res["hits"]["total"]["value"], "relation": page_res["hits"]["total"]["relation"]},
                    "max_score": page_res["hits"]["max_score"],
                    "hits": []
                }
            }

        response = {
            "search_type": search_type,
            "took": page_res["took"] / 1000,
            "timed_out": page_res["timed_out"],
            "_shards": page_res["_shards"],
            "hits": {
                "total": {"value": page_res["hits"]["total"]["value"], "relation": page_res["hits"]["total"]["relation"]},
                "max_score": page_res["hits"]["max_score"],
                "hits": hits
            }
        }

        if not hits and page_res["hits"]["total"]["value"] > 0:
            logger.warning(f"No hits returned despite {page_res['hits']['total']['value']} total matches for mfd: {mfd}")

        # Log the total number of documents meeting the threshold
        logger.info("Total documents with score >= %f: %d", threshold, page_res["hits"]["total"]["value"])

        return response

    except RequestError as e:
        logger.error(f"Elasticsearch error for mfd {mfd}: {e.info}")
        raise HTTPException(status_code=500, detail=f"Elasticsearch error: {e.info}")
    except Exception as e:
        logger.error(f"Error during semantic search for mfd {mfd}: {e}")
        raise HTTPException(status_code=500, detail=f"Error during semantic search: {e}")
        
# endpoint pencarian hybrid
@app.get("/semantic-search/")
def semantic_search(
    keyword: str = Query(..., description="Keyword for hybrid search (BM25 + Cosine)"),
    threshold: float = Query(0.4, description="Minimum similarity score threshold (0.0 to 1.0)", ge=0.0, le=1.0),
    size: int = Query(10, description="Number of results per page", ge=1, le=10000),
    from_: int = Query(0, description="Offset to start results from (for pagination)", ge=0, alias="from"),
    sort: str = Query("relevansi", description="Sort criteria: 'relevansi' (by score) or 'tanggal' (by last_update)"),
    content: str = Query("datacontent", description="Content type: 'glosarium', 'all', 'table', 'publication', etc."),
    mfd: Optional[str] = Query(None, description="Filter by MFD (4-digit code)"),
    date_from: Optional[str] = Query(None, description="Start date filter (YYYY-MM-DD)"),
    date_to: Optional[str] = Query(None, description="End date filter (YYYY-MM-DD)"),
    api_key: str = Depends(validate_api_key)
):
    try:
        sanitized_keyword = html.escape(keyword.lower())
        input_embeddings = model.encode(sanitized_keyword)
        
        # # if len(input_embeddings) != 384:
        # if len(input_embeddings) != 768:
        #     logger.error(f"Query vector has {len(input_embeddings)} dimensions, expected 384")
        #     raise HTTPException(status_code=500, detail="Query vector dimension mismatch")
        # if np.any(np.isnan(input_embeddings)) or np.any(np.isinf(input_embeddings)):
        #     logger.error("Invalid values in query vector")
        #     raise HTTPException(status_code=400, detail="Invalid query vector: contains NaN or infinite values")
        
        query_vector = input_embeddings.tolist()
        index_name = "glosarium" if content == "glosarium" else "datacontent,glosarium" if content == "all" else "datacontent"

        content_filters = CONTENT_FILTERS.get(content, []).copy()
        if mfd and mfd.isdigit() and len(mfd) == 4:
            content_filters.append({"term": {"mfd": mfd}})
        if date_from and date_to:
            content_filters.append({
                "bool": {
                    "should": [
                        {"range": {"tgl_rilis": {"gte": date_from, "lte": date_to}}},
                        {"range": {"last_update": {"gte": date_from, "lte": date_to}}}
                    ],
                    "minimum_should_match": 1
                }
            })

        logger.debug("Applied filters: %s, mfd: %s", json.dumps(content_filters, indent=2), mfd)

        count_query = {
            "query": {
                "bool": {
                    "filter": content_filters if content_filters else []
                }
            }
        }
        count_res = es.count(index=index_name, body=count_query)
        doc_count = count_res["count"]
        
        # Hitung jumlah kata dalam kueri
        query_length = len(sanitized_keyword.split())
        
        # # Tentukan operator berdasarkan panjang kueri
        # operator = "OR" if query_length <= 2 else "AND"
        # min_should_match = "50%" if query_length <= 2 else "50%"

        # # tambahkan pembersihan stopwords "statistik" "data" dll
        # ## ada kecenderungan lebih mengikuti stopword dibanding kata kunci misalnya banyak freelancer --> 'banyak' lebih muncul
        # bm25_query = {
        #     "match": {
        #         "judul": {
        #             "query": sanitized_keyword,
        #             "operator": operator,
        #             "minimum_should_match": min_should_match
        #         }
        #     }
        # }

        # Daftar stopwords khusus BPS/statistik
        custom_stopwords = {"statistik", "data", "banyak", "jumlah", "total", "informasi", "angka"}
        
        # Tokenisasi kueri
        tokens = sanitized_keyword.lower().split()
        query_length = len(tokens)
        
        # Hapus stopwords hanya jika kueri terdiri dari 2 kata
        if query_length == 2:
            tokens = [word for word in tokens if word not in custom_stopwords]
            sanitized_keyword = " ".join(tokens)
            query_length = len(tokens)  # perbarui panjang kueri setelah stopword removal
        
        # Tentukan operator berdasarkan panjang kueri
        # operator = "OR" if query_length <= 2 else "AND"
        operator = "OR"
        min_should_match = "100%" if query_length <= 2 else "50%"
        
        # Susun kueri BM25
        bm25_query = {
            "match": {
                "judul": {
                    "query": sanitized_keyword,
                    "operator": operator,
                    "fuzziness": "0",
                    "minimum_should_match": min_should_match
                }
            }
        }

        if doc_count > 10000:
            search_type = "knn"
            cosine_query = {
                "knn": {
                    "field": "title_embeddings_384",
                    "query_vector": query_vector,
                    "k": 1000,
                    "num_candidates": 10000
                }
            }
        else:
            search_type = "script_score"
            cosine_query = {
                "script_score": {
                    "query": {
                        "bool": {
                            "filter": [
                                {"exists": {"field": "title_embeddings_384"}}
                            ]
                        }
                    },
                    "script": {
                        "source": "(cosineSimilarity(params.query_vector, 'title_embeddings_384')+1)/2",
                        "params": {"query_vector": query_vector}
                    }
                }
            }

        hybrid_query = {
            "query": {
                "bool": {
                    "should": [
                        {
                            "function_score": {
                                "query": bm25_query,
                                "script_score": {
                                    "script": {
                                        "source": "_score"
                                    }
                                },
                                # "boost": ALPHA
                            }
                        },
                        cosine_query
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

        logger.debug("Executing Elasticsearch query for keyword: %s, mfd: %s, doc_count: %d, search_type: %s",
                     sanitized_keyword, mfd, doc_count, search_type)
        page_res = es.search(index=index_name, body=hybrid_query)

        hits = page_res["hits"]["hits"]
        if not hits:
            return {
                "search_type": search_type,
                "took": page_res["took"] / 1000,
                "timed_out": page_res["timed_out"],
                "_shards": page_res["_shards"],
                "hits": {
                    "total": {"value": page_res["hits"]["total"]["value"], "relation": page_res["hits"]["total"]["relation"]},
                    "max_score": page_res["hits"]["max_score"],
                    "hits": []
                }
            }

        # Gunakan ALPHA dan BETA untuk normalisasi skoring
        bm25_scores = np.array([hit["_score"] * ALPHA / (ALPHA + BETA) for hit in hits])
        cosine_scores = np.array([hit["_score"] * BETA / (ALPHA + BETA) for hit in hits])
        # bm25_scores = np.array([hit["_score"] for hit in hits])
        # cosine_scores = np.array([hit["_score"] for hit in hits])

        def min_max_normalize(scores):
            if not scores.size:
                return scores
            min_score, max_score = scores.min(), scores.max()
            if max_score == min_score:
                return np.zeros_like(scores)
            return (scores - min_score) / (max_score - min_score)

        norm_bm25_scores = min_max_normalize(bm25_scores)
        norm_cosine_scores = min_max_normalize(cosine_scores)
        # norm_cosine_scores = cosine_scores

        combined_hits = [hit.copy() for hit in hits]
        for hit, norm_bm25, norm_cosine in zip(combined_hits, norm_bm25_scores, norm_cosine_scores):
            hit["_score"] = ALPHA * norm_bm25 + BETA * norm_cosine
            # hit["_score"] = norm_bm25 + norm_cosine

        if sort == "tanggal":
            combined_hits.sort(key=lambda x: x["_score"], reverse=True)

        response = {
            "search_type": search_type,
            "took": page_res["took"] / 1000,
            "timed_out": page_res["timed_out"],
            "_shards": page_res["_shards"],
            "hits": {
                "total": {"value": page_res["hits"]["total"]["value"], "relation": page_res["hits"]["total"]["relation"]},
                "max_score": max([hit["_score"] for hit in combined_hits], default=0.0),
                "hits": combined_hits
            }
        }

        if not combined_hits and page_res["hits"]["total"]["value"] > 0:
            logger.warning(f"No hits returned despite {page_res['hits']['total']['value']} total matches for mfd: {mfd}")

        return response

    except RequestError as e:
        logger.error(f"Elasticsearch error for mfd {mfd}: {e.info}")
        raise HTTPException(status_code=500, detail=f"Elasticsearch error: {e.info}")
    except Exception as e:
        logger.error(f"Error during hybrid search for mfd {mfd}: {e}")
        raise HTTPException(status_code=500, detail=f"Error during hybrid search: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3700)
