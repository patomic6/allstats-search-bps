import pandas as pd
import json
import requests
from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer
from datetime import datetime
import logging
import base64
import time
import re
import math
import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ElasticsearchIndexer:
    def __init__(self, es_host='http://127.0.0.1:9200', index_name='datacontent', bypass_date_filter=False):
        self.es = Elasticsearch([es_host])
        self.index_name = index_name
        self.batch_size = 2500
        self.api_key = '7a62d6af2de1e805bc5f44d8a0f6ad17'  # Replace with your BPS API key
        self.embedding_model = SentenceTransformer('yahyaabd/allstats-search-mini-v1-1-mnrl')
        self.domain_names = self.fetch_domain_names()
        self.bypass_date_filter = bypass_date_filter
        # self.ensure_index()
    # bagian ini digunakan hanya jika ingin *membuat index dari awal* untuk memastikan bahwa indeks Elasticsearch sudah ada dengan mapping yang benar, termasuk sinonim.
    # def ensure_index(self):
    #     """Ensure the Elasticsearch index exists with synonym mapping."""
    #     mapping = {
    #         "settings": {
    #             "analysis": {
    #                 "filter": {
    #                     "synonym_filter": {
    #                         "type": "synonym",
    #                         "synonyms_path": "analysis/synonyms.txt"
    #                     }
    #                 },
    #                 "analyzer": {
    #                     "synonym_analyzer": {
    #                         "tokenizer": "standard",
    #                         "filter": [
    #                             "lowercase",
    #                             "synonym_filter"
    #                         ]
    #                     }
    #                 }
    #             }
    #         },
    #         "mappings": {
    #             "properties": {
    #                 "id": {"type": "text"},
    #                 "konten": {"type": "keyword"},
    #                 "jenis": {"type": "keyword"},
    #                 "judul": {
    #                     "type": "text",
    #                     "analyzer": "synonym_analyzer"
    #                 },
    #                 "deskripsi": {
    #                     "type": "text",
    #                     "analyzer": "synonym_analyzer"
    #                 },
    #                 "title_embeddings_384": {
    #                     "type": "dense_vector",
    #                     "dims": 384,
    #                     "index": True
    #                 },
    #                 "mfd": {"type": "keyword"},
    #                 "domain_name": {"type": "keyword"},
    #                 "tgl_rilis": {
    #                     "type": "date",
    #                     "format": "yyyy-MM-dd||yyyy/MM/dd||dd-MM-yyyy||dd/MM/yyyy||yyyy-MM-dd HH:mm:ss||epoch_millis"
    #                 },
    #                 "last_update": {
    #                     "type": "date",
    #                     "format": "yyyy-MM-dd||yyyy/MM/dd||dd-MM-yyyy||dd/MM/yyyy||yyyy-MM-dd HH:mm:ss||epoch_millis"
    #                 },
    #                 "source": {"type": "keyword"},
    #                 "url": {"type": "keyword"}
    #             }
    #         }
    #     }
    #     try:
    #         if self.es.indices.exists(index=self.index_name):
    #             logger.warning(f"Index {self.index_name} exists, deleting to apply synonym mapping")
    #             self.es.indices.delete(index=self.index_name)
    #         self.es.indices.create(index=self.index_name, body=mapping)
    #         logger.info(f"Created index {self.index_name} with synonym mapping")
    #     except Exception as e:
    #         logger.error(f"Failed to create index {self.index_name}: {e}")
    #         raise

    def get_last_indexed_date_for_mfd(self, mfd):
        """Retrieve the latest last_update date for a specific MFD."""
        try:
            response = self.es.options(ignore_status=[404]).search(
                index=self.index_name,
                body={
                    "query": {"term": {"mfd": mfd}},
                    "aggs": {"max_date": {"max": {"field": "last_update"}}},
                    "size": 0
                }
            )
            max_date = response['aggregations']['max_date'].get('value_as_string')
            if max_date:
                logger.info(f"Latest last_update for MFD {mfd}: {max_date}")
                return max_date
            logger.info(f"No data found for MFD {mfd}")
        except Exception as e:
            logger.warning(f"Error retrieving last_update for MFD {mfd}: {e}")
        return None

    def save_last_indexed_date(self, date):
        """Save the last indexed date to Elasticsearch."""
        try:
            response = self.es.index(
                index=self.index_name,
                id='_metadata',
                body={'last_indexed_date': date},
                refresh="true"
            )
            logger.info(f"Updated global last indexed date: {date}, Response: {response.get('result')}")
        except Exception as e:
            logger.error(f"Failed to save global last indexed date: {e}")
            raise

    def clean_title(self, title):
        """Clean title for URL formatting."""
        if title is None or (isinstance(title, float) and math.isnan(title)):
            logger.warning("Title is None or NaN, returning empty string")
            return ""
        title = str(title).strip()
        if not title:
            logger.warning("Title is empty, returning empty string")
            return ""
        title = title.lower()
        title = re.sub(r'[^\w\s-]', '', title)
        return title.replace(' ', '-')[:100]

    def encode_table_id(self, table_id, table_source):
        """Encode table ID with table source."""
        return base64.b64encode(f"{table_id}#{table_source}".encode()).decode()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(requests.RequestException),
        before_sleep=lambda retry_state: logger.warning(
            f"Retrying API call (attempt {retry_state.attempt_number}/3) due to {retry_state.outcome.exception()}"
        )
    )
    def fetch_api_response(self, url):
        """Fetch API response with retries."""
        logger.debug(f"Calling API: {url}")
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()

    @lru_cache(maxsize=1000)
    def mfdName(self, mfd):
        """Fetch domain URL for given MFD with caching."""
        if mfd == "0000":
            return "https://www.bps.go.id"

        try:
            mfd = str(mfd).zfill(4)
        except:
            return None

        try:
            if mfd.endswith("00"):
                url = f"https://webapi.bps.go.id/v1/api/domain/type/prov/key/{self.api_key}/"
                resp = self.fetch_api_response(url)
                data_list = resp.get("data", [])
                if isinstance(data_list, list) and len(data_list) > 1:
                    for item in data_list[1]:
                        domain_id = str(item.get("domain_id", "")).zfill(4)
                        if domain_id == mfd:
                            return item.get("domain_url")
            else:
                prov = mfd[:2] + "00"
                url = f"https://webapi.bps.go.id/v1/api/domain/type/kabbyprov/key/{self.api_key}/prov/{prov}"
                resp = self.fetch_api_response(url)
                data_list = resp.get("data", [])
                if isinstance(data_list, list) and len(data_list) > 1:
                    for item in data_list[1]:
                        domain_id = str(item.get("domain_id", "")).zfill(4)
                        if domain_id == mfd:
                            return item.get("domain_url")
        except Exception as e:
            logger.error(f"Error in mfdName({mfd}): {e}")
        return None

    def fetch_domain_names(self):
        """Fetch domain names from BPS API."""
        domain_names = {}
        try:
            prov_url = f"https://webapi.bps.go.id/v1/api/domain/type/prov/key/{self.api_key}/"
            prov_resp = self.fetch_api_response(prov_url)
            prov_data = prov_resp.get("data", [])
            if isinstance(prov_data, list) and len(prov_data) > 1:
                for item in prov_data[1]:
                    domain_id = str(item.get("domain_id", "")).zfill(4)
                    domain_name = item.get("domain_name", "")
                    if domain_name:
                        domain_names[domain_id] = domain_name

            for prov_id in domain_names.keys():
                if prov_id.endswith("00"):
                    kab_url = f"https://webapi.bps.go.id/v1/api/domain/type/kabbyprov/key/{self.api_key}/prov/{prov_id}"
                    kab_resp = self.fetch_api_response(kab_url)
                    kab_data = kab_resp.get("data", [])
                    if isinstance(kab_data, list) and len(kab_data) > 1:
                        for item in kab_data[1]:
                            domain_id = str(item.get("domain_id", "")).zfill(4)
                            domain_name = item.get("domain_name", "")
                            if domain_name:
                                domain_names[domain_id] = domain_name
            logger.info(f"Fetched {len(domain_names)} domain names")
        except Exception as e:
            logger.error(f"Failed to fetch domain names: {e}")
        return domain_names

    def fetch_all_mfds(self):
        """Fetch all MFDs from BPS API."""
        mfds = ["0000"]
        try:
            prov_url = f"https://webapi.bps.go.id/v1/api/domain/type/prov/key/{self.api_key}/"
            prov_resp = self.fetch_api_response(prov_url)
            prov_data = prov_resp.get("data", [])
            if isinstance(prov_data, list) and len(prov_data) > 1:
                for item in prov_data[1]:
                    domain_id = str(item.get("domain_id", "")).zfill(4)
                    mfds.append(domain_id)

            for prov_id in mfds[1:]:
                kab_url = f"https://webapi.bps.go.id/v1/api/domain/type/kabbyprov/key/{self.api_key}/prov/{prov_id}"
                kab_resp = self.fetch_api_response(kab_url)
                kab_data = kab_resp.get("data", [])
                if isinstance(kab_data, list) and len(kab_data) > 1:
                    for item in kab_data[1]:
                        domain_id = str(item.get("domain_id", "")).zfill(4)
                        mfds.append(domain_id)
            logger.info(f"Fetched {len(mfds)} MFDs")
        except Exception as e:
            logger.error(f"Failed to fetch MFDs: {e}")
        return mfds

    def format_data(self, row, title_embedding, table_source="1"):
        """Format a row for Elasticsearch indexing."""
        domain_id = row.get('domain_id', '0000')
        table_id = str(row['table_id'])
        doc_id = f"table_website_{domain_id}_{table_id}"
        domain_url = self.mfdName(domain_id)
        encoded_table_id = self.encode_table_id(table_id, table_source)
        cleaned_title = self.clean_title(row['title'])
        domain_name = self.domain_names.get(domain_id, "")

        updt_date = row.get('updt_date')
        formatted_date = None
        if updt_date and updt_date != '':
            try:
                parsed_date = pd.to_datetime(updt_date)
                if pd.notna(parsed_date):
                    formatted_date = parsed_date.strftime('%Y-%m-%d')
            except Exception as e:
                logger.warning(f"Failed to format date for table_id {table_id}: {updt_date}, Error: {e}")

        return {
            "_index": self.index_name,
            "_id": doc_id,
            "_source": {
                "id": table_id,
                "judul": row['title'],
                "deskripsi": row['title'],
                "title_embeddings_384": title_embedding.tolist(),
                "url": f"{domain_url}/statistics-table/{table_source}/{encoded_table_id}/{cleaned_title}.html",
                "tgl_rilis": None,
                "last_update": formatted_date,
                "mfd": domain_id,
                "domain_name": domain_name,
                "jenis": "statictable",
                "konten": "table",
                "source": table_source
            }
        }

    def index_dataframe(self, df):
        """Index DataFrame data into Elasticsearch using bulk."""
        start_time = time.time()
        success_count = 0
        failure_count = 0
        error_messages = []
        batch_titles = []
        batch_data = []
        latest_date = 0
        doc_ids = set()

        for idx, row in df.iterrows():
            updt_date = row.get('updt_date')
            item_date = 0
            if updt_date and updt_date != '':
                try:
                    parsed_date = pd.to_datetime(updt_date)
                    if pd.notna(parsed_date):
                        item_date = parsed_date.timestamp()
                except Exception as e:
                    logger.warning(f"Failed to parse date for table_id {row.get('table_id')}: {updt_date}, Error: {e}")

            title = str(row['title']) if pd.notna(row['title']) else ""
            if not title:
                logger.warning(f"Title is empty for table_id: {row.get('table_id')}, skipping")
                continue

            doc_id = f"table_website_{row.get('domain_id', '0000')}_{row.get('table_id')}"
            if doc_id in doc_ids:
                logger.warning(f"Duplicate doc_id detected: {doc_id}, skipping")
                continue
            doc_ids.add(doc_id)

            batch_titles.append(title)
            batch_data.append(row)
            latest_date = max(latest_date, item_date)

            if len(batch_data) >= self.batch_size:
                embeddings = self.embedding_model.encode(batch_titles, show_progress_bar=False)
                actions = [
                    self.format_data(row, embedding, table_source="1")
                    for row, embedding in zip(batch_data, embeddings)
                ]
                successes, errors = helpers.bulk(self.es, actions, refresh="true")
                success_count += successes
                failure_count += len(errors)
                if errors:
                    for error in errors:
                        error_msg = f"ID: {error.get('_id', 'unknown')}, Error: {json.dumps(error)}"
                        logger.error(error_msg)
                        error_messages.append(error_msg)
                batch_data = []
                batch_titles = []
                doc_ids.clear()
                gc.collect()

        if batch_data:
            embeddings = self.embedding_model.encode(batch_titles, show_progress_bar=False)
            actions = [
                self.format_data(row, embedding, table_source="1")
                for row, embedding in zip(batch_data, embeddings)
            ]
            successes, errors = helpers.bulk(self.es, actions, refresh="true")
            success_count += successes
            failure_count += len(errors)
            if errors:
                for error in errors:
                    error_msg = f"ID: {error.get('_id', 'unknown')}, Error: {json.dumps(error)}"
                    logger.error(error_msg)
                    error_messages.append(error_msg)
            gc.collect()

        if batch_titles:
            sample_text = batch_titles[0]
            sample_embedding = self.embedding_model.encode([sample_text], show_progress_bar=False)[0]
            logger.info(
                f"Sample embedding verification: text='{sample_text}', "
                f"embedding_shape={sample_embedding.shape}, embedding_sample={sample_embedding[:5]}"
            )

        if success_count > 0 and latest_date > 0:
            latest_date_str = datetime.fromtimestamp(latest_date).strftime('%Y-%m-%d %H:%M:%S')
            self.save_last_indexed_date(latest_date_str)

        execution_time = time.time() - start_time
        result = (
            f"Indexing completed. "
            f"Success: {success_count}, Failed: {failure_count}, Time: {round(execution_time, 2)} seconds. "
            f"Errors: {'; '.join(error_messages) if error_messages else 'None'}"
        )
        logger.info(result)
        return result

    def fetch_mfd_data(self, mfd, base_url):
        """Fetch data for a single MFD with optional date filtering."""
        if not self.bypass_date_filter:
            last_indexed_date = self.get_last_indexed_date_for_mfd(mfd)
            last_indexed_timestamp = pd.to_datetime(last_indexed_date).timestamp() if last_indexed_date else 0
            logger.info(f"Processing MFD {mfd} with last indexed timestamp: {last_indexed_timestamp}")
        else:
            last_indexed_timestamp = 0
            logger.info(f"Processing MFD {mfd} with date filtering bypassed")

        page = 1
        mfd_data = []
        while True:
            try:
                url = base_url.format(mfd=mfd, page=page, api_key=self.api_key)
                logger.info(f"Fetching API data for MFD {mfd}, page {page}")
                response = self.fetch_api_response(url)

                if response.get("status") != "OK" or response.get("data-availability") != "available":
                    logger.warning(f"Invalid API response for MFD {mfd}, page {page}: {response.get('status')}")
                    break

                metadata = response.get("data", [])[0]
                table_data = response.get("data", [])[1]

                for item in table_data:
                    item["domain_id"] = mfd
                    updt_date = item.get("updt_date")
                    item_timestamp = 0
                    if updt_date and updt_date != '':
                        try:
                            parsed_date = pd.to_datetime(updt_date, errors='coerce')
                            if pd.notna(parsed_date):
                                item_timestamp = parsed_date.timestamp()
                            else:
                                logger.warning(f"Invalid date format for table_id {item.get('table_id')}: {updt_date}")
                        except Exception as e:
                            logger.warning(f"Failed to parse date for table_id {item.get('table_id')}: {updt_date}, Error: {e}")

                    if self.bypass_date_filter or item_timestamp > last_indexed_timestamp:
                        mfd_data.append(item)

                total_pages = metadata.get("pages", 1)
                logger.info(f"MFD {mfd}: Page {page} of {total_pages}, retrieved {len(table_data)} records, kept {len(mfd_data)} after date filter")
                if page >= total_pages:
                    break
                page += 1

            except Exception as e:
                logger.error(f"Failed to fetch data for MFD {mfd}, page {page}: {e}")
                break

        return mfd, mfd_data

def fetch_api_data(indexer):
    """Fetch and index data from BPS Web API with parallel MFD processing."""
    mfds = indexer.fetch_all_mfds()
    base_url = "https://webapi.bps.go.id/v1/api/list/model/statictable/lang/ind/domain/{mfd}/page/{page}/key/{api_key}/"
    total_success = 0
    total_failure = 0
    total_errors = []

    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_mfd = {
            executor.submit(indexer.fetch_mfd_data, mfd, base_url): mfd
            for mfd in mfds
        }
        for future in as_completed(future_to_mfd):
            mfd = future_to_mfd[future]
            try:
                mfd, mfd_data = future.result()
                if mfd_data:
                    df = pd.DataFrame(mfd_data, columns=['table_id', 'title', 'updt_date', 'excel', 'domain_id'])
                    df = df.drop_duplicates(subset=['table_id', 'domain_id'])
                    df = df.dropna(subset=['title'])
                    df = df[df['title'].str.strip() != ""]
                    df['updt_date'] = pd.to_datetime(df['updt_date'], errors='coerce').astype(str).replace('NaT', '')
                    df['table_id'] = df['table_id'].astype(str)
                    df['domain_id'] = df['domain_id'].astype('category')

                    logger.info(f"Processing {len(df)} records for MFD {mfd}")
                    result = indexer.index_dataframe(df)
                    success = int(result.split("Success: ")[1].split(",")[0])
                    failure = int(result.split("Failed: ")[1].split(",")[0])
                    errors = result.split("Errors: ")[1].split("; ") if "Errors: " in result and result.split("Errors: ")[1] != "None" else []
                    total_success += success
                    total_failure += failure
                    total_errors.extend(errors)
                    del df
                    gc.collect()
                else:
                    logger.warning(f"No new data to index for MFD {mfd} after date filtering")
            except Exception as e:
                logger.error(f"Failed to process MFD {mfd}: {e}")
                total_errors.append(f"MFD {mfd}: {str(e)}")
                total_failure += 1

    logger.info(f"Total indexing: Success={total_success}, Failed={total_failure}, Errors={' '.join(total_errors) if total_errors else 'None'}")
    return total_success, total_failure, total_errors

def main():
    try:
        # Set bypass_date_filter=True to index all data, False to use date filtering
        indexer = ElasticsearchIndexer(bypass_date_filter=False)
        success, failure, errors = fetch_api_data(indexer)
        print(f"Indexing completed. Success: {success}, Failed: {failure}, Errors: {' '.join(errors) if errors else 'None'}")
    except Exception as e:
        logger.error(f"Failed to execute main: {e}")
    finally:
        indexer.es.close()

if __name__ == "__main__":
    main()