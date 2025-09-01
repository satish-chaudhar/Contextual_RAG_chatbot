# ingestion.py
from __future__ import annotations
import os
import sys
import hashlib
import argparse
import pathlib
import logging
import requests
import time
from typing import List
from PyPDF2 import PdfReader
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

try:
    from llama_index.core import Document, StorageContext, Settings, VectorStoreIndex
except Exception:
    from llama_index import Document, StorageContext, Settings, VectorStoreIndex

try:
    from llama_index.vector_stores.postgres import PGVectorStore
except Exception as e:
    raise RuntimeError('Missing llama-index-vector-stores-postgres. Install it in the container.') from e

# Configure logging at the top
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
log = logging.getLogger('ingest')

# Define environment variables at the top
DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://postgres:admin@postgres:5432/postgres')
OLLAMA_URL = os.getenv('OLLAMA_URL', 'http://ollama:11434')
EMBED_MODEL = os.getenv('EMBED_MODEL', 'nomic-embed-text')

class SimpleOllamaEmbedding:
    def __init__(self, model: str, url: str):
        self.model = model
        self.url = url.rstrip("/")

    def _embed(self, text: str, max_retries: int = 5):
        for attempt in range(max_retries):
            try:
                r = requests.post(
                    f"{self.url}/api/embeddings",
                    json={"model": self.model, "prompt": text},
                    timeout=300
                )
                r.raise_for_status()
                data = r.json()
                embedding = data.get("embedding", [])
                if not embedding:
                    raise ValueError("No embedding returned from Ollama")
                return embedding
            except requests.RequestException as e:
                if attempt == max_retries - 1:
                    raise
                log.warning(f"Embedding failed (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {2 ** attempt} seconds...")
                time.sleep(2 ** attempt)

    def _embed_batch(self, texts: List[str], max_retries: int = 5):
        with ThreadPoolExecutor() as executor:
            future_to_text = {executor.submit(self._embed, text, max_retries): text for text in texts}
            return [future.result() for future in as_completed(future_to_text)]

    def get_text_embedding(self, text: str):
        return self._embed(text)

    def get_query_embedding(self, query: str):
        return self._embed(query)

    def get_text_embedding_batch(self, texts: List[str], show_progress: bool = False):
        return self._embed_batch(texts)

def _embed_dim_probe() -> int:
    for attempt in range(5):
        try:
            r = requests.post(
                f"{OLLAMA_URL}/api/embeddings",
                json={"model": EMBED_MODEL, "prompt": "dimension probe"},
                timeout=300
            )
            r.raise_for_status()
            resp = r.json()
            emb = resp.get('embedding', None)
            if not emb:
                raise RuntimeError("Embeddings response missing 'embedding'.")
            return len(emb)
        except requests.RequestException as e:
            if attempt == 4:
                raise
            log.warning(f"Dimension probe failed (attempt {attempt + 1}/5): {e}. Retrying in {2 ** attempt} seconds...")
            time.sleep(2 ** attempt)

def _file_md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()

def parse_document(path: str) -> str:
    reader = PdfReader(path)
    md = [f'# Page {i+1}\n\n{page.extract_text() or ""}'.strip() for i, page in enumerate(reader.pages) if page.extract_text()]
    return '\n\n'.join(md) if md else "No text extracted"

def load_documents_from_folder(folder: str) -> List[Document]:
    from rag.utils.chunking import split_markdown_into_chunks
    docs = []
    for name in sorted(os.listdir(folder)):
        if not name.lower().endswith(('.pdf', '.txt', '.md')):
            continue
        path = os.path.join(folder, name)
        log.info(f"Processing document: {name}")
        md = parse_document(path)
        if not md or md == "No text extracted":
            log.warning(f"No usable text extracted from {name}, skipping chunking.")
            continue
        chunks = split_markdown_into_chunks(md, target_chars=800, overlap=80)
        file_id = _file_md5(path)
        for idx, ch in enumerate(chunks):
            docs.append(Document(text=ch, metadata={
                'uri': f'file://{name}',
                'file_id': file_id,
                'chunk_index': idx,
                'filename': name,
            }))
    return docs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', required=True, help='Folder with PDFs/MD/TXT to ingest')
    args = parser.parse_args()

    folder = pathlib.Path(args.path)
    if not folder.exists():
        log.error('Folder not found: %s', folder)
        sys.exit(1)

    log.info("Probing embedding dimension from Ollama model %r...", EMBED_MODEL)
    embed_dim = _embed_dim_probe()
    log.info("Embedding dimension: %d", embed_dim)

    Settings.embed_model = SimpleOllamaEmbedding(EMBED_MODEL, OLLAMA_URL)

    sync_connection_string = DATABASE_URL
    async_connection_string = sync_connection_string.replace("postgresql://", "postgresql+asyncpg://", 1)
    store = PGVectorStore(
        connection_string=sync_connection_string,
        async_connection_string=async_connection_string,
        schema_name='public',
        table_name='llamaindex_chunks',
        embed_dim=embed_dim,
    )
    storage_context = StorageContext.from_defaults(vector_store=store)

    docs = load_documents_from_folder(str(folder))
    if not docs:
        log.warning('No supported files found or no usable text extracted from %s', folder)
        return

    log.info('Upserting %d chunks to PGVector...', len(docs))
    batch_size = 100
    for i in tqdm(range(0, len(docs), batch_size), desc="Upserting chunks"):
        batch = docs[i:i + batch_size]
        try:
            VectorStoreIndex.from_documents(batch, storage_context=storage_context)
        except Exception as e:
            log.error(f"Failed to build index for batch {i//batch_size}: {e}")
            raise
    log.info('Ingestion complete.')

if __name__ == '__main__':
    main()