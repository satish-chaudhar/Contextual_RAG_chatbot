# Enable annotations for type hinting
from __future__ import annotations

# Import standard libraries for file handling, system operations, and logging
import os
import sys
import hashlib
import argparse
import pathlib
import logging
import requests
import time

# Import List type for type hinting
from typing import List

# Import PyPDF2 for PDF text extraction
from PyPDF2 import PdfReader

# Import ThreadPoolExecutor for parallel embedding and tqdm for progress bars
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Import LlamaIndex modules with version tolerance
try:
    from llama_index.core import Document, StorageContext, Settings, VectorStoreIndex
    from llama_index.core.node_parser import SemanticSplitterNodeParser, SentenceSplitter
except Exception:
    from llama_index import Document, StorageContext, Settings, VectorStoreIndex
    from llama_index.node_parser import SemanticSplitterNodeParser, SentenceSplitter

# Import PGVectorStore for vector storage
try:
    from llama_index.vector_stores.postgres import PGVectorStore
except Exception as e:
    # Raise error if PGVectorStore is missing
    raise RuntimeError('Missing llama-index-vector-stores-postgres. Install it in the container.') from e

# Configure logging with timestamp and level
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
log = logging.getLogger('ingest')

# Define environment variables for database and Ollama
DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://postgres:admin@postgres:5432/postgres')
OLLAMA_URL = os.getenv('OLLAMA_URL', 'http://ollama:11434')
EMBED_MODEL = os.getenv('EMBED_MODEL', 'nomic-embed-text')

# Define embedding class for Ollama integration
class SimpleOllamaEmbedding:
    # Initialize with model name and Ollama URL
    def __init__(self, model: str, url: str):
        self.model = model
        self.url = url.rstrip("/")  # Remove trailing slash from URL

    # Embed a single text with retry logic
    def _embed(self, text: str, max_retries: int = 5):
        # Attempt embedding with retries
        for attempt in range(max_retries):
            try:
                # Send POST request to Ollama's embedding API
                r = requests.post(
                    f"{self.url}/api/embeddings",
                    json={"model": self.model, "prompt": text},
                    timeout=300
                )
                r.raise_for_status()  # Raise exception for bad status codes
                data = r.json()
                embedding = data.get("embedding", [])  # Extract embedding
                if not embedding:
                    raise ValueError("No embedding returned from Ollama")
                return embedding
            except requests.RequestException as e:
                # If last attempt, raise the exception
                if attempt == max_retries - 1:
                    raise
                # Log warning and wait before retrying
                log.warning(f"Embedding failed (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {2 ** attempt} seconds...")
                time.sleep(2 ** attempt)

    # Embed a batch of texts using ThreadPoolExecutor
    def _embed_batch(self, texts: List[str], max_retries: int = 5):
        # Create thread pool for parallel embedding
        with ThreadPoolExecutor() as executor:
            # Map texts to futures
            future_to_text = {executor.submit(self._embed, text, max_retries): text for text in texts}
            # Collect results as they complete
            return [future.result() for future in as_completed(future_to_text)]

    # Get embedding for a single text
    def get_text_embedding(self, text: str):
        return self._embed(text)

    # Get embedding for a query (same as text)
    def get_query_embedding(self, query: str):
        return self._embed(query)

    # Get embeddings for a batch of texts
    def get_text_embedding_batch(self, texts: List[str], show_progress: bool = False):
        return self._embed_batch(texts)

# Probe embedding dimension from Ollama
def _embed_dim_probe() -> int:
    # Attempt probing dimension with retries
    for attempt in range(5):
        try:
            # Send test embedding request
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
            return len(emb)  # Return dimension of embedding
        except requests.RequestException as e:
            # If last attempt, raise exception
            if attempt == 4:
                raise
            # Log warning and wait before retrying
            log.warning(f"Dimension probe failed (attempt {attempt + 1}/5): {e}. Retrying in {2 ** attempt} seconds...")
            time.sleep(2 ** attempt)

# Calculate MD5 hash of a file
def _file_md5(path: str) -> str:
    h = hashlib.md5()  # Initialize MD5 hash object
    with open(path, 'rb') as f:
        # Read file in chunks and update hash
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()  # Return hexadecimal hash

# Parse document to extract text (PDFs)
def parse_document(path: str) -> str:
    reader = PdfReader(path)  # Initialize PDF reader
    # Extract text from each page, adding page headers
    md = [f'# Page {i+1}\n\n{page.extract_text() or ""}'.strip() for i, page in enumerate(reader.pages) if page.extract_text()]
    return '\n\n'.join(md) if md else "No text extracted"  # Join pages or return error message

# Load documents from a folder
def load_documents_from_folder(folder: str) -> List[Document]:
    docs = []  # Initialize list for documents
    # Iterate over files in folder
    for name in sorted(os.listdir(folder)):
        # Process only supported file types
        if not name.lower().endswith(('.pdf', '.txt', '.md')):
            continue
        path = os.path.join(folder, name)  # Construct full path
        log.info(f"Processing document: {name}")  # Log processing
        md = parse_document(path)  # Parse document text
        if not md or md == "No text extracted":
            log.warning(f"No usable text extracted from {name}, skipping.")  # Log warning for empty text
            continue
        file_id = _file_md5(path)  # Calculate file hash
        # Create Document with text and metadata
        docs.append(Document(text=md, metadata={
            'uri': f'file://{name}',
            'file_id': file_id,
            'filename': name,
        }))
    return docs  # Return list of documents

# Main function for ingestion
def main():
    # Set up argument parser for folder path
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', required=True, help='Folder with PDFs/MD/TXT to ingest')
    args = parser.parse_args()

    folder = pathlib.Path(args.path)  # Convert path to Path object
    if not folder.exists():
        log.error('Folder not found: %s', folder)  # Log error if folder doesn't exist
        sys.exit(1)  # Exit with error code

    log.info("Probing embedding dimension from Ollama model %r...", EMBED_MODEL)  # Log dimension probing
    embed_dim = _embed_dim_probe()  # Get embedding dimension
    log.info("Embedding dimension: %d", embed_dim)  # Log dimension

    embed_model = SimpleOllamaEmbedding(EMBED_MODEL, OLLAMA_URL)  # Initialize embedding model
    Settings.embed_model = embed_model  # Set global embed model

    # Initialize semantic splitter with embedding-based breakpoints
    semantic_parser = SemanticSplitterNodeParser(
        buffer_size=1,  # Number of sentences to buffer
        breakpoint_percentile_threshold=95,  # Threshold for semantic breaks
        embed_model=embed_model,  # Embedding model for semantic splitting
        sentence_splitter=SentenceSplitter(chunk_size=800, chunk_overlap=80)  # Inner recursive splitter
    )

    # Configure database connections
    sync_connection_string = DATABASE_URL
    async_connection_string = sync_connection_string.replace("postgresql://", "postgresql+asyncpg://", 1)
    store = PGVectorStore(
        connection_string=sync_connection_string,  # Sync connection string
        async_connection_string=async_connection_string,  # Async connection string
        schema_name='public',  # Database schema
        table_name='llamaindex_chunks',  # Table for storing chunks
        embed_dim=embed_dim,  # Embedding dimension
    )
    storage_context = StorageContext.from_defaults(vector_store=store)  # Create storage context

    docs = load_documents_from_folder(str(folder))  # Load documents
    if not docs:
        log.warning('No supported files found or no usable text extracted from %s', folder)  # Log warning if no docs
        return

    log.info('Parsing %d documents into semantic chunks...', len(docs))  # Log parsing
    nodes = semantic_parser.get_nodes_from_documents(docs)  # Parse documents into semantic nodes

    # Add chunk_index metadata to nodes
    for idx, node in enumerate(nodes):
        if 'chunk_index' not in node.metadata:
            node.metadata['chunk_index'] = idx

    log.info('Upserting %d semantic chunks to PGVector...', len(nodes))  # Log upserting
    batch_size = 100  # Set batch size for indexing
    # Process nodes in batches
    for i in tqdm(range(0, len(nodes), batch_size), desc="Upserting chunks"):
        batch = nodes[i:i + batch_size]  # Get batch of nodes
        try:
            VectorStoreIndex(nodes=batch, storage_context=storage_context)  # Index batch
        except Exception as e:
            log.error(f"Failed to build index for batch {i//batch_size}: {e}")  # Log error
            raise
    log.info('Ingestion complete.')  # Log completion

# Run main function if script is executed directly
if __name__ == '__main__':
    main()