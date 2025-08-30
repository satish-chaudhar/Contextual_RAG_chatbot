from __future__ import annotations
import os  # For file system operations and environment variables
import sys  # For system-specific functions like exiting the program
import hashlib  # For computing MD5 hashes of files
import argparse  # For parsing command-line arguments
import pathlib  # For modern file path handling
import logging  # For logging messages to track execution
import requests  # For making HTTP requests to the Ollama API
import time  # For implementing delays in retry logic
from typing import List
from PyPDF2 import PdfReader

try:
    from llama_index.core import Document, StorageContext, Settings, VectorStoreIndex
except Exception:
    from llama_index import Document, StorageContext, Settings, VectorStoreIndex

try:
    from llama_index.vector_stores.postgres import PGVectorStore
except Exception as e:
    raise RuntimeError('Missing llama-index-vector-stores-postgres. Install it in the container.') from e

# Define a class to interface with the Ollama API for generating embeddings
class SimpleOllamaEmbedding:
    # Initialize with model name and API URL
    def __init__(self, model: str, url: str):
        self.model = model  # Store the embedding model name
        self.url = url  # Store the Ollama API endpoint

    # Private method to generate embedding for a single text with retry logic
    def _embed(self, text: str, max_retries: int = 5):
        # Attempt embedding up to max_retries times
        for attempt in range(max_retries):
            try:
                # Send POST request to Ollama's embeddings endpoint
                r = requests.post(
                    f"{self.url}/api/embeddings",
                    json={"model": self.model, "prompt": text},
                    timeout=300  # Set timeout to 300 seconds
                )
                r.raise_for_status()  # Raise an exception for HTTP errors
                data = r.json()  # Parse the JSON response
                embedding = data.get("embedding", [])  # Get embedding, default to empty list
                # Check if embedding is empty
                if not embedding:
                    raise ValueError("No embedding returned from Ollama")
                return embedding  # Return the embedding list
            # Handle request-related exceptions
            except requests.RequestException as e:
                # If last attempt, re-raise the exception
                if attempt == max_retries - 1:
                    raise
                # Log warning with attempt number and error
                log.warning(f"Embedding failed (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {2 ** attempt} seconds...")
                time.sleep(2 ** attempt)  # Exponential backoff delay

    # Private method to generate embeddings for a list of texts
    def _embed_batch(self, texts: List[str], max_retries: int = 5):
        embeddings = []  # Initialize list to store embeddings
        # Process each text in the list
        for text in texts:
            embeddings.append(self._embed(text, max_retries))  # Generate and append embedding
        return embeddings  # Return list of embeddings

    # Public method to get embedding for a single text (required by llama_index)
    def get_text_embedding(self, text: str):
        return self._embed(text)  # Delegate to _embed

    # Public method to get embedding for a query (required by llama_index)
    def get_query_embedding(self, query: str):
        return self._embed(query)  # Delegate to _embed

    # Public method to get embeddings for a list of texts
    def get_text_embedding_batch(self, texts: List[str], show_progress: bool = False):
        return self._embed_batch(texts)  # Delegate to _embed_batch

# Configure logging with a specific format and INFO level
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
log = logging.getLogger('ingest')  # Create a logger named 'ingest'

# Get PostgreSQL connection string from environment or use default
DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://postgres:admin@postgres:5432/postgres')
# Get Ollama API endpoint from environment or use default
OLLAMA_URL = os.getenv('OLLAMA_URL', 'http://ollama:11434')
# Get embedding model name from environment or use default
EMBED_MODEL = os.getenv('EMBED_MODEL', 'nomic-embed-text')

# Determine the dimensionality of the Ollama model's embeddings
def _embed_dim_probe() -> int:
    # Attempt up to 5 times to get embedding dimension
    for attempt in range(5):
        try:
            # Send POST request with a dummy prompt to get an embedding
            r = requests.post(
                f"{OLLAMA_URL}/api/embeddings",
                json={"model": EMBED_MODEL, "prompt": "dimension probe"},
                timeout=300  # Set timeout to 300 seconds
            )
            r.raise_for_status()  # Raise an exception for HTTP errors
            resp = r.json()  # Parse the JSON response
            emb = resp.get('embedding', None)  # Get embedding, default to None
            # Check if embedding is missing
            if not emb:
                raise RuntimeError("Embeddings response missing 'embedding'. Check EMBED_MODEL and Ollama.")
            return len(emb)  # Return length of embedding (dimension)
        # Handle request-related exceptions
        except requests.RequestException as e:
            # If last attempt, re-raise the exception
            if attempt == 4:
                raise
            # Log warning with attempt number and error
            log.warning(f"Dimension probe failed (attempt {attempt + 1}/5): {e}. Retrying in {2 ** attempt} seconds...")
            time.sleep(2 ** attempt)  # Exponential backoff delay

# Compute MD5 hash of a file
def _file_md5(path: str) -> str:
    h = hashlib.md5()  # Create an MD5 hash object
    # Open file in binary read mode
    with open(path, 'rb') as f:
        # Read file in chunks of 8192 bytes
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)  # Update hash with each chunk
    return h.hexdigest()  # Return hexadecimal MD5 hash

# Extract text from a file using docling, with PyPDF2 as fallback
def parse_with_docling(path: str, max_retries: int = 5) -> str:
    # Attempt up to max_retries times to parse with docling
    for attempt in range(max_retries):
        try:
            # Import and instantiate DocumentConverter from docling
            from docling.document_converter import DocumentConverter
            converter = DocumentConverter()
            result = converter.convert(path)  # Convert file to text
            # Check if result has a document with export_to_markdown method
            if hasattr(result, 'document') and hasattr(result.document, 'export_to_markdown'):
                return result.document.export_to_markdown()  # Return Markdown text
            # Check if result has a text attribute that is a string
            if hasattr(result, 'text') and isinstance(result.text, str):
                return result.text  # Return plain text
        # Handle exceptions during docling parsing
        except Exception as e:
            # If last attempt, log warning and break to fallback
            if attempt == max_retries - 1:
                log.warning(f"Docling failed after {max_retries} attempts for {path}: {e}. Falling back to PyPDF2.")
                break
            # Log warning and retry with exponential backoff
            log.warning(f"Docling failed (attempt {attempt + 1}/{max_retries}) for {path}: {e}. Retrying in {2 ** attempt} seconds...")
            time.sleep(2 ** attempt)
    # Fallback to PyPDF2 for text extraction
    reader = PdfReader(path)  # Create PdfReader for the file
    md = []  # Initialize list to store Markdown-formatted text
    # Iterate over each page in the PDF
    for i, page in enumerate(reader.pages):
        txt = page.extract_text() or ''  # Extract text, default to empty string
        # If text is non-empty after stripping, format as Markdown
        if txt.strip():
            md.append(f'# Page {i+1}\n\n{txt.strip()}')
    # Join Markdown sections with double newlines or return error message
    return '\n\n'.join(md) if md else "No text extracted"

# Load documents from a folder and create Document objects
def load_documents_from_folder(folder: str) -> List[Document]:
    # Import chunking function from custom module
    from rag.utils.chunking import split_markdown_into_chunks
    docs: List[Document] = []  # Initialize list to store Document objects
    # Iterate over files in folder, sorted alphabetically
    for name in sorted(os.listdir(folder)):
        # Skip files that don't end with .pdf, .txt, or .md
        if not name.lower().endswith(('.pdf', '.txt', '.md')):
            continue
        path = os.path.join(folder, name)  # Construct full file path
        log.info(f"Processing document: {name}")  # Log file being processed
        md = parse_with_docling(path)  # Extract text from file
        # Check if text extraction failed
        if not md or md == "No text extracted":
            log.warning(f"No usable text extracted from {name}, skipping chunking.")
            continue
        # Split text into chunks with target length and overlap
        chunks = split_markdown_into_chunks(md, target_chars=1200, overlap=120)
        file_id = _file_md5(path)  # Compute MD5 hash of file
        # Create Document object for each chunk
        for idx, ch in enumerate(chunks):
            docs.append(Document(text=ch, metadata={
                'uri': f'file://{name}',  # File URI
                'file_id': file_id,  # MD5 hash as unique identifier
                'chunk_index': idx,  # Chunk index
                'filename': name,  # File name
            }))
    return docs  # Return list of Document objects

# Main function to orchestrate document ingestion
def main():
    # Create argument parser for command-line arguments
    parser = argparse.ArgumentParser()
    # Add required --path argument for folder path
    parser.add_argument('--path', required=True, help='Folder with PDFs/MD/TXT to ingest')
    args = parser.parse_args()  # Parse arguments

    folder = pathlib.Path(args.path)  # Convert folder path to Path object
    # Check if folder exists
    if not folder.exists():
        log.error('Folder not found: %s', folder)  # Log error
        sys.exit(1)  # Exit with error code

    # Log that embedding dimension probing is starting
    log.info("Probing embedding dimension from Ollama model %r...", EMBED_MODEL)
    embed_dim = _embed_dim_probe()  # Get embedding dimension
    log.info("Embedding dimension: %d", embed_dim)  # Log the dimension

    # Set custom embedding model for llama_index
    Settings.embed_model = SimpleOllamaEmbedding(EMBED_MODEL, OLLAMA_URL)

    # Use DATABASE_URL for synchronous connection
    sync_connection_string = DATABASE_URL
    # Create asynchronous connection string by replacing protocol
    async_connection_string = sync_connection_string.replace("postgresql://", "postgresql+asyncpg://", 1)
    # Create PostgreSQL vector store
    store = PGVectorStore(
        connection_string=sync_connection_string,  # Synchronous connection
        async_connection_string=async_connection_string,  # Asynchronous connection
        schema_name='public',  # Database schema
        table_name='llamaindex_chunks',  # Table name for storing chunks
        embed_dim=embed_dim,  # Embedding dimension
    )
    # Create storage context with the vector store
    storage_context = StorageContext.from_defaults(vector_store=store)

    # Load documents from the specified folder
    docs = load_documents_from_folder(str(folder))
    # Check if any documents were loaded
    if not docs:
        log.warning('No supported files found or no usable text extracted from %s', folder)
        return  # Exit if no documents

    # Log the number of chunks to be upserted
    log.info('Upserting %d chunks to PGVector...', len(docs))
    try:
        # Create vector index from documents and store in PostgreSQL
        VectorStoreIndex.from_documents(docs, storage_context=storage_context)
    except Exception as e:
        log.error(f"Failed to build index: {e}")  # Log error if indexing fails
        raise  # Re-raise the exception
    log.info('Ingestion complete.')  # Log successful completion

# Check if script is run directly and call main
if __name__ == '__main__':
    main()