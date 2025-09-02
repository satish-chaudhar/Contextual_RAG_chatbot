# Enable annotations for better type hinting support
from __future__ import annotations

# Import standard libraries for system, networking, timing, and logging
import os, requests, time, asyncio, logging

# Import typing utilities
from typing import List, Tuple, Dict, Any, Optional

# Import contextlib for null context in tracing
from contextlib import nullcontext

# Import OpenTelemetry for tracing
from opentelemetry import trace as _trace

# Initialize tracer for RAG operations
tracer = _trace.get_tracer("rag")
_get_current_span = _trace.get_current_span  # Get current span for tracing

# Import LlamaIndex modules with version tolerance
try:
    from llama_index.core import VectorStoreIndex, StorageContext, Settings
    from llama_index.core.indices.postprocessor import SimilarityPostprocessor
    from llama_index.core.schema import NodeWithScore
except Exception:
    from llama_index import VectorStoreIndex, StorageContext, Settings
    try:
        from llama_index.indices.postprocessor import SimilarityPostprocessor
    except Exception:
        SimilarityPostprocessor = None  # Disable postprocessor if unavailable
    try:
        from llama_index.schema import NodeWithScore
    except Exception:
        # Define minimal shim for NodeWithScore
        class NodeWithScore:
            node: Any
            score: float

# Import PGVectorStore for vector storage
try:
    from llama_index.vector_stores.postgres import PGVectorStore
except Exception as e:
    raise RuntimeError('Missing llama-index-vector-stores-postgres. Install it in the container.') from e

# Initialize logger for RAG operations
log = logging.getLogger('rag')
log.setLevel(logging.INFO)  # Set logging level to INFO

# Import Crew.AI prompt optimization (with fallback)
try:
    from .agents import optimize_prompt
except Exception as e:
    optimize_prompt = None  # Fallback if import fails
    log.warning("Crew.AI prompt optimization not available: %s", e)

# Define configuration variables from environment
DATABASE_URL = os.getenv("DATABASE_URL", 'postgresql://postgres:admin@postgres:5432/postgres')  # Database connection
OLLAMA_URL = os.getenv("OLLAMA_URL", 'http://ollama:11434')  # Ollama API URL
EMBED_MODEL = os.getenv("EMBED_MODEL", 'nomic-embed-text')  # Embedding model
LLM_MODEL = os.getenv("LLM_MODEL", 'llama3.2:3b')  # LLM model

TOP_K = int(os.getenv("TOP_K", "24"))  # Number of top documents to retrieve
TOP_N = int(os.getenv("TOP_N", "8"))  # Number of documents after reranking
MAX_CONTEXT = int(os.getenv("MAX_CONTEXT_CHARS", "12000"))  # Max context length
SIM_CUTOFF = float(os.getenv("SIMILARITY_CUTOFF", "0.1"))  # Similarity cutoff for postprocessing
RERANK_STRATEGY = (os.getenv("RERANK_STRATEGY", "mmr") or "mmr").lower()  # Reranking strategy

# Define system prompt for LLM
SYSTEM_PROMPT = (
    "You are a careful, concise assistant. Answer ONLY from the provided context.\n"
    "- If context is insufficient, say you don't have enough information.\n"
    "- Cite sources inline like [filename #chunk]. Keep answers short."
)

# Define embedding class for Ollama
class SimpleOllamaEmbedding:
    # Initialize with model, URL, and other parameters
    def __init__(self, model: str, url: str, timeout: int = 300, batch_size: int = 16, sleep_s: float = 0.0):
        self.model = model
        self.url = url.rstrip("/")  # Remove trailing slash
        self.timeout = timeout  # Request timeout
        self.batch_size = batch_size  # Batch size for embedding
        self.sleep_s = sleep_s  # Sleep time between requests

    # Embed a single text
    def _embed_one(self, text: str) -> List[float]:
        # Send POST request to Ollama embedding API
        r = requests.post(
            f"{self.url}/api/embeddings",
            json={"model": self.model, "prompt": text},
            timeout=self.timeout,
        )
        r.raise_for_status()  # Raise exception for bad status
        data = r.json()
        # Handle different response formats
        if isinstance(data, dict) and "embedding" in data:
            return data["embedding"]
        if isinstance(data, dict) and "data" in data and data["data"]:
            return data["data"][0].get("embedding", [])
        raise RuntimeError("Embeddings response missing 'embedding'.")

    # Get embedding for a single text
    def get_text_embedding(self, text: str) -> List[float]:
        return self._embed_one(text)

    # Get embedding for a query
    def get_query_embedding(self, query: str) -> List[float]:
        return self._embed_one(query)

    # Get embeddings for a batch of texts
    def get_text_embedding_batch(self, texts: List[str], show_progress: bool = False, **kwargs) -> List[List[float]]:
        out: List[List[float]] = []  # Initialize output list
        # Process texts in batches
        for i in range(0, len(texts), self.batch_size):
            for t in texts[i : i + self.batch_size]:
                out.append(self._embed_one(t))  # Embed each text
                if self.sleep_s:
                    time.sleep(self.sleep_s)  # Sleep if specified
        return out

    # Get embeddings for a batch of queries
    def get_query_embedding_batch(self, queries: List[str], show_progress: bool = False, **kwargs) -> List[List[float]]:
        return self.get_text_embedding_batch(queries, show_progress=show_progress, **kwargs)

    # Aggregate embeddings from multiple queries
    def get_agg_embedding_from_queries(self, queries: List[str], **kwargs) -> List[float]:
        from numpy import mean, array  # Import numpy for mean calculation
        embeddings = self.get_query_embedding_batch(queries)  # Get batch embeddings
        if not embeddings:
            raise ValueError("No embeddings generated for queries")
        aggregated = mean(array(embeddings), axis=0).tolist()  # Compute mean embedding
        return aggregated

# Convert PostgreSQL DSN to SQLAlchemy sync format
def _to_sqla_sync(dsn: str) -> str:
    if dsn.startswith("postgresql+psycopg2://"):
        return dsn
    return dsn.replace("postgresql://", "postgresql+psycopg2://", 1)

# Convert PostgreSQL DSN to SQLAlchemy async format
def _to_sqla_async(dsn: str) -> str:
    if dsn.startswith("postgresql+asyncpg://"):
        return dsn
    return dsn.replace("postgresql://", "postgresql+asyncpg://", 1)

# Probe embedding dimension from Ollama
def _embed_dim() -> int:
    # Send test embedding request
    r = requests.post(
        f"{OLLAMA_URL.rstrip('/')}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": "dimension probe"},
        timeout=300,
    )
    r.raise_for_status()
    data = r.json()
    # Handle different response formats
    if isinstance(data, dict) and "embedding" in data:
        return len(data["embedding"])
    if isinstance(data, dict) and "data" in data and data["data"]:
        return len(data["data"][0].get("embedding", []))
    raise RuntimeError("Couldn't determine embedding dimension from Ollama.")

# Initialize global index and retriever
_index: Optional[VectorStoreIndex] = None
_retriever = None

# Ensure index is initialized
def _ensure_index() -> VectorStoreIndex:
    global _index, _retriever
    # Start tracing span for index initialization
    ctx = tracer.start_as_current_span("ensure_index", attributes={"db": DATABASE_URL, "embed_model": EMBED_MODEL}, kind=_trace.SpanKind.INTERNAL) if tracer else nullcontext()
    with ctx:
        if _index is None:
            # Set embedding model
            Settings.embed_model = SimpleOllamaEmbedding(EMBED_MODEL, OLLAMA_URL)
            # Initialize PGVectorStore
            store = PGVectorStore(
                connection_string=_to_sqla_sync(DATABASE_URL),
                async_connection_string=_to_sqla_async(DATABASE_URL),
                schema_name="public",
                table_name="llamaindex_chunks",
                embed_dim=_embed_dim(),
            )
            sc = StorageContext.from_defaults(vector_store=store)  # Create storage context
            _index = VectorStoreIndex.from_vector_store(vector_store=store, storage_context=sc)  # Initialize index
            _retriever = _index.as_retriever(similarity_top_k=max(TOP_K, TOP_N))  # Initialize retriever
    return _index

# Retrieve relevant nodes for a query
def _retrieve(query: str) -> List[NodeWithScore]:
    # Start tracing span for retrieval
    ctx = tracer.start_as_current_span("retrieve", attributes={"input.query": query, "top_k": TOP_K}, kind=_trace.SpanKind.INTERNAL) if tracer else nullcontext()
    with ctx:
        _ensure_index()  # Ensure index is initialized
        nodes: List[NodeWithScore] = _retriever.retrieve(query)  # Retrieve nodes
        # Apply similarity postprocessing if available
        if SimilarityPostprocessor is not None and SIM_CUTOFF > 0:
            nodes = SimilarityPostprocessor(similarity_cutoff=SIM_CUTOFF).postprocess_nodes(nodes)
        # Log tracing attributes
        if tracer:
            try:
                span = _get_current_span()
                span.set_attribute("output.retrieved_nodes_count", len(nodes))
                span.add_event("output", {"value": {"count": len(nodes), "sample": nodes[0].text[:100] if nodes else "None"}})
            except Exception as e:
                log.warning("Failed to set retrieve attributes: %s", e)
        return nodes

# Import reranking utilities with fallback
try:
    from rag.rerank import Scored, mmr_rerank, cross_encoder_rerank
except Exception:
    from dataclasses import dataclass

    # Define Scored dataclass for fallback
    @dataclass
    class Scored:
        text: str
        score: float
        meta: Dict[str, Any]

    # Fallback MMR reranking
    def mmr_rerank(candidates: List[Scored], top_n: int = 8) -> List[Scored]:
        return sorted(candidates, key=lambda x: x.score, reverse=True)[:top_n]

    # Fallback cross-encoder reranking
    def cross_encoder_rerank(query: str, texts: List[Scored], top_n: int = 8) -> List[Scored]:
        return mmr_rerank(texts, top_n=top_n)

# Post-process retrieved nodes
def _post_process(nodes: List[NodeWithScore], query: str) -> List[Scored]:
    # Start tracing span for post-processing
    ctx = tracer.start_as_current_span("post_process", attributes={"input.nodes_count": len(nodes), "rerank_strategy": RERANK_STRATEGY}, kind=_trace.SpanKind.INTERNAL) if tracer else nullcontext()
    with ctx:
        scored: List[Scored] = []  # Initialize list for scored nodes
        for n in nodes:
            node = getattr(n, "node", n)  # Get node object
            score = float(getattr(n, "score", 0.0) or 0.0)  # Get score
            content = ""  # Initialize content
            # Extract content from node
            if hasattr(node, "get_content"):
                content = node.get_content()
            elif hasattr(node, "text"):
                content = node.text
            meta = getattr(node, "metadata", {}) or {}  # Get metadata
            scored.append(Scored(text=content, score=score, meta=meta))  # Add to scored list

        # Apply reranking based on strategy
        if RERANK_STRATEGY == "ce":
            return cross_encoder_rerank(query, scored, top_n=TOP_N)
        if RERANK_STRATEGY == "mmr":
            return mmr_rerank(scored, top_n=TOP_N)
        return sorted(scored, key=lambda x: x.score, reverse=True)[:TOP_N]  # Default sorting
        # Log tracing attributes (this line is unreachable due to returns above, kept for original code fidelity)
        if tracer:
            try:
                span = _get_current_span()
                span.set_attribute("output.scored_count", len(scored))
                span.add_event("output", {"value": {"count": len(scored), "sample_score": scored[0].score if scored else 0}})
            except Exception as e:
                log.warning("Failed to set post_process attributes: %s", e)

# Build context from scored items
def _build_context(items: List[Scored]) -> Tuple[str, List[Dict[str, Any]]]:
    # Start tracing span for context building
    ctx = tracer.start_as_current_span("build_context", attributes={"input.items_count": len(items)}, kind=_trace.SpanKind.INTERNAL) if tracer else nullcontext()
    with ctx:
        parts: List[str] = []  # Initialize list for context parts
        sources: List[Dict[str, Any]] = []  # Initialize list for source metadata
        total = 0  # Track total length
        for it in items:
            meta = it.meta or {}  # Get metadata
            # Create source tag with filename and chunk index
            tag = f"{meta.get('filename', meta.get('uri', ''))} #{meta.get('chunk_index', '?')}"
            block = f"[Source: {tag}]\n{it.text}\n\n"  # Format context block
            if total + len(block) > MAX_CONTEXT:
                break  # Stop if context exceeds max length
            parts.append(block)  # Add block to parts
            total += len(block)  # Update total length
            # Add source metadata
            sources.append(
                {"filename": meta.get("filename"), "chunk": meta.get("chunk_index"), "uri": meta.get("uri")}
            )
        context_text = "".join(parts)  # Join parts into context
        # Log tracing attributes
        if tracer:
            try:
                span = _get_current_span()
                span.set_attribute("output.context_length_chars", total)
                span.add_event("output", {"value": {"text_snippet": context_text[:500] + "..." if context_text else "None"}})
            except Exception as e:
                log.warning("Failed to set build_context attributes: %s", e)
        return context_text, sources  # Return context and sources

# Perform chat with LLM using context
def _chat(query: str, context_md: str) -> str:
    # Start tracing span for LLM chat
    ctx = tracer.start_as_current_span("llama_chat", attributes={"input.llm_model": LLM_MODEL}, kind=_trace.SpanKind.INTERNAL) if tracer else nullcontext()
    with ctx:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT + "\n\nContext:\n" + context_md},
            {"role": "user", "content": query},
        ]
        # Create full prompt for tracing
        prompt = SYSTEM_PROMPT + "\n\nContext:\n" + context_md + "\n\n" + query
        # Start tracing span for Ollama call
        call_ctx = tracer.start_as_current_span("call_ollama", attributes={"input.url": OLLAMA_URL, "input.prompt_length": len(prompt.split())}, kind=_trace.SpanKind.CLIENT) if tracer else nullcontext()
        with call_ctx:
            session = requests.Session()  # Create HTTP session
            adapter = requests.adapters.HTTPAdapter(max_retries=0)  # Disable retries
            session.mount("http://", adapter)  # Mount adapter
            # Send chat request to Ollama
            r = session.post(
                f"{OLLAMA_URL.rstrip('/')}/api/chat",
                json={"model": LLM_MODEL, "messages": messages, "stream": False},
                timeout=600,
            )
            try:
                r.raise_for_status()  # Raise exception for bad status
            finally:
                # Log tracing attributes
                if tracer:
                    try:
                        span = _get_current_span()
                        span.set_attribute("output.http_status_code", r.status_code)
                        response = r.json().get("message", {}).get("content", "No response")
                        span.add_event("output", {"value": {"response": response[:500] + "..." if len(response) > 500 else response}})
                        span.set_attribute("output.response_token_estimate", len(response.split()))
                    except Exception as e:
                        log.warning("Failed to set call_ollama attributes: %s", e)
        data = r.json()
        # Handle different response formats
        if isinstance(data, dict) and "message" in data:
            return (data["message"] or {}).get("content", "").strip()
        if isinstance(data, dict) and "choices" in data and data["choices"]:
            return (data["choices"][0]["message"] or {}).get("content", "").strip()
        return str(data)  # Fallback to string representation

# Main answer function for RAG pipeline
def answer(query: str) -> str:
    if not query or not query.strip():
        return "Please provide a question."  # Handle empty query
    try:
        # Start tracing span for answer
        parent_span = _get_current_span()
        ctx = tracer.start_as_current_span("rag_answer", attributes={"input.query": query}, kind=_trace.SpanKind.INTERNAL) if tracer and not parent_span.is_recording() else nullcontext()
        with ctx:
            nodes = _retrieve(query)  # Retrieve relevant nodes
            items = _post_process(nodes, query)  # Post-process nodes
            ctx_text, _ = _build_context(items)  # Build context
            if not ctx_text.strip():
                return "I don't have enough information in the knowledge base to answer that confidently."  # Handle empty context
            response = _chat(query, ctx_text)  # Get LLM response
            # Log tracing attributes
            if tracer:
                try:
                    span = _get_current_span()
                    span.set_attribute("output.response", response[:500] + "..." if len(response) > 500 else response)
                    span.set_attribute("output.total_duration_ms", int(time.time() * 1000) - span.start_time)
                except Exception as e:
                    log.warning("Failed to set answer attributes: %s", e)
            return response
    except requests.exceptions.ConnectionError:
        # Handle Ollama connection error
        if tracer:
            try:
                _get_current_span().add_event("error", {"message": "Ollama unreachable"})
            except Exception:
                pass
        return "The model/embedding service at OLLAMA_URL seems unreachable. Make sure Ollama is running and OLLAMA_URL is correct."
    except Exception as e:
        # Handle general errors
        if tracer:
            try:
                _get_current_span().add_event("error", {"message": str(e)})
            except Exception:
                pass
        return f"Error while answering: {e}"

# Define QARunnable class for integration
class QARunnable:
    # Synchronous invoke method
    def invoke(self, inputs: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        q = inputs.get("question") if isinstance(inputs, dict) else str(inputs)  # Extract question
        return {"answer": answer(str(q))}  # Return answer

    # Asynchronous invoke method
    async def ainvoke(self, inputs: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        loop = asyncio.get_event_loop()  # Get event loop
        q = inputs.get("question") if isinstance(inputs, dict) else str(inputs)  # Extract question
        res = await loop.run_in_executor(None, answer, str(q))  # Run answer synchronously
        return {"answer": res}

# Get QARunnable instance
def get_graph() -> QARunnable:
    try:
        _ensure_index()  # Ensure index is initialized
    except Exception:
        pass  # Ignore initialization errors
    return QARunnable()  # Return QARunnable instance

# Run answer function if script is executed directly
if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "What can you do?"  # Get query from command line
    print(answer(q))  # Print answer