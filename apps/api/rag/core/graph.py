# apps/api/rag/graph.py
from __future__ import annotations
import os, requests, time, asyncio, logging
from typing import List, Tuple, Dict, Any, Optional
from contextlib import nullcontext

from opentelemetry import trace as _trace

tracer = _trace.get_tracer("rag")
_get_current_span = _trace.get_current_span

# ---------- LlamaIndex imports (version tolerant) ----------
try:
    from llama_index.core import VectorStoreIndex, StorageContext, Settings
    from llama_index.core.indices.postprocessor import SimilarityPostprocessor
    from llama_index.core.schema import NodeWithScore
except Exception:
    from llama_index import VectorStoreIndex, StorageContext, Settings  # type: ignore
    try:
        from llama_index.indices.postprocessor import SimilarityPostprocessor  # type: ignore
    except Exception:
        SimilarityPostprocessor = None  # type: ignore
    try:
        from llama_index.schema import NodeWithScore  # type: ignore
    except Exception:
        class NodeWithScore:  # minimal shim
            node: Any
            score: float

# PGVector store (works on 0.9/0.10+)
try:
    from llama_index.vector_stores.postgres import PGVectorStore
except Exception as e:
    raise RuntimeError('Missing llama-index-vector-stores-postgres. Install it in the container.') from e

log = logging.getLogger('rag')
log.setLevel(logging.INFO)

# ---------- Crew.AI integration for prompt optimization ----------
try:
    from .agents import optimize_prompt  # Import your new agents.py
except Exception as e:
    optimize_prompt = None  # Fallback if not available
    log.warning("Crew.AI prompt optimization not available: %s", e)

# ---------- Configuration / defaults ----------
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:admin@postgres:5432/postgres")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2:3b")

TOP_K = int(os.getenv("TOP_K", "24"))
TOP_N = int(os.getenv("TOP_N", "8"))
MAX_CONTEXT = int(os.getenv("MAX_CONTEXT_CHARS", "12000"))
SIM_CUTOFF = float(os.getenv("SIMILARITY_CUTOFF", "0.1"))
RERANK_STRATEGY = (os.getenv("RERANK_STRATEGY", "mmr") or "mmr").lower()

SYSTEM_PROMPT = (
    "You are a careful, concise assistant. Answer ONLY from the provided context.\n"
    "- If context is insufficient, say you don't have enough information.\n"
    "- Cite sources inline like [filename #chunk]. Keep answers short."
)

# ---------- Simple embedding adapter (Ollama) ----------
class SimpleOllamaEmbedding:
    def __init__(self, model: str, url: str, timeout: int = 300, batch_size: int = 16, sleep_s: float = 0.0):
        self.model = model
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.batch_size = batch_size
        self.sleep_s = sleep_s

    def _embed_one(self, text: str) -> List[float]:
        r = requests.post(
            f"{self.url}/api/embeddings",
            json={"model": self.model, "prompt": text},
            timeout=self.timeout,
        )
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict) and "embedding" in data:
            return data["embedding"]
        if isinstance(data, dict) and "data" in data and data["data"]:
            return data["data"][0].get("embedding", [])
        raise RuntimeError("Embeddings response missing 'embedding'.")

    def get_text_embedding(self, text: str) -> List[float]:
        return self._embed_one(text)

    def get_query_embedding(self, query: str) -> List[float]:
        return self._embed_one(query)

    def get_text_embedding_batch(self, texts: List[str], show_progress: bool = False, **kwargs) -> List[List[float]]:
        out: List[List[float]] = []
        for i in range(0, len(texts), self.batch_size):
            for t in texts[i : i + self.batch_size]:
                out.append(self._embed_one(t))
                if self.sleep_s:
                    time.sleep(self.sleep_s)
        return out

    def get_query_embedding_batch(self, queries: List[str], show_progress: bool = False, **kwargs) -> List[List[float]]:
        return self.get_text_embedding_batch(queries, show_progress=show_progress, **kwargs)

    def get_agg_embedding_from_queries(self, queries: List[str], **kwargs) -> List[float]:
        from numpy import mean, array
        embeddings = self.get_query_embedding_batch(queries)
        if not embeddings:
            raise ValueError("No embeddings generated for queries")
        aggregated = mean(array(embeddings), axis=0).tolist()
        return aggregated

# ---------- Helpers ----------
def _to_sqla_sync(dsn: str) -> str:
    if dsn.startswith("postgresql+psycopg2://"):
        return dsn
    return dsn.replace("postgresql://", "postgresql+psycopg2://", 1)

def _to_sqla_async(dsn: str) -> str:
    if dsn.startswith("postgresql+asyncpg://"):
        return dsn
    return dsn.replace("postgresql://", "postgresql+asyncpg://", 1)

def _embed_dim() -> int:
    r = requests.post(
        f"{OLLAMA_URL.rstrip('/')}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": "dimension probe"},
        timeout=300,
    )
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict) and "embedding" in data:
        return len(data["embedding"])
    if isinstance(data, dict) and "data" in data and data["data"]:
        return len(data["data"][0].get("embedding", []))
    raise RuntimeError("Couldn't determine embedding dimension from Ollama.")

# ---------- Build (and cache) the index from existing PGVector table ----------
_index: Optional[VectorStoreIndex] = None
_retriever = None

def _ensure_index() -> VectorStoreIndex:
    global _index, _retriever
    ctx = tracer.start_as_current_span("ensure_index", attributes={"db": DATABASE_URL, "embed_model": EMBED_MODEL}, kind=_trace.SpanKind.INTERNAL) if tracer else nullcontext()
    with ctx:
        if _index is None:
            Settings.embed_model = SimpleOllamaEmbedding(EMBED_MODEL, OLLAMA_URL)
            store = PGVectorStore(
                connection_string=_to_sqla_sync(DATABASE_URL),
                async_connection_string=_to_sqla_async(DATABASE_URL),
                schema_name="public",
                table_name="llamaindex_chunks",
                embed_dim=_embed_dim(),
            )
            sc = StorageContext.from_defaults(vector_store=store)
            _index = VectorStoreIndex.from_vector_store(vector_store=store, storage_context=sc)
            _retriever = _index.as_retriever(similarity_top_k=max(TOP_K, TOP_N))
    return _index

def _retrieve(query: str) -> List[NodeWithScore]:
    ctx = tracer.start_as_current_span("retrieve", attributes={"input.query": query, "top_k": TOP_K}, kind=_trace.SpanKind.INTERNAL) if tracer else nullcontext()
    with ctx:
        _ensure_index()
        nodes: List[NodeWithScore] = _retriever.retrieve(query)  # type: ignore
        if SimilarityPostprocessor is not None and SIM_CUTOFF > 0:
            nodes = SimilarityPostprocessor(similarity_cutoff=SIM_CUTOFF).postprocess_nodes(nodes)  # type: ignore
        if tracer:
            try:
                span = _get_current_span()
                span.set_attribute("output.retrieved_nodes_count", len(nodes))
                span.add_event("output", {"value": {"count": len(nodes), "sample": nodes[0].text[:100] if nodes else "None"}})
            except Exception as e:
                log.warning("Failed to set retrieve attributes: %s", e)
        return nodes

# ---------- Re-ranking (with graceful fallbacks) ----------
try:
    from rag.rerank import Scored, mmr_rerank, cross_encoder_rerank
except Exception:
    from dataclasses import dataclass

    @dataclass
    class Scored:
        text: str
        score: float
        meta: Dict[str, Any]

    def mmr_rerank(candidates: List[Scored], top_n: int = 8) -> List[Scored]:
        return sorted(candidates, key=lambda x: x.score, reverse=True)[:top_n]

    def cross_encoder_rerank(query: str, texts: List[Scored], top_n: int = 8) -> List[Scored]:
        return mmr_rerank(texts, top_n=top_n)

def _post_process(nodes: List[NodeWithScore], query: str) -> List[Scored]:
    ctx = tracer.start_as_current_span("post_process", attributes={"input.nodes_count": len(nodes), "rerank_strategy": RERANK_STRATEGY}, kind=_trace.SpanKind.INTERNAL) if tracer else nullcontext()
    with ctx:
        scored: List[Scored] = []
        for n in nodes:
            node = getattr(n, "node", n)
            score = float(getattr(n, "score", 0.0) or 0.0)
            content = ""
            if hasattr(node, "get_content"):
                content = node.get_content()
            elif hasattr(node, "text"):
                content = node.text
            meta = getattr(node, "metadata", {}) or {}
            scored.append(Scored(text=content, score=score, meta=meta))

        if RERANK_STRATEGY == "ce":
            return cross_encoder_rerank(query, scored, top_n=TOP_N)
        if RERANK_STRATEGY == "mmr":
            return mmr_rerank(scored, top_n=TOP_N)
        return sorted(scored, key=lambda x: x.score, reverse=True)[:TOP_N]
        if tracer:
            try:
                span = _get_current_span()
                span.set_attribute("output.scored_count", len(scored))
                span.add_event("output", {"value": {"count": len(scored), "sample_score": scored[0].score if scored else 0}})
            except Exception as e:
                log.warning("Failed to set post_process attributes: %s", e)

def _build_context(items: List[Scored]) -> Tuple[str, List[Dict[str, Any]]]:
    ctx = tracer.start_as_current_span("build_context", attributes={"input.items_count": len(items)}, kind=_trace.SpanKind.INTERNAL) if tracer else nullcontext()
    with ctx:
        parts: List[str] = []
        sources: List[Dict[str, Any]] = []
        total = 0
        for it in items:
            meta = it.meta or {}
            tag = f"{meta.get('filename', meta.get('uri', ''))} #{meta.get('chunk_index', '?')}"
            block = f"[Source: {tag}]\n{it.text}\n\n"
            if total + len(block) > MAX_CONTEXT:
                break
            parts.append(block)
            total += len(block)
            sources.append(
                {"filename": meta.get("filename"), "chunk": meta.get("chunk_index"), "uri": meta.get("uri")}
            )
        context_text = "".join(parts)
        if tracer:
            try:
                span = _get_current_span()
                span.set_attribute("output.context_length_chars", total)
                span.add_event("output", {"value": {"text_snippet": context_text[:500] + "..." if context_text else "None"}})
            except Exception as e:
                log.warning("Failed to set build_context attributes: %s", e)
        return context_text, sources

def _chat(query: str, context_md: str) -> str:
    ctx = tracer.start_as_current_span("llama_chat", attributes={"input.llm_model": LLM_MODEL}, kind=_trace.SpanKind.INTERNAL) if tracer else nullcontext()
    with ctx:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT + "\n\nContext:\n" + context_md},
            {"role": "user", "content": query},
        ]
        prompt = SYSTEM_PROMPT + "\n\nContext:\n" + context_md + "\n\n" + query
        call_ctx = tracer.start_as_current_span("call_ollama", attributes={"input.url": OLLAMA_URL, "input.prompt_length": len(prompt.split())}, kind=_trace.SpanKind.CLIENT) if tracer else nullcontext()
        with call_ctx:
            session = requests.Session()
            adapter = requests.adapters.HTTPAdapter(max_retries=0)
            session.mount("http://", adapter)
            r = session.post(
                f"{OLLAMA_URL.rstrip('/')}/api/chat",
                json={"model": LLM_MODEL, "messages": messages, "stream": False},
                timeout=600,
            )
            try:
                r.raise_for_status()
            finally:
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
        if isinstance(data, dict) and "message" in data:
            return (data["message"] or {}).get("content", "").strip()
        if isinstance(data, dict) and "choices" in data and data["choices"]:
            return (data["choices"][0]["message"] or {}).get("content", "").strip()
        return str(data)

# ---------- Public entry points ----------
def answer(query: str) -> str:
    if not query or not query.strip():
        return "Please provide a question."
    try:
        parent_span = _get_current_span()
        ctx = tracer.start_as_current_span("rag_answer", attributes={"input.query": query}, kind=_trace.SpanKind.INTERNAL) if tracer and not parent_span.is_recording() else nullcontext()
        with ctx:
            nodes = _retrieve(query)
            items = _post_process(nodes, query)
            ctx_text, _ = _build_context(items)
            if not ctx_text.strip():
                return "I don't have enough information in the knowledge base to answer that confidently."
            response = _chat(query, ctx_text)
            if tracer:
                try:
                    span = _get_current_span()
                    span.set_attribute("output.response", response[:500] + "..." if len(response) > 500 else response)
                    span.set_attribute("output.total_duration_ms", int(time.time() * 1000) - span.start_time)
                except Exception as e:
                    log.warning("Failed to set answer attributes: %s", e)
            return response
    except requests.exceptions.ConnectionError:
        if tracer:
            try:
                _get_current_span().add_event("error", {"message": "Ollama unreachable"})
            except Exception:
                pass
        return "The model/embedding service at OLLAMA_URL seems unreachable. Make sure Ollama is running and OLLAMA_URL is correct."
    except Exception as e:
        if tracer:
            try:
                _get_current_span().add_event("error", {"message": str(e)})
            except Exception:
                pass
        return f"Error while answering: {e}"

class QARunnable:
    def invoke(self, inputs: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        q = inputs.get("question") if isinstance(inputs, dict) else str(inputs)
        return {"answer": answer(str(q))}

    async def ainvoke(self, inputs: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        loop = asyncio.get_event_loop()
        q = inputs.get("question") if isinstance(inputs, dict) else str(inputs)
        res = await loop.run_in_executor(None, answer, str(q))
        return {"answer": res}

def get_graph() -> QARunnable:
    try:
        _ensure_index()
    except Exception:
        pass
    return QARunnable()

if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "What can you do?"
    print(answer(q))