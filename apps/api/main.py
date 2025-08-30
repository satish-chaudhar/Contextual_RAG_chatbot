# apps/api/main.py
from __future__ import annotations
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import os
import requests
import time
import logging

# OTel imports for global tracer and context
from opentelemetry import trace
from opentelemetry.propagate import extract

# Register OTEL (your custom phoenix_otel.py)
from phoenix_otel import register as register_otel
provider = register_otel()  # Returns TracerProvider or None

# If provider set, make it global
if provider:
    trace.set_tracer_provider(provider)

# Instrument LlamaIndex if available
try:
    from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
    LlamaIndexInstrumentor().instrument()
except Exception as e:
    logging.getLogger("phoenix_otel").warning("LlamaIndexInstrumentor not available; skipping: %s", e)

# RAG function import (unchanged)
from rag.core.graph import answer, get_graph  # ensure get_graph available for pre-warm

app = FastAPI()

# startup: instrument app (if instrumentation exists) and pre-warm index
@app.on_event("startup")
async def startup_event():
    # Try to instrument FastAPI automatically if opentelemetry FastAPI instrumentation is available.
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        FastAPIInstrumentor().instrument_app(app)
    except Exception:
        # This is optional and non-fatal; we continue even if not present.
        logging.getLogger("phoenix_otel").debug("FastAPIInstrumentor not available; skipping automatic instrumentation", exc_info=True)

    # Pre-warm index / retriever so first request is faster (single execution)
    try:
        get_graph()
    except Exception:
        logging.getLogger("rag").exception("Pre-warm _ensure_index failed (DB might not be ready yet)")

@app.get("/v1/models")
def list_models():
    try:
        ollama_url = os.getenv('OLLAMA_URL', 'http://ollama:11434')
        r = requests.get(f"{ollama_url}/api/tags")
        r.raise_for_status()
        ollama_models = r.json().get("models", [])
        models = {
            "object": "list",
            "data": [
                {
                    "id": model.get("name", "unknown"),
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "ollama"
                } for model in ollama_models
            ]
        }
        return models
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/v1/chat/completions")
def chat_completions(request: dict):
    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span("chat_completions_endpoint", kind=trace.SpanKind.SERVER) as span:
        query = request['messages'][-1]['content']
        span.set_attribute("input", {"query": query, "model": request.get('model', 'llama3.2:3b')})  # Structured input
        try:
            # Pass context to answer
            rag_response = answer(query)
            span.set_attribute("output", {"response": rag_response[:500] + "..." if len(rag_response) > 500 else rag_response})
            return {
                "id": "chatcmpl-" + str(int(time.time())),
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request.get('model', 'llama3.2:3b'),
                "choices": [{"index": 0, "message": {"role": "assistant", "content": rag_response}, "finish_reason": "stop"}]
            }
        except Exception as e:
            span.add_event("error", {"message": str(e)})
            return JSONResponse(content={"error": str(e)}, status_code=500)