# apps/api/main.py
from __future__ import annotations
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import os
import requests
import time
import logging
import traceback
import json

# OTel imports for global tracer and context
from opentelemetry import trace
from opentelemetry.propagate import extract

# Register OTEL (your custom phoenix_otel.py)
from phoenix_otel import register as register_otel
provider = register_otel()  # Returns TracerProvider or None

# If provider set, make it global (avoid override warning)
if provider and not trace.get_tracer_provider():
    trace.set_tracer_provider(provider)

# Instrument LlamaIndex if available
try:
    from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
    LlamaIndexInstrumentor().instrument()
except ImportError as e:
    logging.getLogger("phoenix_otel").warning("LlamaIndexInstrumentor not available; skipping: %s", e)
except Exception as e:
    logging.getLogger("phoenix_otel").warning("Unexpected error with LlamaIndexInstrumentor; skipping: %s", e)

# Phoenix evaluator setup (optional, with fallback)
evaluators = None
try:
    from phoenix.evals import QAEval, HallucinationEval, RelevanceEval
    from phoenix import evaluate
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        evaluators = {
            "qa_correctness": QAEval(model="gpt-4o-mini", api_key=api_key),
            "hallucination": HallucinationEval(model="gpt-4o-mini", api_key=api_key),
            "relevance": RelevanceEval(model="gpt-4o-mini", api_key=api_key),
        }
        logging.getLogger("phoenix_evals").info("Evaluators initialized with OpenAI API key.")
    else:
        logging.getLogger("phoenix_evals").warning("OPENAI_API_KEY not set; evaluations disabled.")
except (ImportError, Exception) as e:
    logging.getLogger("phoenix_evals").warning("Phoenix evaluators not available; skipping: %s", e)

# Fallback for LLM semantic attributes
try:
    from opentelemetry.semantics.genai.attributes import GenAiAttributes as GenAiAttr
except ImportError:
    logging.getLogger("phoenix_otel").warning("opentelemetry-semantic-conventions-ai not found; using fallback LLM attributes.")
    class GenAiAttr:
        LLM_MODEL_NAME = "llm.model.name"
        LLM_OUTPUT_MESSAGES = "llm.output_messages"

# RAG function import
from rag.core.graph import answer, get_graph

app = FastAPI()

# startup: instrument app and pre-warm index
@app.on_event("startup")
async def startup_event():
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        FastAPIInstrumentor().instrument_app(app)
    except Exception:
        logging.getLogger("phoenix_otel").debug("FastAPIInstrumentor not available; skipping automatic instrumentation", exc_info=True)

    try:
        get_graph()
    except Exception as e:
        logging.getLogger("rag").exception("Pre-warm _ensure_index failed: %s", e)

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
        if not request.get("messages") or not isinstance(request["messages"], list):
            return JSONResponse(content={"error": "Invalid request: 'messages' must be a non-empty list"}, status_code=422)
        query = request['messages'][-1]['content']
        # Pass full conversation history
        chat_history = request.get("messages", [])[:-1]  # Exclude current query
        span.set_attribute("input.value", query)
        span.set_attribute(GenAiAttr.LLM_MODEL_NAME, request.get('model', 'llama3.2:3b'))
        try:
            rag_response = answer(query, chat_history)
            span.set_attribute("output.value", rag_response)
            span.add_event("llm.response", attributes={GenAiAttr.LLM_OUTPUT_MESSAGES: json.dumps({"content": rag_response})})
            if evaluators:
                evaluation = evaluate(
                    input=query,
                    output=rag_response,
                    reference="Expected response from your eval set"
                , evaluators=[evaluators["qa_correctness"], evaluators["hallucination"], evaluators["relevance"]]
                )
                span.add_event("evaluations", attributes={
                    "qa_correctness": evaluation.get("qa_correctness", "N/A"),
                    "hallucination": evaluation.get("hallucination", "N/A"),
                    "relevance": evaluation.get("relevance", "N/A")
                })
            return {
                "id": "chatcmpl-" + str(int(time.time())),
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request.get('model', 'llama3.2:3b'),
                "choices": [{"index": 0, "message": {"role": "assistant", "content": rag_response}, "finish_reason": "stop"}]
            }
        except Exception as e:
            logging.error("Chat completion error: %s\n%s", e, traceback.format_exc())
            span.add_event("error", {"message": str(e)})
            return JSONResponse(content={"error": str(e)}, status_code=500)