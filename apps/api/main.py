# Enable annotations for better type hinting support
from __future__ import annotations

# Import FastAPI and JSONResponse for API handling
from fastapi import FastAPI
from fastapi.responses import JSONResponse
# Import lifespan for modern event handling
from contextlib import asynccontextmanager

# Import standard libraries for system, networking, and logging
import os
import requests
import time
import logging
import traceback
import json
import uuid

# Import OpenTelemetry for tracing
from opentelemetry import trace
from opentelemetry.propagate import extract

# Import custom OpenTelemetry registration
from phoenix_otel import register as register_otel

# Import RAG functions
from rag.core.graph import answer, get_graph, optimize_prompt

# Import chat history management
from chat_history import ChatHistory

# Register tracer provider
provider = register_otel()  # Returns TracerProvider or None

# Set global tracer provider if not already set
if provider and not trace.get_tracer_provider():
    trace.set_tracer_provider(provider)

# Instrument LlamaIndex for tracing
try:
    from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
    LlamaIndexInstrumentor().instrument()  # Enable LlamaIndex tracing
except ImportError as e:
    logging.getLogger("phoenix_otel").warning("LlamaIndexInstrumentor not available; skipping: %s", e)
except Exception as e:
    logging.getLogger("phoenix_otel").warning("Unexpected error with LlamaIndexInstrumentor; skipping: %s", e)

# Initialize evaluators (with fallback for import issues)
evaluators = None
try:
    from phoenix.evals import QAEvaluator, HallucinationEvaluator, RelevanceEvaluator
    from phoenix import evaluate
    api_key = os.getenv("OPENAI_API_KEY")  # Get OpenAI API key
    optimize_evals = os.getenv("OPTIMIZE_EVALS", "false").lower() == "true"  # Check if evaluations enabled
    if api_key and optimize_evals:
        # Initialize evaluators with OpenAI model
        evaluators = {
            "qa_correctness": QAEvaluator(model="gpt-4o-mini", api_key=api_key),
            "hallucination": HallucinationEvaluator(model="gpt-4o-mini", api_key=api_key),
            "relevance": RelevanceEvaluator(model="gpt-4o-mini", api_key=api_key),
        }
        logging.getLogger("phoenix_evals").info("Evaluators initialized with OpenAI API key for optimization.")
    else:
        logging.getLogger("phoenix_evals").warning("Evaluations disabled: OPENAI_API_KEY not set or OPTIMIZE_EVALS not enabled.")
except (ImportError, Exception) as e:
    logging.getLogger("phoenix_evals").warning("Phoenix evaluators not available; skipping: %s", e)

# Define fallback for LLM semantic attributes
try:
    from opentelemetry.semantics.genai.attributes import GenAiAttributes as GenAiAttr
except ImportError:
    logging.getLogger("phoenix_otel").warning("opentelemetry-semantic-conventions-ai not found; using fallback LLM attributes.")
    class GenAiAttr:
        LLM_MODEL_NAME = "llm.model.name"
        LLM_OUTPUT_MESSAGES = "llm.output_messages"

# Initialize FastAPI app
app = FastAPI()

# Initialize chat history
chat_history = ChatHistory()

# Define lifespan handler for startup and shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        FastAPIInstrumentor().instrument_app(app)  # Instrument FastAPI for tracing
    except Exception:
        logging.getLogger("phoenix_otel").debug("FastAPIInstrumentor not available; skipping automatic instrumentation", exc_info=True)

    try:
        get_graph()  # Pre-warm index
    except Exception as e:
        logging.getLogger("rag").exception("Pre-warm _ensure_index failed: %s", e)
    
    yield  # Yield control to FastAPI

    # Shutdown logic
    chat_history.close()  # Close chat history database connection

# Attach lifespan handler to app
app.lifespan = lifespan

# Endpoint to list available models
@app.get("/v1/models")
def list_models():
    try:
        ollama_url = os.getenv('OLLAMA_URL', 'http://ollama:11434')  # Get Ollama URL
        r = requests.get(f"{ollama_url}/api/tags")  # Request model list
        r.raise_for_status()  # Raise exception for bad status
        ollama_models = r.json().get("models", [])  # Extract models
        # Format response
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
        return JSONResponse(content={"error": str(e)}, status_code=500)  # Return error response

# Endpoint for chat completions
@app.post("/v1/chat/completions")
def chat_completions(request: dict):
    tracer = trace.get_tracer(__name__)  # Initialize tracer
    # Start tracing span for endpoint
    with tracer.start_as_current_span("chat_completions_endpoint", kind=trace.SpanKind.SERVER) as span:
        # Validate request
        if not request.get("messages") or not isinstance(request["messages"], list):
            return JSONResponse(content={"error": "Invalid request: 'messages' must be a non-empty list"}, status_code=422)
        
        # Extract session ID or generate a new one
        session_id = request.get("session_id", str(uuid.uuid4()))
        chat_history.create_session(session_id)  # Create session if it doesn't exist
        
        # Extract query from the latest message
        query = request['messages'][-1]['content']  # Extract latest message content
        span.set_attribute("input.value", query)  # Log query
        span.set_attribute(GenAiAttr.LLM_MODEL_NAME, request.get('model', 'llama3.2:3b'))  # Log model
        logging.info(f"Processing query: {query} with model: {request.get('model', 'llama3.2:3b')}")  # Log processing
        
        try:
            optimize_crewai = os.getenv("OPTIMIZE_CREWAI", "false").lower() == "true"  # Check if Crew.AI enabled
            optimized_query = query  # Initialize with original query
            # Optimize query if enabled and conditions met
            if optimize_crewai and ("explain" in query.lower() or len(query.split()) > 5):
                optimized_query = optimize_prompt(query, "") if optimize_prompt else query
                logging.info(f"Optimized query: {optimized_query}")
            rag_response = answer(optimized_query)  # Get RAG response
            optimized_response = rag_response  # Initialize with RAG response
            # Optimize response if enabled and conditions met
            if optimize_crewai and ("explain" in query.lower() or len(rag_response.split()) > 50):
                optimized_response = optimize_prompt("", rag_response) if optimize_prompt else rag_response
                logging.info(f"Optimized response: {optimized_response[:100]}...")
            
            # Store the question and answer in the chat history
            chat_history.add_message(session_id, query, optimized_response)
            # Prune old sessions to keep only the last 10
            chat_history.prune_old_sessions()
            
            # Retrieve the last 15 messages for the current session
            history = chat_history.get_session_history(session_id, limit=15)
            # Format history for Open WebUI
            history_messages = [
                {"role": "user", "content": msg["question"]} for msg in reversed(history)
            ] + [
                {"role": "assistant", "content": msg["answer"]} for msg in reversed(history)
            ]
            
            span.set_attribute("output.value", optimized_response)  # Log response
            span.add_event("llm.response", attributes={GenAiAttr.LLM_OUTPUT_MESSAGES: json.dumps({"content": optimized_response})})  # Log response event
            # Run evaluations if enabled
            if evaluators and optimize_evals:
                evaluation = evaluate(
                    input=optimized_query,
                    output=optimized_response,
                    reference="Expected response from your eval set",  # Placeholder reference
                    evaluators=[evaluators["qa_correctness"], evaluators["hallucination"], evaluators["relevance"]]
                )
                # Log evaluation results
                span.add_event("evaluations", attributes={
                    "qa_correctness": evaluation.get("qa_correctness", "N/A"),
                    "hallucination": evaluation.get("hallucination", "N/A"),
                    "relevance": evaluation.get("relevance", "N/A")
                })
                logging.info("Evaluations completed: %s", evaluation)
            
            # Return formatted response with history
            return {
                "id": "chatcmpl-" + str(int(time.time())),
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request.get('model', 'llama3.2:3b'),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": optimized_response},
                        "finish_reason": "stop"
                    }
                ],
                "session_id": session_id,  # Include session ID for continuity
                "history": history_messages  # Include last 15 messages
            }
        except Exception as e:
            logging.error("Chat completion error: %s\n%s", e, traceback.format_exc())  # Log error
            span.add_event("error", {"message": str(e)})  # Log error event
            return JSONResponse(content={"error": str(e)}, status_code=500)  # Return error response