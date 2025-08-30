# Open WebUI + Contextual RAG (Docling · LlamaIndex · PGVector · Ollama)

Local-first, production-leaning starter. Drop PDFs in `./docs/`, run Docker, ingest, and chat in Open WebUI.

## Quick Start

1) Put your PDFs in `./docs/`.

2) Start services:
```bash
docker compose up -d --build
docker compose logs -f ollama-init  # watch model pulls (first run only)
```

3) Ingest docs into PGVector:
```bash
docker compose exec api python -m rag.pipelines.ingest --path /app/docs
```

4) Open Open WebUI: http://localhost:3000  
Settings → Provider = **OpenAI Compatible**  
API Base = `http://localhost:8000/v1`  
API Key = any non-empty value (e.g., `sk-local`)  
Model = `llama3.1:8b`

## Notes
- Postgres runs the **pgvector** distribution.
- Defaults to password=`admin`.
- Adjust `.env` if you change models or ports.
