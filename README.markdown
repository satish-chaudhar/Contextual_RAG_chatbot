# Contextual RAG Chatbot

Welcome to the Contextual RAG Chatbot, a sophisticated Retrieval-Augmented Generation (RAG) application designed to provide accurate, context-aware responses from a knowledge base of PDF documents. Built with FastAPI, LlamaIndex, PGVector, Open WebUI, Crew.AI, and enhanced with input guardrails, this project is ideal for enterprise use cases such as summarizing procurement standards or HR bylaws.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Architecture](#architecture)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Overview
The Contextual RAG Chatbot leverages advanced natural language processing to retrieve relevant document chunks from a PostgreSQL database using PGVector and generates responses with Ollama’s language models. It supports conversation history for the last 15 chats, optimizes prompts with Crew.AI, and includes input guardrails to ensure safe and valid queries.

## Features
- **RAG Pipeline**: Retrieves and augments responses with LlamaIndex and PGVector.
- **Conversation Memory**: Preserves the last 15 chat interactions for context.
- **Prompt Optimization**: Uses Crew.AI to refine queries and responses (optional).
- **Input Guardrails**: Validates and sanitizes inputs to prevent malicious content.
- **Performance Monitoring**: Integrates Arize Phoenix for tracing and evaluation.
- **User Interface**: Accessible via Open WebUI.

## Requirements
### Hardware
- **CPU**: Multi-core processor (minimum 2 cores, 4 recommended).
- **RAM**: 6GB minimum, 8GB recommended.
- **Storage**: 2GB for application, plus space for document storage.

### Software
- **Docker** and **Docker Compose** (latest stable version).
- **Python 3.11**.
- **PostgreSQL** with PGVector extension.
- **Ollama** for LLM and embedding models.

## Installation
### Prerequisites
1. Install Docker and Docker Compose on your system. Follow the official guides:
   - [Docker Installation](https://docs.docker.com/get-docker/)
   - [Docker Compose Installation](https://docs.docker.com/compose/install/)
2. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/openwebui-contextual-rag-v2.git
   cd openwebui-contextual-rag-v2
   ```

### Configuration
1. Create a `.env` file in the project root with the following variables:
   ```
   DATABASE_URL=postgresql://postgres:admin@postgres:5432/postgres
   OLLAMA_URL=http://ollama:11434
   EMBED_MODEL=nomic-embed-text
   LLM_MODEL=llama3.2:3b
   OPENAI_API_KEY=your-openai-api-key
   OPTIMIZE_CREWAI=false
   OPTIMIZE_EVALS=false
   ```
   - Replace `your-openai-api-key` with a valid OpenAI API key if using Crew.AI or Phoenix evaluators.

### Building and Running
1. Build the Docker images and start the services:
   ```bash
   docker compose build
   docker compose up -d
   ```
2. Verify services are running:
   - Check logs: `docker compose logs`
   - Access Open WebUI at `http://localhost:3000`

## Usage
### Ingestion Process
To populate the knowledge base with documents:
1. Place PDF, TXT, or MD files in the `docs` folder.
2. Run the ingestion script:
   ```bash
   docker compose exec api python /app/ingest.py --path /app/docs
   ```
   - Monitor progress with `docker compose logs api --tail=100 -f`.

### Interacting with the Chatbot
1. Open your browser and navigate to `http://localhost:3000`.
2. Log in with default credentials or configure as needed.
3. Enter queries such as "Summarize section 3 of the document" or "What was my first question?" to test.

## Architecture
### Components
- **FastAPI**: RESTful API server handling requests.
- **LlamaIndex**: Indexes and retrieves documents with vector search.
- **PGVector**: Stores embeddings in PostgreSQL for efficient retrieval.
- **Ollama**: Provides LLM and embedding models (e.g., `llama3.2:3b`, `nomic-embed-text`).
- **Crew.AI**: Optimizes prompts and responses (optional).
- **Phoenix**: Monitors performance and traces execution.
- **Open WebUI**: User-friendly chat interface.

### Workflow
1. User submits a query via Open WebUI.
2. FastAPI validates the input with guardrails.
3. LlamaIndex retrieves relevant chunks from PGVector.
4. Ollama generates a response using retrieved context and chat history.
5. Crew.AI optimizes the query/response if enabled.
6. Phoenix logs the trace for analysis.
7. Response is returned to the user.

## Configuration
- **Environment Variables**: Defined in `.env` (see Installation).
- **Customizable Parameters**:
  - `TOP_K`: Number of documents to retrieve (default 24).
  - `TOP_N`: Number of documents to rerank (default 8).
  - `MAX_CONTEXT_CHARS`: Maximum context length (default 12000).
  - Adjust in `.env` or environment settings as needed.

## Troubleshooting
### Common Issues
- **Ingestion Fails**: Check logs for "Failed to build index" and ensure PostgreSQL is healthy (`docker compose logs postgres`).
- **No Response**: Verify documents are ingested (`SELECT COUNT(*) FROM public.llamaindex_chunks`) and Ollama is running.
- **Extra Tasks in Phoenix**: Disable `OPTIMIZE_CREWAI` and `OPTIMIZE_EVALS` in `.env`.
- **Performance Slow**: Increase `memory` limit in `docker-compose.yml` or optimize `ingest.py` batch size.

## Contributing
1. Fork the repository.
2. Create a feature branch: `git checkout -b feature-name`.
3. Commit changes: `git commit -m "Add feature name"`.
4. Push to the branch: `git push origin feature-name`.
5. Submit a pull request.
