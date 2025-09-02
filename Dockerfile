# Use Python 3.11 slim base image for a minimal footprint
FROM python:3.11-slim

# Set environment variables to prevent bytecode writing and buffer output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/root/.local/bin:${PATH}"

# Install system dependencies in a single RUN command to reduce layers
# build-essential: For compiling Python dependencies
# libpq-dev: For PostgreSQL client libraries
# curl: For downloading uv package installer
# tesseract-ocr: For OCR processing in document ingestion
# libtesseract-dev: Tesseract development libraries
# libblas-dev, liblapack-dev: For numerical computations (sentence-transformers)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    tesseract-ocr \
    libtesseract-dev \
    libblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv package installer for faster dependency installation
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Set working directory to /app
WORKDIR /app

# Copy requirements.txt for dependency installation (cached layer)
COPY requirements.txt /app/requirements.txt

# Install Python dependencies using uv
RUN uv pip install --system -r /app/requirements.txt

# Copy ingestion script
COPY apps/api/rag/pipelines/ingest.py /app/ingest.py

# Copy rag directory with all RAG-related code
COPY apps/api/rag /app/rag

# Copy main.py and other Python files in apps/api
COPY apps/api/main.py /app/
COPY apps/api/*.py /app/

# Expose port 8000 for the FastAPI application
EXPOSE 8000

# Run FastAPI application with uvicorn in debug mode
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "debug"]