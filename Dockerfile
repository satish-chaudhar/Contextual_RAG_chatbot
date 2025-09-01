FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/root/.local/bin:${PATH}"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    tesseract-ocr \
    libtesseract-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv (faster package installer than pip)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Set working directory
WORKDIR /app

# Copy requirements.txt first (for caching)
COPY requirements.txt /app/requirements.txt

# Install dependencies with uv
RUN uv pip install --system -r /app/requirements.txt

COPY apps/api/rag/pipelines/ingest.py /app/ingest.py

# Copy rag directory
COPY apps/api/rag /app/rag

# Copy main.py and other Python files
COPY apps/api/main.py /app/
COPY apps/api/*.py /app/

# Expose API port
EXPOSE 8000

# Run application with debug logs
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "debug"]
