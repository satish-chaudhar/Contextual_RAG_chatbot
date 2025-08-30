FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    tesseract-ocr \
    libtesseract-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements.txt
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --verbose -r /app/requirements.txt

# Copy the rag directory including eval subdirectory
COPY apps/api/rag /app/rag

# Copy main.py and other Python files from apps/api to /app
COPY apps/api/main.py /app/
COPY apps/api/*.py /app/

# Expose port
EXPOSE 8000

# Run the application with debug logging
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "debug"]