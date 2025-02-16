FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    poppler-utils \
    tesseract-ocr \
    libtesseract-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create data directory with correct permissions
RUN mkdir -p /app/data && chmod 777 /app/data
RUN mkdir -p /app/data/chroma && chmod 777 /app/data/chroma

# Copy requirements and setup files
COPY requirements.txt requirements-dev.txt setup.py ./

# Copy source code and tests
COPY src/ ./src/
COPY tests/ ./tests/

# Install dependencies and package in development mode
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -r requirements-dev.txt \
    && pip install -e .

# Environment variables
ENV OLLAMA_HOST=${OLLAMA_HOST}
ENV OLLAMA_PORT=${OLLAMA_PORT}
ENV PYTHONPATH=/app/src

EXPOSE 8000

# Default command remains the same
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]