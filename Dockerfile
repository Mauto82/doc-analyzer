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

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ .

ENV OLLAMA_HOST=${OLLAMA_HOST}
ENV OLLAMA_PORT=${OLLAMA_PORT}

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
