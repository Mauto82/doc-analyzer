FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ .

ENV OLLAMA_HOST=${OLLAMA_HOST}
ENV OLLAMA_PORT=${OLLAMA_PORT}

EXPOSE 8000
