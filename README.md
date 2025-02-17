# Doc Analyzer

A web application that analyzes PDF documents using DeepSeek R1 language model and RAG (Retrieval-Augmented Generation) architecture.

## Overview

Doc Analyzer enables users to:
- Upload PDF documents
- Ask questions about document content
- Receive AI-generated responses based on document content
- Process documents using state-of-the-art language models

The application leverages:
- DeepSeek R1 for text generation and embeddings
- ChromaDB for vector storage
- LangChain for document processing
- FastAPI and Gradio for the web interface

## Requirements

- Docker and Docker Compose
- Ollama with DeepSeek R1 1.5b (or later) model
- 8GB RAM minimum
- 20GB disk space

### Installing Ollama

1. Install Ollama:
   - Linux: `curl https://ollama.ai/install.sh | sh`
   - macOS/Windows: Download from https://ollama.ai

2. Pull DeepSeek model:
```bash
ollama pull deepseek-r1:1.5b
```

## Quick Start with Docker

1. Clone and setup:
```bash
git clone https://github.com/Mauto82/doc-analyzer
cd doc-analyzer
cp .env.example .env
```

2. Configure `.env`:
```env
OLLAMA_HOST=host.docker.internal  # Use 'localhost' for local dev
OLLAMA_PORT=11434
CHROMA_DB_PATH=/app/data/chroma
DEEPSEEK_MODEL=deepseek-r1:1.5b
```

3. Start application:
```bash
docker-compose build
docker-compose up -d
```

4. Access: http://localhost:8000

## Usage Guide

1. **Upload Document**
   - Click upload area or drag-and-drop PDF file
   - Wait for processing completion

2. **Ask Questions**
   - Type your question in the text input
   - Click "Submit" or press Enter
   - Wait for AI-generated response

3. **Best Practices**
   - Use clear, specific questions
   - Ask one question at a time
   - For complex documents, start with general questions
   - Wait for each response before asking next question

## Architecture

The application consists of several components:

### Web Interface (FastAPI + Gradio)
- Handles file uploads and user interactions
- Provides intuitive interface
- Manages user sessions

### PDF Processor
- Extracts text from PDF documents
- Splits content into manageable chunks
- Uses LangChain for document handling

### RAG Processor
- Creates embeddings using DeepSeek
- Stores vectors in ChromaDB
- Retrieves relevant content for queries
- Generates responses using DeepSeek

### Vector Store (ChromaDB)
- Stores document embeddings
- Enables semantic search
- Maintains document-query relevance

## Local Development

### Without Docker
```bash
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows
pip install -r requirements.txt
pip install -r requirements-dev.txt
mkdir -p ./data/chroma
chmod -R 777 ./data
uvicorn src.app:app --reload
```

### With Docker
```bash
# Start with logs
docker-compose up

# View logs
docker-compose logs -f

# Rebuild after changes
docker-compose build

# Restart services
docker-compose restart
```

## Troubleshooting

### Database Issues
If you get "readonly database" error in local development:
```bash
chmod -R 777 ./data
rm -rf ./data/chroma/*
```

### Ollama Connection
1. Verify Ollama is running: `ollama list`
2. Check service: `curl http://localhost:11434/api/tags`
3. Check `.env` configuration matches your setup
4. For Docker Desktop users, ensure `host.docker.internal` is used

### Memory Issues
- Docker Desktop: Increase memory limit (Preferences → Resources → Memory)
- Recommended minimum: 8GB
- If needed, reduce chunk size in `pdf_processor.py`:
```python
chunk_size=500  # Decrease if experiencing memory issues
chunk_overlap=100
```

## Contributing

1. Fork repository
2. Create feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push branch: `git push origin feature-name`
5. Submit Pull Request

### Development Guidelines
- Follow PEP 8 style guide
- Add tests for new features
- Update documentation
- Keep commits focused and clean

## License

MIT License - see LICENSE file for details.