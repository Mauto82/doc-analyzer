# Doc Analyzer

A web application that analyzes documents and code files using local large language models via Ollama and RAG (Retrieval-Augmented Generation). No cloud, no API keys — everything runs on your machine.

## Overview

Doc Analyzer enables you to:
- Upload and analyze **PDF, DOCX, DOC, TXT, RTF** documents
- Process **30+ code file formats** (Python, JS, Java, Go, Rust, and more)
- Handle **tabular data** (Excel, CSV, ODS, JSON)
- Process **Markdown** and **YAML** files
- Add multiple documents to the context and query across all of them
- Ask questions and get **streaming AI responses** generated progressively
- Select any **LLM model** installed in Ollama
- Use a **dedicated embedding model** (`mxbai-embed-large`) separate from the LLM
- Extract text from **scanned or vector-path PDFs** via OCR fallback

The application uses:
- **Ollama** — local LLM inference and embeddings
- **Qdrant** — local vector store (in-memory or file-based)
- **FastAPI** — REST API backend
- **Vanilla HTML/JS** — lightweight, zero-dependency frontend

## Project Structure

```
doc-analyzer/
├── src/
│   ├── app.py                      # FastAPI app + inline HTML/JS frontend
│   ├── config/
│   │   └── prompts.py              # Role-based prompts
│   ├── models/
│   │   └── document.py             # Document dataclass
│   ├── utils/
│   │   └── text_splitter.py        # Recursive text splitter
│   └── processors/
│       ├── base/
│       │   └── document_processor.py
│       ├── factory.py
│       ├── pdf_processor.py        # PDF + OCR fallback
│       ├── word_processor.py
│       ├── text_processor.py
│       ├── rtf_processor.py
│       ├── code_processor.py
│       ├── table_processor.py
│       └── rag_processor.py        # Qdrant + Ollama RAG
├── tests/
│   ├── processors/
│   └── unit/
├── data/
│   └── qdrant/                     # Qdrant file storage (when PERSIST_VECTORDB=true)
├── Dockerfile
├── docker-compose.yml
├── docker-compose.test.yml
├── requirements.txt
└── setup.py
```

## Requirements

- Docker and Docker Compose (recommended)
- Ollama running locally with at least one LLM model and `mxbai-embed-large`
- 8 GB RAM minimum (more for larger models)

### Ollama setup

1. Install Ollama: https://ollama.ai

2. Pull the required models:
```bash
ollama pull mxbai-embed-large     # embedding model (required)
ollama pull qwen2.5:14b           # or any other LLM you prefer
```

## Quick Start

### With Docker (recommended)

```bash
git clone https://github.com/Mauto82/doc-analyzer
cd doc-analyzer
cp .env.example .env
# Edit .env and set LLM_MODEL to the model you pulled
docker compose up -d
```

Open http://localhost:8000

### Local development

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env — make sure OLLAMA_HOST=localhost
uvicorn src.app:app --reload --port 8000
```

## Configuration

Copy `.env.example` to `.env` and adjust:

```env
# LLM model for text generation (any model available in Ollama)
LLM_MODEL=qwen2.5:14b

# Dedicated embedding model
EMBEDDING_MODEL=mxbai-embed-large:latest

# Ollama connection
OLLAMA_HOST=localhost        # use 'localhost' for local dev
OLLAMA_PORT=11434

# Vector store
QDRANT_DB_PATH=./data/qdrant
PERSIST_VECTORDB=false      # set to 'true' to keep data between restarts

# Chunking
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

> **Docker note:** `docker-compose.yml` automatically overrides `OLLAMA_HOST` to `host.docker.internal` so the container can reach Ollama on the host machine.

> **Qdrant persistence:** With `PERSIST_VECTORDB=false` (default), the vector store is in-memory and resets on every restart. Set to `true` to persist data in `./data/qdrant` (volume-mounted in Docker).

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Web interface |
| `GET` | `/api/status` | Server status, loaded files, available models |
| `POST` | `/api/upload` | Upload and index a document |
| `DELETE` | `/api/files/{name}` | Remove a specific document from context |
| `DELETE` | `/api/files` | Clear all documents |
| `POST` | `/api/query` | Query (full response) |
| `POST` | `/api/query/stream` | Query with SSE streaming response |

## Supported File Types

### Documents
- PDF (`.pdf`) — with automatic OCR fallback for scanned/vector-path PDFs
- Microsoft Word (`.doc`, `.docx`)
- Rich Text Format (`.rtf`)
- Plain Text (`.txt`)
- Markdown (`.md`)

### Tabular Data
- Excel (`.xlsx`, `.xls`)
- CSV (`.csv`)
- OpenDocument Spreadsheet (`.ods`)
- JSON (`.json`)

### Configuration
- YAML (`.yaml`, `.yml`)

### Code Files
Python, JavaScript, TypeScript, Java, C/C++, C#, PHP, Go, Ruby, Rust, HTML, CSS, and many more.

> **Dockerfiles** have no extension — rename them (e.g. `Dockerfile.txt`) before uploading. The processor will detect Dockerfile content automatically.

## Usage

1. **Upload documents** — click "Upload Document" in the sidebar. Each file is chunked and indexed into the vector store. Multiple documents can be loaded simultaneously.

2. **Ask questions** — type in the chat input and press Enter (or Shift+Enter for newline). The response streams progressively as the model generates it.

3. **Select role** — choose an analysis perspective:
   - **default** — general document analysis
   - **legal** — legal implications and regulatory analysis
   - **financial** — costs, ROI, economic considerations
   - **technical** — implementation details and architecture
   - **travel** — travel info, logistics, attractions
   - **travel_agent** — conversational travel recommendations

4. **Select model** — all models installed in Ollama are available. The default from `.env` is preselected.

5. **Remove documents** — click `×` next to a file to remove it from context, or "Clear All" to reset everything.

## Architecture

### Frontend
Vanilla HTML/JS page served by FastAPI. No framework, no build step, no CDN dependencies. Communicates with the backend via REST (`fetch`) and reads streaming responses using the `ReadableStream` API.

### RAG Processor (`rag_processor.py`)
- Embeds chunks using `mxbai-embed-large` via `ollama.Client.embed()`
- Stores vectors in Qdrant (cosine distance, 1024 dimensions)
- On query: embeds the question, retrieves top-4 chunks, builds a prompt with context, generates the answer via `ollama.Client.chat()`
- Streaming: `chat(stream=True)` returns a generator; each token is forwarded as an SSE event
- Automatic retry with progressive truncation (80% each attempt) when a chunk exceeds the embedding model's context length

### PDF Processor (`pdf_processor.py`)
- Extracts text with PyMuPDF (`fitz`)
- If a page returns no text (scanned PDF or vector-path text from macOS Quartz), falls back to OCR: renders the page at 300 DPI and runs `pytesseract.image_to_string()`

### Vector Store
Qdrant in in-memory mode (`:memory:`) by default, or file-based when `PERSIST_VECTORDB=true`. Each chunk is stored as a point with its text and source metadata. Documents are removed by their chunk UUIDs.

## Running Tests

```bash
# With Docker
docker compose -f docker-compose.test.yml up --abort-on-container-exit

# Locally
pip install -r requirements-dev.txt
pytest
```

## Troubleshooting

### Ollama not reachable
- Local dev: verify `OLLAMA_HOST=localhost` and Ollama is running (`ollama list`)
- Docker: `OLLAMA_HOST` is overridden to `host.docker.internal` automatically
- Check: `curl http://localhost:11434/api/tags`

### "No content extracted" on PDF upload
The PDF likely contains only scanned images or vector-drawn text. Make sure `pytesseract` and `tesseract-ocr` are installed (they are included in the Docker image). For local dev:
```bash
brew install tesseract        # macOS
sudo apt install tesseract-ocr  # Ubuntu/Debian
pip install pytesseract Pillow
```

### "Input length exceeds context length" during upload
This is handled automatically — the processor retries with progressively shorter chunks. If it persists, reduce `CHUNK_SIZE` in `.env`.

### Qdrant data lost after restart
Set `PERSIST_VECTORDB=true` in `.env`. The `./data` directory is volume-mounted in Docker, so data will survive restarts.

### Docker container management
```bash
docker compose up -d           # start
docker compose up --build -d   # rebuild and start
docker compose restart         # restart without rebuild
docker compose logs -f         # follow logs
docker stop doc-analyzer-web-1 # stop specific container
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit your changes: `git commit -m 'Add feature'`
4. Push and open a Pull Request

Follow PEP 8, add tests for new features, update docs.

## License

MIT License — see LICENSE file for details.
