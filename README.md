# Doc Analyzer

A web application that analyzes documents and code files using local large language models via Ollama and RAG (Retrieval-Augmented Generation). No cloud, no API keys — everything runs on your machine.

## Overview

Doc Analyzer enables you to:
- Upload and analyze **PDF, DOCX, DOC, TXT, RTF** documents
- Process **30+ code file formats** (Python, JS, Java, Go, Rust, and more)
- Handle **tabular data** (Excel, CSV, ODS, JSON)
- Process **Markdown** and **YAML** files
- Organize documents into **named collections** and query across multiple collections simultaneously
- Switch between **in-memory** (volatile) and **persistent** storage from the UI
- Ask questions and get **streaming AI responses** rendered in Markdown
- Select any **LLM model** installed in Ollama
- Choose an **analysis role** (legal, financial, technical…) and add custom ones without touching the code
- Extract text from **scanned or vector-path PDFs** via automatic OCR fallback

The application uses:
- **Ollama** — local LLM inference and embeddings
- **Qdrant** — local vector store (in-memory or file-based)
- **FastAPI** — REST API backend
- **Vanilla HTML/JS** — lightweight, zero-dependency frontend

## Project Structure

```
doc-analyzer/
├── src/
│   ├── app.py                          # FastAPI routes + inline HTML/JS frontend
│   ├── config/
│   │   └── prompts.py                  # PromptRegistry: auto-discovers src/prompts/
│   ├── models/
│   │   └── document.py                 # Document dataclass
│   ├── prompts/                        # Role prompt files — add .md to create new roles
│   │   ├── default.md
│   │   ├── financial.md
│   │   ├── legal.md
│   │   ├── technical.md
│   │   ├── travel.md
│   │   └── travel_agent.md
│   ├── services/
│   │   └── document_service.py         # State management and document orchestration
│   ├── utils/
│   │   └── text_splitter.py            # Recursive character text splitter
│   └── processors/
│       ├── base/
│       │   └── document_processor.py   # Abstract base with shared utilities
│       ├── factory.py                  # Extension → processor registry
│       ├── pdf_processor.py            # PDF + OCR fallback
│       ├── word_processor.py
│       ├── text_processor.py
│       ├── rtf_processor.py
│       ├── code_processor.py
│       ├── table_processor.py
│       └── rag_processor.py            # Qdrant + Ollama RAG engine
├── tests/
│   ├── processors/
│   └── unit/
├── data/
│   └── qdrant/                         # Qdrant file storage (PERSIST_VECTORDB=true)
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

> **Qdrant persistence:** With `PERSIST_VECTORDB=false` (default), the vector store is in-memory and resets on every restart. Set to `true` to persist data in `./data/qdrant` (volume-mounted in Docker). You can also switch modes at runtime from the UI toggle without editing `.env`.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Web interface |
| `GET` | `/api/status` | Server status, files, models, roles |
| `POST` | `/api/mode` | Switch between `memory` and `persist` |
| `GET` | `/api/collections` | List collections |
| `POST` | `/api/collections` | Create a collection |
| `DELETE` | `/api/collections/{name}` | Delete a collection |
| `POST` | `/api/upload` | Upload and index a document (form field: `collection`) |
| `DELETE` | `/api/collections/{col}/files/{file}` | Remove a file from a collection |
| `DELETE` | `/api/files` | Clear all documents |
| `POST` | `/api/query` | Query (full response) |
| `POST` | `/api/query/stream` | Query with SSE streaming response |
| `POST` | `/api/prompts/reload` | Reload prompts from disk without restart |

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

> **Dockerfiles** have no extension — rename them (e.g. `Dockerfile.txt`) before uploading. The processor detects Dockerfile content automatically.

## Usage

### Documents and collections

**Memory mode** (default): one implicit collection, files are lost on restart. Ideal for quick sessions.

**Persist mode**: toggle the switch in the header to enable. Documents are saved to disk and survive restarts. You can create named collections to organize documents by topic, project, or type.

1. **Upload documents** — click "Upload Document". In persist mode, select the target collection first.
2. **Query across collections** — use the checkboxes to select which collections to include in the search.
3. **Remove documents** — click `×` next to a file, or "Clear All" to reset everything.

### Chat

- Type your question and press **Enter** (Shift+Enter for newline)
- Responses stream progressively and render **Markdown** (code blocks, lists, bold, etc.)
- Select a **role** to frame the analysis perspective
- Select a **model** — all models installed in Ollama are available

### Custom roles

Add a file to `src/prompts/` and press the `↻` button in the header — no restart needed.

**File format** (`src/prompts/my_role.md`):
```markdown
# My Role Name

Instructions for the model. Describe the persona, focus areas,
and tone. This text is injected into every RAG prompt when
this role is selected.
```

- The `# Heading` becomes the display name in the UI select
- The filename stem (`my_role`) becomes the key sent in API requests
- `default.md` is always listed first; others appear in alphabetical order

## Architecture

### Service layer (`document_service.py`)
Centralises all document state and orchestration. FastAPI routes are thin HTTP adapters that delegate to the service — no global state, no business logic in route handlers.

### RAG Processor (`rag_processor.py`)
- Accepts injected Ollama and Qdrant clients (Dependency Injection) — fully testable without environment setup
- Embeds chunks using `mxbai-embed-large` via `ollama.Client.embed()`
- Stores vectors in Qdrant (cosine distance, 1024 dimensions)
- On query: embeds the question, retrieves top-4 chunks per collection, merges and re-ranks globally, generates the answer via `ollama.Client.chat(stream=True)`
- Automatic retry with progressive truncation (up to 12 attempts at 80% each) when a chunk exceeds the embedding model's context length

### Prompt Registry (`config/prompts.py`)
Auto-discovers `*.md` files from `src/prompts/`, parses display name from the first `# Heading`, exposes `reload()` for hot-reload without restart.

### PDF Processor (`pdf_processor.py`)
Extracts text with PyMuPDF (`fitz`). If a page returns no text (scanned PDF or vector-path text), falls back to OCR: renders at 300 DPI and runs `pytesseract.image_to_string()`.

### Factory (`factory.py`)
Extension → processor class registry (`dict`). Adding support for a new format is one line. Falls back to `CodeProcessor` for 40+ code extensions and Dockerfiles without extension.

## Running Tests

```bash
# With Docker
docker compose -f docker-compose.test.yml up --abort-on-container-exit

# Locally
pip install -r requirements-dev.txt
pytest
```

92 tests covering: RAG processor, document service, prompt registry, all processors, API endpoints.

## Troubleshooting

### Ollama not reachable
- Local dev: verify `OLLAMA_HOST=localhost` and Ollama is running (`ollama list`)
- Docker: `OLLAMA_HOST` is automatically overridden to `host.docker.internal`
- Check: `curl http://localhost:11434/api/tags`

### "No content extracted" on PDF upload
The PDF likely contains only scanned images or vector-drawn text. Make sure `tesseract-ocr` is installed (included in the Docker image). For local dev:
```bash
brew install tesseract          # macOS
sudo apt install tesseract-ocr  # Ubuntu/Debian
pip install pytesseract Pillow
```

### "Input length exceeds context length" during upload
Handled automatically — the processor retries with progressively shorter chunks. If it persists, reduce `CHUNK_SIZE` in `.env`.

### Qdrant data lost after restart
Switch to Persist mode in the UI toggle, or set `PERSIST_VECTORDB=true` in `.env`. The `./data` directory is volume-mounted in Docker.

### Docker container management
```bash
docker compose up -d            # start
docker compose up --build -d    # rebuild and start
docker compose restart          # restart without rebuild
docker compose logs -f          # follow logs
docker stop doc-analyzer-web-1  # stop the container
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit your changes: `git commit -m 'Add feature'`
4. Push and open a Pull Request

Follow PEP 8, add tests for new features, update docs.

## License

MIT License — see LICENSE file for details.
