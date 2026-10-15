# Doc Analyzer

A web application that analyzes documents and code files using local large language models via Ollama and RAG (Retrieval-Augmented Generation). No cloud, no API keys — everything runs on your machine.

## Overview

Doc Analyzer enables you to:
- Upload and analyze **PDF, DOCX, DOC, TXT, RTF** documents
- Process **30+ code file formats** (Python, JS, Java, Go, Rust, and more)
- Handle **tabular data** (Excel, CSV, ODS, JSON)
- Process **Markdown** and **YAML** files
- **Import web pages, API documentation, and YouTube video transcripts** directly from a URL — NotebookLM-style
- Organize documents into **named collections** and query across multiple collections simultaneously
- Switch between **in-memory** (volatile) and **persistent** storage from the UI
- Ask questions and get **streaming AI responses** with **source citations** rendered in Markdown
- **Deep mode** reads the selected documents in full (map → reduce) for summaries, exhaustive lists and comparisons that top-k retrieval misses; **Auto mode** picks deep or standard from the question
- Select any **LLM model** installed in Ollama
- Choose an **analysis role** (legal, financial, technical…) and add custom ones without touching the code
- Extract text from **scanned or vector-path PDFs** via automatic OCR fallback
- Save responses as **notes** (SQLite) and review them in the built-in notebook panel
- **Export** the conversation as a Markdown file
- **Drag & drop** files directly onto the upload zone
- Access the app with **HTTP Basic Auth** — credentials set in `.env`

The application uses:
- **Ollama** — local LLM inference and embeddings
- **Qdrant** — local vector store (in-memory or file-based)
- **FastAPI** — REST API backend
- **Vanilla HTML/JS** — lightweight, zero-dependency frontend

## Project Structure

```
doc-analyzer/
├── src/
│   ├── app.py                          # FastAPI routes (239 lines)
│   ├── templates/
│   │   └── index.html                  # Frontend — HTML/CSS/JS
│   ├── config/
│   │   └── prompts.py                  # PromptRegistry: auto-discovers src/prompts/
│   ├── models/
│   │   ├── document.py                 # Document dataclass
│   │   └── note.py                     # Note dataclass
│   ├── prompts/                        # Role prompt files — add .md to create new roles
│   │   ├── default.md
│   │   ├── financial.md
│   │   ├── legal.md
│   │   ├── technical.md
│   │   ├── travel.md
│   │   └── travel_agent.md
│   ├── services/
│   │   ├── document_service.py         # State management and document orchestration
│   │   ├── note_service.py             # SQLite-backed notes CRUD
│   │   └── query_router.py             # Auto mode: standard vs deep from the question
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
│       ├── web_processor.py            # Web pages, API docs, YouTube transcripts
│       └── rag_processor.py            # Qdrant + Ollama RAG engine
├── tests/
│   ├── processors/
│   └── unit/
├── data/
│   ├── qdrant/                         # Qdrant file storage (persist mode)
│   └── notes.db                        # SQLite notes database
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
# Auth (HTTP Basic)
APP_USERNAME=admin
APP_PASSWORD=changeme

# LLM model for text generation (any model available in Ollama)
LLM_MODEL=qwen2.5:14b

# Dedicated embedding model
EMBEDDING_MODEL=mxbai-embed-large:latest

# Ollama connection
OLLAMA_HOST=localhost        # use 'localhost' for local dev
OLLAMA_PORT=11434

# Vector store
QDRANT_DB_PATH=./data/qdrant

# Chunks embedded per Ollama call during upload
EMBEDDING_BATCH_SIZE=32

# Chunking
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# Deep mode (optional — defaults shown)
DEEP_BLOCK_CHARS=10000   # characters per block sent to the LLM; keep under the model's num_ctx
DEEP_MAX_BLOCKS=60       # refuse deep queries over more blocks than this
DEEP_MAX_WORKERS=2       # parallel LLM calls; match OLLAMA_NUM_PARALLEL

# Auto mode embedding router (optional — defaults calibrated for mxbai-embed-large)
ROUTER_EMBEDDINGS=true       # false = regex rules only
ROUTER_MIN_SIMILARITY=0.5
ROUTER_MARGIN=0.06
```

> **Docker note:** `docker-compose.yml` automatically overrides `OLLAMA_HOST` to `host.docker.internal` so the container can reach Ollama on the host machine.

> **Qdrant persistence:** The storage mode (memory vs persist) is controlled from the UI toggle in the header. In persist mode, data is saved to `QDRANT_DB_PATH` and survives restarts. The `./data` directory is volume-mounted in Docker.

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
| `POST` | `/api/import` | Import content from a URL (web page, API docs, YouTube) |
| `DELETE` | `/api/collections/{col}/files/{file}` | Remove a file from a collection |
| `DELETE` | `/api/files` | Clear all documents |
| `POST` | `/api/query` | Query (full response). Body: `{question, role, model, collections, mode}` with `mode` = `standard` (default), `deep` or `auto`. The legacy `deep: true` still means `mode: deep` |
| `POST` | `/api/query/stream` | Query with SSE streaming response (same body; deep adds `progress` events, auto adds a `route` event first) |
| `POST` | `/api/prompts/reload` | Reload prompts from disk without restart |
| `GET` | `/api/notes` | List all saved notes |
| `POST` | `/api/notes` | Save a new note |
| `GET` | `/api/notes/{id}` | Get a single note |
| `DELETE` | `/api/notes/{id}` | Delete a note |

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
2. **Import from URL** — paste any URL in the "Import from URL" field and press Enter or click Import:
   - **Web pages / API docs** — text is extracted with [trafilatura](https://trafilatura.readthedocs.io/), removing boilerplate and ads
   - **YouTube videos** — the transcript is fetched automatically (no API key required). Works with both auto-generated and manual subtitles
3. **Query across collections** — use the checkboxes to select which collections to include in the search.
4. **Remove documents** — click `×` next to a file, or "Clear All" to reset everything.

### Chat

- Type your question and press **Enter** (Shift+Enter for newline)
- Responses stream progressively and render **Markdown** (code blocks, lists, bold, etc.)
- Each response shows **source citations** — the documents and pages used to generate the answer
- Select a **role** to frame the analysis perspective
- Select a **model** — all models installed in Ollama are available
- Pick the query mode in the header: **Auto** (default), **Standard** or **Deep** (see below); the choice is remembered per browser

### Deep mode

Standard queries retrieve the 4 chunks closest to the question. That works for targeted questions ("what does clause 5 say?") but cannot answer questions that need the whole document: "list every deadline", "summarise the report", "compare the termination clauses in A and B".

Deep mode reads everything in the selected collections, inspired by [Recursive Language Models](https://arxiv.org/abs/2512.24601) (Zhang, Kraska, Khattab):

1. **Rebuild** — each file is reassembled from its stored chunks in original order and grouped into blocks of `DEEP_BLOCK_CHARS`. Blocks never span two files.
2. **Map** — every block is sent to the LLM with the question, asking for all relevant facts or `NONE`. Calls run in parallel (`DEEP_MAX_WORKERS`).
3. **Reduce** — if the collected notes don't fit in one block, they are merged in batches, recursively (max 4 rounds).
4. **Answer** — the final notes are passed to the LLM with the selected role and the answer streams as usual.

Sources list each file that contributed, with the number of relevant blocks; the percentage is the share of the file's blocks that contained relevant information.

**Trade-offs:** one LLM call per block, so deep mode is much slower than a standard query (on a ~45k-char document with a local 12B model: about 2 minutes versus 24 s, but 5 deadlines found out of 5 versus 1 out of 5). Use it for aggregation questions, not lookups. Queries over more than `DEEP_MAX_BLOCKS` blocks are refused. Narrow the collections or raise the limit.

### Auto mode

Auto mode (the UI default) picks the pipeline from the question in two steps, with no extra LLM call:

1. **Regex rules** (Italian and English): deterministic and instant, they catch the common phrasings below.
2. **Embedding router**, only when no rule fires: the question embedding (already computed for retrieval, so no extra cost) is compared with 60 example questions split across categories (`query_router.EXEMPLARS`). The score per category is the mean cosine similarity of its 3 closest examples. The question goes to deep only if the best deep category scores at least `ROUTER_MIN_SIMILARITY` **and** beats the targeted examples by `ROUTER_MARGIN`. This catches paraphrases ("what deadlines are there?") and other languages.

| Route | Triggered by | Example |
|-------|--------------|---------|
| Deep · summary | riassumi, sintesi, panoramica, summarize, overview | "Riassumi il report" |
| Deep · count | quante volte, how many times, count | "Quante volte compare *penale*?" |
| Deep · comparison | confronta, differenze, compare *X* with *Y*, versus | "Confronta le clausole di recesso" |
| Deep · exhaustive list | elenca, lista, list every, *tutte le scadenze/clausole/date…* | "Elenca tutte le scadenze" |
| Deep · whole document | in tutto il documento, entire document | "Quali rischi in tutto il documento?" |
| Standard | anything else | "Cosa dice l'articolo 5?" |

Each answer shows the chosen route and its reason: *Deep · auto: exhaustive list* for a rule, *Deep · auto: similar to list* for the embedding router. If the router is wrong, pick Standard or Deep explicitly. When deep is chosen but the documents exceed `DEEP_MAX_BLOCKS`, auto falls back to standard instead of failing.

**Calibration.** The defaults favour precision: a wrong deep route costs minutes, a missed one only an incomplete answer. On 36 held-out questions (not among the examples, all left on standard by the regex rules) with `mxbai-embed-large`, margin 0.06 sends 10 of 18 aggregation questions to deep and none of the 18 targeted ones. Similarities depend on the embedding model, so after changing `EMBEDDING_MODEL` run:

```bash
python scripts/calibrate_router.py
```

It prints the scores and suggests `ROUTER_MARGIN` and `ROUTER_MIN_SIMILARITY`. Rules and examples live in `src/services/query_router.py`, covered by `tests/unit/test_query_router.py`.

### Notes

- Click **💾 Save** (appears on hover over any bot response) to save a Q&A pair as a note
- Open the **📓 Notes** panel from the header to browse saved notes
- Notes persist in `./data/notes.db` across restarts and are independent from the vector store
- Click a note to expand the full question and answer

### Chat history and export

- The conversation is automatically saved to `localStorage` and restored on page refresh
- Click **⬇ Export** in the header to download the full conversation as a `.md` file
- **Clear All** resets both the vector store and the chat history

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
- Embeds chunks using `mxbai-embed-large` via `ollama.Client.embed()`, in batches of `EMBEDDING_BATCH_SIZE`; a failing batch falls back to per-chunk embedding with truncation retry
- Stores vectors in Qdrant (cosine distance, configurable dimensions via `EMBEDDING_VECTOR_SIZE`)
- On query: embeds the question, retrieves top-4 chunks per collection, merges and re-ranks globally, streams the answer via `ollama.Client.chat(stream=True)`
- `query_stream()` emits structured events: `{"type":"sources", "sources":[...]}` first, then `{"type":"text", "text":"token"}` — the frontend renders citations and text separately
- `query_stream_auto()` routes via `query_router.route_question()` and, when no rule fires, `EmbeddingRouter.classify()` on the question embedding (then reused for retrieval); emits `{"type":"route", "mode", "reason"}` and delegates to the standard or deep stream
- `query_stream_deep()` implements deep mode: rebuilds files from Qdrant payloads (ordered by `chunk_index`, stored at indexing time; older chunks fall back to page order), maps every block, reduces recursively and streams the answer. It also emits `{"type":"progress", "stage":"map"|"reduce", "done":n, "total":m}`
- Automatic retry with progressive truncation (configurable via `EMBEDDING_TRUNCATION_FACTOR` and `EMBEDDING_MAX_TRUNCATION_ATTEMPTS`) when a chunk exceeds the embedding model's context length

### Note Service (`services/note_service.py`)
- Persists Q&A pairs with sources, collection, and role to `./data/notes.db` (SQLite, no extra dependencies)
- Auto-creates the DB and schema on first run
- Notes are independent from Qdrant — survive `Clear All` and storage mode switches

### Prompt Registry (`config/prompts.py`)
Auto-discovers `*.md` files from `src/prompts/`, parses display name from the first `# Heading`, exposes `reload()` for hot-reload without restart.

### PDF Processor (`pdf_processor.py`)
Extracts text with PyMuPDF (`fitz`). If a page returns no text (scanned PDF or vector-path text), falls back to OCR: renders at 300 DPI and runs `pytesseract.image_to_string()`.

### Web Processor (`web_processor.py`)
Handles URL-based import. Detects YouTube URLs via regex and fetches the transcript using `youtube-transcript-api` (v1.x instance API, no API key required, falls back across `it`/`en`/`en-US`/`en-GB`). For all other URLs, fetches and cleans the page with `trafilatura`. Both paths produce `Document` chunks fed through the standard text splitter.

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

289 tests covering: RAG processor (including sources events, truncation logic and deep mode), document service, note service, prompt registry, all processors (including WebProcessor), API endpoints (upload, URL import, collections, streaming, notes).

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

### Warning: "APP_PASSWORD is the default 'changeme'"
Logged at startup when the password was not changed. Set `APP_PASSWORD` in `.env` before exposing the app beyond localhost.

### Deep mode is slow or answers are incomplete
- Each block is one LLM call: reduce the selected collections, or raise `DEEP_MAX_WORKERS` together with `OLLAMA_NUM_PARALLEL` on the Ollama side
- If the model ignores parts of a block, its context window is too small: lower `DEEP_BLOCK_CHARS` or raise the model's `num_ctx`
- Documents indexed before deep mode have no `chunk_index`; they are rebuilt in page order, which is exact for PDFs but approximate for other formats. Re-upload them for exact ordering

### Qdrant data lost after restart
Switch to Persist mode using the toggle in the UI header. The `./data` directory is volume-mounted in Docker.

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
