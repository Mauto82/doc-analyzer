# Changelog

## [0.3.0] - 2026-03-31

Interfaccia completamente riscritta: rimosso Gradio, sostituito con frontend HTML/JS puro su FastAPI. Aggiunti OCR per PDF vettoriali/scansionati, streaming delle risposte LLM, e vari fix di stabilità.

### Cambiamenti principali

#### UI: da Gradio a HTML/JS puro
- **Rimosso Gradio** completamente (addio SSE drop, `BodyStreamBuffer aborted`, reset di pagina durante l'inference)
- Nuova interfaccia **vanilla HTML/JS** servita direttamente da FastAPI come `HTMLResponse`
- Layout sidebar + chat, dark theme, completamente responsive (mobile/desktop)
- Sidebar con lista documenti caricati, pulsante upload e clear all
- Chat con bubble differenziate utente/bot
- Textarea auto-resize, invio con `Enter` (Shift+Enter per andare a capo)
- Send button disabilitato automaticamente quando non ci sono documenti
- Nessuna dipendenza frontend esterna (zero npm, zero CDN)

#### Streaming risposte LLM
- **Nuovo endpoint `POST /api/query/stream`** che restituisce `StreamingResponse` con `text/event-stream`
- Il testo appare progressivamente token per token mentre il modello genera
- **Cursore lampeggiante** `|` mostrato in attesa del primo token
- Ollama `stream=True` con generatore Python lato server
- Header `X-Accel-Buffering: no` per evitare buffering con proxy/nginx
- Gestione errori inline: gli errori durante lo stream vengono mostrati nel bubble

#### OCR per PDF vettoriali/scansionati
- **Fallback OCR automatico** per PDF che non contengono testo estraibile (es. PDF generati da Quartz/macOS con testo come vector path)
- Rendering pagine a 300 DPI con PyMuPDF → `PIL.Image` → `pytesseract.image_to_string()`
- Dipendenze opzionali: se `pytesseract` o `Pillow` non sono installati, il fallback è disabilitato silenziosamente
- Aggiunti `pytesseract>=0.3.10` e `Pillow>=10.0.0` a `requirements.txt`

#### Embedding: gestione context length
- **Retry con troncamento progressivo** in `_get_embeddings`: se il testo supera il limite di token del modello (es. testi lunghi in italiano con molti caratteri accentati su `mxbai-embed-large`), il chunk viene ridotto all'80% e ritentato automaticamente fino a quando non rientra nel contesto

#### Fix Docker e configurazione
- **Healthcheck Docker** aggiornato: sostituito `curl` (non disponibile in `python:3.11-slim`) con `urllib.request` Python nativo
- **`OLLAMA_HOST`**: `.env` ora usa `localhost` per sviluppo locale; `docker-compose.yml` sovrascrive con `host.docker.internal` solo dentro il container
- **`load_dotenv()`** aggiunto correttamente all'inizio di `src/app.py` (fix "LLM_MODEL not set" in sviluppo locale)

### Dipendenze

| Pacchetto | Prima | Dopo |
|-----------|-------|------|
| gradio | >=4.44.1 | **rimosso** |
| pytesseract | - | >=0.3.10 |
| Pillow | - | >=10.0.0 |

### Nuovi endpoint API

| Metodo | Path | Descrizione |
|--------|------|-------------|
| `POST` | `/api/query/stream` | Query con risposta in streaming SSE |

### File modificati
- `src/app.py` — rimosso Gradio, nuovo frontend HTML/JS, endpoint streaming
- `src/processors/rag_processor.py` — metodo `query_stream()`, retry troncamento embedding
- `src/processors/pdf_processor.py` — fallback OCR con pytesseract
- `requirements.txt` — rimosso gradio, aggiunti pytesseract e Pillow
- `docker-compose.yml` — healthcheck con urllib, OLLAMA_HOST override

---

## [0.2.0] - 2026-03-30

Modernizzazione completa dello stack tecnologico: rimossa la dipendenza da LangChain, migrato il vector store a Qdrant, aggiornata l'integrazione con Ollama all'API ufficiale, e rinnovata l'interfaccia utente.

### Cambiamenti principali

#### Vector Store: da ChromaDB a Qdrant
- **Sostituito ChromaDB** con [Qdrant](https://qdrant.tech/) come vector database
- Supporto per storage file-based (`QDRANT_DB_PATH`) o in-memory
- Ricerca semantica tramite distanza coseno su vettori a 1024 dimensioni
- Gestione collezioni con creazione, eliminazione e pulizia automatica

#### Embedding: modello dedicato
- **Introdotto `EMBEDDING_MODEL`** come variabile d'ambiente separata dal modello LLM
- Default: `mxbai-embed-large:latest` (1024 dimensioni) al posto di usare il modello LLM per gli embedding
- Embedding batch tramite la nuova API `ollama.embed()`
- Separazione netta tra modello di embedding e modello di generazione

#### Ollama: API ufficiale aggiornata
- **Aggiornato il client Ollama** da v0.1.6 a >=0.4.0 (API ufficiale)
- Utilizzo di `ollama.Client.chat()` con response tipizzate (`ChatResponse`)
- Utilizzo di `ollama.Client.embed()` per embedding batch (`EmbedResponse`)
- Utilizzo di `ollama.Client.list()` con oggetti `Model` tipizzati
- Client inizializzato una sola volta e riutilizzato

#### Rimosso LangChain
- **Eliminata completamente la dipendenza da LangChain** (`langchain`, `langchain-community`)
- Creato `Document` dataclass custom (`src/models/document.py`) come sostituto di `langchain.schema.Document`
- Creato `RecursiveCharacterTextSplitter` custom (`src/utils/text_splitter.py`) con la stessa logica di splitting ricorsivo
- PDF processor riscritto per usare `fitz` (PyMuPDF) direttamente al posto di `PyMuPDFLoader`

### Bug fix
- Rimosso decoratore `@classmethod` duplicato in `code_processor.py`
- Import `textract` reso lazy per compatibilita con Python 3.12+
- Corretto reset del valore dropdown dopo rimozione file

### Dipendenze

| Pacchetto | Prima | Dopo |
|-----------|-------|------|
| ollama | 0.1.6 | >=0.4.0 |
| qdrant-client | - | >=1.9.0 |
| langchain | 0.0.350 | **rimosso** |
| langchain-community | 0.0.10 | **rimosso** |
| chromadb | 0.4.22 | **rimosso** |

### Nuove variabili d'ambiente

| Variabile | Default | Descrizione |
|-----------|---------|-------------|
| `EMBEDDING_MODEL` | `mxbai-embed-large:latest` | Modello dedicato per gli embedding |
| `QDRANT_DB_PATH` | `./data/qdrant` | Path storage Qdrant |

### Variabili rimosse

| Variabile | Motivo |
|-----------|--------|
| `CHROMA_DB_PATH` | Sostituita da `QDRANT_DB_PATH` |

### File aggiunti
- `src/models/document.py` — Dataclass Document
- `src/utils/text_splitter.py` — Text splitter custom

### File riscritti
- `src/processors/rag_processor.py` — Qdrant + Ollama moderno
- `src/processors/pdf_processor.py` — PyMuPDF diretto
- `src/app.py` — UI rinnovata

---

## [0.1.0] - 2025

Release iniziale con supporto RAG locale basato su LangChain, ChromaDB e Ollama.

- Supporto documenti: PDF, DOCX, DOC, TXT, RTF
- Supporto codice: 30+ linguaggi di programmazione
- Supporto dati tabulari: Excel, CSV, ODS, JSON
- Analisi basata su ruoli: default, legal, financial, travel, technical
- Interfaccia Gradio con chat e gestione documenti
- Containerizzazione Docker
