# Changelog

## [0.5.0] - 2026-03-31

Refactoring architetturale completo e sistema di prompt esterno con auto-discovery.

### Refactoring architetturale

#### Service layer: `DocumentService`
- Estratto `DocumentService` (`src/services/document_service.py`) che centralizza tutto lo stato e la business logic dei documenti
- Eliminato lo stato globale mutabile `processed_files_map` da `app.py` — sostituito con `document_service._files_map` incapsulato
- Rimossi `global` keyword, helper function libere (`_has_any_files`, `_resolve_collections`, `_all_files_flat`) e l'import locale `from qdrant_client import QdrantClient` dentro `clear_all()`
- Le route FastAPI sono ora thin HTTP adapters che delegano al service

#### Dependency Injection in `RAGProcessor`
- Il costruttore accetta `ollama_client` e `qdrant_client` iniettati — zero coupling alle env var
- Aggiunto `RAGProcessor.from_env()` classmethod per l'uso in produzione
- Aggiunto `reset()` method che incapsula la logica di reset senza esporre `qdrant` all'esterno
- Estratto `_embed_single()` con bound esplicito (`_MAX_TRUNCATION_ATTEMPTS = 12`) e `_TRUNCATION_FACTOR = 0.8` come costanti documentate — eliminato il magic number e il potenziale loop infinito
- I test non richiedono più `monkeypatch` di env var o `patch` di client: si passano `MagicMock()` direttamente al costruttore

#### `ProcessorFactory` → registry dict
- Eliminata la classe statica con if/elif chain (OCP violation)
- Introdotto `_PROCESSOR_MAP: dict[str, type[DocumentProcessor]]` — aggiungere un formato è una riga
- Aggiunta funzione `get_processor()` a livello di modulo; `ProcessorFactory` rimane come facade per backward compatibility

#### `DocumentProcessor` base class
- Aggiunto `_temp_path()` context manager condiviso per la gestione dei file temporanei
- Aggiunte costanti `DEFAULT_CHUNK_SIZE` e `DEFAULT_CHUNK_OVERLAP` lette da env **una sola volta** — eliminate le 5 chiamate duplicate a `os.getenv()` nei processori

#### Processori: interfaccia semplificata
- `WordProcessor`, `TextProcessor`, `RtfProcessor` — rimossa la gestione di file-like objects: `process()` accetta solo `str` path (app.py gestisce già i temp file prima di chiamare i processori)
- Fix bug `WordProcessor`: eliminato `if 'tmp_file' in locals()` in finally block (unreliable) — sostituito con pattern corretto
- Import warning `\*` nel raw string HTML risolto con prefisso `r"""`

#### Test: +30 nuovi test, 92 totali
- `tests/unit/test_document_service.py` — **18 nuovi test** su `DocumentService`: duplicate detection, remove fallback su failure RAG, mode switch, collection lifecycle, resolve_query_collections
- `tests/unit/test_prompt_registry.py` — **12 nuovi test** su `PromptRegistry`: parsing file con/senza heading, ordinamento, hot-reload, file non-.md ignorati, directory mancante
- Fixture `test_rag_processor.py` e `test_rag_processor_remove.py` aggiornate per DI diretta (no `patch`, no `monkeypatch`)

### Sistema prompt esterno con auto-discovery

#### `src/prompts/` — cartella prompt
- I ruoli vivono ora come file `.md` indipendenti in `src/prompts/`
- Aggiungere un nuovo ruolo = creare un file, senza toccare il codice
- `default` sempre primo nella select, gli altri in ordine alfabetico

#### Formato file
```markdown
# Nome Visualizzato nella Select

Testo del prompt che il modello riceve come istruzione di ruolo...
```
- La prima riga `# Heading` diventa il nome display nella UI
- Il filename stem diventa la chiave API (es. `legal.md` → `"legal"`)
- Se manca l'heading, il nome viene ricavato dal filename (`my_role.md` → `"My Role"`)

#### `PromptRegistry` (`src/config/prompts.py`)
- Classe con auto-discovery via `glob("*.md")` sulla cartella `src/prompts/`
- `reload()` — ri-scansiona da disco senza restart del server
- `get_prompt(role)` — restituisce il testo, `ValueError` con messaggio chiaro se il ruolo non esiste
- `as_api_list()` — serializza come `[{key, name}]` per il frontend
- `ROLE_PROMPTS` mantenuto come alias backward-compatible

#### Nuovi endpoint API

| Metodo | Path | Descrizione |
|--------|------|-------------|
| `POST` | `/api/prompts/reload` | Ricarica i prompt da disco, aggiorna la select |

#### Frontend aggiornato
- Select ruoli mostra il **nome display** (es. "Legal Analyst") invece della chiave grezza
- Pulsante `↻` nell'header per ricaricare i prompt senza restart
- `/api/status` restituisce `roles` come `[{key, name}]` invece di `[string]`

### File aggiunti
- `src/services/__init__.py`
- `src/services/document_service.py` — service layer per gestione documenti
- `src/prompts/default.md`
- `src/prompts/legal.md`
- `src/prompts/financial.md`
- `src/prompts/technical.md`
- `src/prompts/travel.md`
- `src/prompts/travel_agent.md`
- `tests/unit/test_document_service.py`
- `tests/unit/test_prompt_registry.py`

### File modificati
- `src/config/prompts.py` — `PromptRegistry` con auto-discovery, `RoleConfig` dataclass, backward-compat alias
- `src/processors/rag_processor.py` — DI, `from_env()`, `reset()`, `_embed_single()`, usa `prompt_registry`
- `src/processors/factory.py` — registry dict, funzione `get_processor()`, facade backward-compat
- `src/processors/base/document_processor.py` — `_temp_path()`, `DEFAULT_CHUNK_SIZE/OVERLAP`
- `src/processors/word_processor.py` — semplificato, fix bug `locals()`, usa costanti base
- `src/processors/text_processor.py` — semplificato, usa costanti base
- `src/processors/rtf_processor.py` — semplificato, usa costanti base
- `src/app.py` — usa `DocumentService`, `get_processor()`, `prompt_registry`; endpoint reload prompts

---

## [0.4.0] - 2026-03-31

Introdotta la gestione multi-collection con toggle Memory/Persist direttamente dall'interfaccia. In modalità Persist le collection vengono salvate su disco e sopravvivono ai restart; in modalità Memory tutto è in-memory e volatile.

### Cambiamenti principali

#### Storage mode: Memory vs Persist (da UI)
- **Toggle Memory/Persist** nell'header — nessuna modifica a `.env` necessaria
- **Memory mode**: Qdrant in-memory (`:memory:`), comportamento identico alle versioni precedenti; tutti i dati vengono persi al restart
- **Persist mode**: Qdrant file-based su `QDRANT_DB_PATH`; i dati sopravvivono ai restart del container
- Al passaggio a Persist, il `files_map` viene ricostruito automaticamente dai payload Qdrant (`rebuild_files_map()`)

#### Gestione multi-collection
- **Creazione collection** dalla sidebar con form inline
- **Eliminazione collection** con conferma (rimuove anche tutti i documenti indicizzati)
- **Collection attiva** per l'upload: click sul nome per selezionarla come target
- **Checkbox per la query**: ogni collection ha un checkbox — la query viene eseguita solo sulle collection selezionate
- I risultati di più collection vengono uniti e riordinati per score (cosine), mantenendo i top 4 globali
- **Memory mode**: un'unica collection implicita `default`, sidebar semplificata

#### Nuovi endpoint API

| Metodo | Path | Descrizione |
|--------|------|-------------|
| `POST` | `/api/mode` | Switcha tra `memory` e `persist` |
| `GET` | `/api/collections` | Lista delle collection esistenti |
| `POST` | `/api/collections` | Crea una nuova collection |
| `DELETE` | `/api/collections/{name}` | Elimina una collection e i suoi documenti |
| `DELETE` | `/api/collections/{col}/files/{file}` | Rimuove un file da una collection specifica |

#### Endpoint modificati
- `POST /api/upload` — accetta il campo form `collection` (default: `"default"`)
- `POST /api/query` e `/api/query/stream` — accettano `collections: List[str]`; se vuota, usa tutte le collection caricate
- `GET /api/status` — restituisce `mode`, `collections`, `files_map` (al posto di `files`)

---

## [0.3.0] - 2026-03-31

Interfaccia completamente riscritta: rimosso Gradio, sostituito con frontend HTML/JS puro su FastAPI. Aggiunti OCR per PDF vettoriali/scansionati, streaming delle risposte LLM, e vari fix di stabilità.

### Cambiamenti principali

#### UI: da Gradio a HTML/JS puro
- **Rimosso Gradio** completamente (addio SSE drop, `BodyStreamBuffer aborted`, reset di pagina durante l'inference)
- Nuova interfaccia **vanilla HTML/JS** servita direttamente da FastAPI come `HTMLResponse`
- Layout sidebar + chat, dark theme, completamente responsive (mobile/desktop)
- Textarea auto-resize, invio con `Enter` (Shift+Enter per andare a capo)
- Send button disabilitato automaticamente quando non ci sono documenti
- Nessuna dipendenza frontend esterna (zero npm, zero CDN)

#### Streaming risposte LLM
- **Nuovo endpoint `POST /api/query/stream`** con `text/event-stream` (SSE)
- Testo progressivo token per token, cursore lampeggiante in attesa del primo token
- Rendering markdown in tempo reale durante lo streaming

#### OCR per PDF vettoriali/scansionati
- Fallback automatico con `pytesseract` + `PIL` a 300 DPI per PDF senza testo estraibile

#### Embedding: gestione context length
- Retry con troncamento progressivo (80%, fino a 12 tentativi) quando il chunk supera il context length del modello di embedding

---

## [0.2.0] - 2026-03-30

Modernizzazione completa dello stack: rimosso LangChain, migrato a Qdrant, aggiornato Ollama all'API ufficiale.

- Sostituito ChromaDB con Qdrant (cosine distance, 1024 dimensioni)
- Introdotto `EMBEDDING_MODEL` separato dal modello LLM (`mxbai-embed-large:latest`)
- Aggiornato client Ollama da v0.1.6 a >=0.4.0
- Eliminato LangChain: `Document` dataclass custom, `RecursiveCharacterTextSplitter` custom
- PDF processor riscritto con PyMuPDF diretto

---

## [0.1.0] - 2025

Release iniziale con supporto RAG locale basato su LangChain, ChromaDB e Ollama.

- Supporto documenti: PDF, DOCX, DOC, TXT, RTF
- Supporto codice: 30+ linguaggi di programmazione
- Supporto dati tabulari: Excel, CSV, ODS, JSON
- Analisi basata su ruoli: default, legal, financial, travel, technical
- Interfaccia Gradio con chat e gestione documenti
- Containerizzazione Docker
