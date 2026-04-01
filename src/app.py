from dotenv import load_dotenv
load_dotenv()

import json
import os
import tempfile
from typing import List, Optional

import secrets

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pathlib import Path

from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from src.config.prompts import registry as prompt_registry
from src.processors.factory import get_processor
from src.processors.rag_processor import DEFAULT_COLLECTION, RAGProcessor
from src.processors.web_processor import WebProcessor
from src.services.document_service import DocumentService
from src.services.note_service import NoteService


# ── Bootstrap ─────────────────────────────────────────────────────────────────

app = FastAPI(title="DocAnalyzer")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_processor = RAGProcessor.from_env()
document_service = DocumentService(rag_processor)
note_service = NoteService()

default_model = os.getenv("LLM_MODEL")

# ── Auth ──────────────────────────────────────────────────────────────────────

_security = HTTPBasic()
_APP_USERNAME = os.getenv("APP_USERNAME", "admin")
_APP_PASSWORD = os.getenv("APP_PASSWORD", "changeme")


def require_auth(credentials: HTTPBasicCredentials = Depends(_security)):
    ok = secrets.compare_digest(credentials.username, _APP_USERNAME) and \
         secrets.compare_digest(credentials.password, _APP_PASSWORD)
    if not ok:
        raise HTTPException(
            401, "Unauthorized", headers={"WWW-Authenticate": "Basic"}
        )


def _get_available_models() -> List[str]:
    """Fetch models from Ollama and ensure the default model is first."""
    models = rag_processor.get_available_models()
    if default_model in models:
        models.remove(default_model)
    return [default_model, *models] if default_model else models


# ── Status ────────────────────────────────────────────────────────────────────

@app.get("/api/status")
def status(_: None = Depends(require_auth)):
    return {
        "mode": rag_processor.mode,
        "collections": rag_processor.get_collections(),
        "files_map": {
            col: list(files.keys())
            for col, files in document_service.files_map.items()
        },
        "models": _get_available_models(),
        "default_model": default_model,
        "roles": prompt_registry.as_api_list(),
    }


# ── Storage mode ──────────────────────────────────────────────────────────────

class StorageModeRequest(BaseModel):
    mode: str


@app.post("/api/mode")
def set_mode(req: StorageModeRequest, _: None = Depends(require_auth)):
    try:
        document_service.switch_mode(req.mode)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return {
        "mode": rag_processor.mode,
        "collections": rag_processor.get_collections(),
        "files_map": {
            col: list(files.keys())
            for col, files in document_service.files_map.items()
        },
    }


# ── Collections ───────────────────────────────────────────────────────────────

@app.get("/api/collections")
def get_collections(_: None = Depends(require_auth)):
    return {"collections": rag_processor.get_collections()}


class CollectionRequest(BaseModel):
    name: str


@app.post("/api/collections")
def create_collection(req: CollectionRequest, _: None = Depends(require_auth)):
    name = req.name.strip()
    if not name:
        raise HTTPException(400, "Collection name cannot be empty")
    document_service.create_collection(name)
    return {"created": name, "collections": rag_processor.get_collections()}


@app.delete("/api/collections/{name}")
def delete_collection(name: str, _: None = Depends(require_auth)):
    document_service.delete_collection(name)
    return {"deleted": name, "collections": rag_processor.get_collections()}


# ── Prompts ───────────────────────────────────────────────────────────────────

@app.post("/api/prompts/reload")
def reload_prompts(_: None = Depends(require_auth)):
    """Re-scan src/prompts/ and return the updated role list. No restart needed."""
    prompt_registry.reload()
    return {"roles": prompt_registry.as_api_list()}


# ── Import from URL ───────────────────────────────────────────────────────────

class ImportRequest(BaseModel):
    url: str
    collection: str = DEFAULT_COLLECTION


@app.post("/api/import")
def import_url(req: ImportRequest, _: None = Depends(require_auth)):
    url = req.url.strip()
    if not url:
        raise HTTPException(400, "URL cannot be empty")

    # Use the URL as the document name for deduplication
    name = url
    if document_service.is_duplicate(name, req.collection):
        raise HTTPException(400, f"'{url}' is already in collection '{req.collection}'")

    try:
        processor = WebProcessor()
        chunks = processor.process_url(url)

        if not chunks:
            raise HTTPException(422, f"No content extracted from '{url}'")

        for chunk in chunks:
            chunk.metadata["file_name"] = name

        n_chunks = document_service.add_file(name, chunks, req.collection)
        return {"url": url, "chunks": n_chunks, "collection": req.collection}

    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    except Exception as exc:
        raise HTTPException(500, f"Error importing '{url}': {exc}")


# ── Upload ────────────────────────────────────────────────────────────────────

@app.post("/api/upload")
async def upload_file(
    file: UploadFile = File(...),
    collection: str = Form(DEFAULT_COLLECTION),
    _: None = Depends(require_auth),
):
    name = file.filename
    if document_service.is_duplicate(name, collection):
        raise HTTPException(400, f"'{name}' is already in collection '{collection}'")

    suffix = os.path.splitext(name)[1]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        tmp.write(await file.read())
        tmp.close()

        processor = get_processor(tmp.name)
        chunks = processor.process(tmp.name)

        if not chunks:
            raise HTTPException(422, f"No content extracted from '{name}'")

        for chunk in chunks:
            chunk.metadata["file_name"] = name

        n_chunks = document_service.add_file(name, chunks, collection)
        return {"file": name, "chunks": n_chunks, "collection": collection}

    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    except Exception as exc:
        raise HTTPException(500, f"Error processing '{name}': {exc}")
    finally:
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)


# ── Remove file ───────────────────────────────────────────────────────────────

@app.delete("/api/collections/{collection}/files/{file_name}")
def remove_file(collection: str, file_name: str, _: None = Depends(require_auth)):
    ok = document_service.remove_file(file_name, collection)
    if not ok:
        raise HTTPException(404, f"'{file_name}' not found in collection '{collection}'")
    return {"removed": file_name, "collection": collection}


# ── Clear all ─────────────────────────────────────────────────────────────────

@app.delete("/api/files")
def clear_all(_: None = Depends(require_auth)):
    document_service.clear_all()
    return {"cleared": True}


# ── Query ─────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    role: str = "default"
    model: Optional[str] = None
    collections: List[str] = []


@app.post("/api/query")
def query(req: QueryRequest, _: None = Depends(require_auth)):
    if not req.question.strip():
        raise HTTPException(400, "Empty question")
    if not document_service.has_documents():
        raise HTTPException(400, "No documents in context")
    try:
        cols = document_service.resolve_query_collections(req.collections)
        answer = rag_processor.query(req.question.strip(), cols, req.role, req.model)
        return {"answer": answer}
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    except Exception as exc:
        raise HTTPException(500, str(exc))


@app.post("/api/query/stream")
def query_stream(req: QueryRequest, _: None = Depends(require_auth)):
    if not req.question.strip():
        raise HTTPException(400, "Empty question")
    if not document_service.has_documents():
        raise HTTPException(400, "No documents in context")

    cols = document_service.resolve_query_collections(req.collections)

    def generate():
        try:
            for event in rag_processor.query_stream(
                req.question.strip(), cols, req.role, req.model
            ):
                yield f"data: {json.dumps(event)}\n\n"
        except ValueError as exc:
            yield f"data: {json.dumps({'type': 'error', 'error': str(exc)})}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'error': str(exc)})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Notes ─────────────────────────────────────────────────────────────────────

class NoteRequest(BaseModel):
    question: str
    answer: str
    sources: List[dict] = []
    collection: str = DEFAULT_COLLECTION
    role: str = "default"


@app.post("/api/notes")
def save_note(req: NoteRequest, _: None = Depends(require_auth)):
    from src.models.note import Note
    note = note_service.save(Note(
        question=req.question,
        answer=req.answer,
        sources=req.sources,
        collection=req.collection,
        role=req.role,
    ))
    return {"id": note.id, "title": note.title, "created_at": note.created_at}


@app.get("/api/notes")
def list_notes(_: None = Depends(require_auth)):
    notes = note_service.list()
    return {"notes": [
        {"id": n.id, "title": n.title, "question": n.question,
         "answer": n.answer, "sources": n.sources,
         "collection": n.collection, "role": n.role, "created_at": n.created_at}
        for n in notes
    ]}


@app.get("/api/notes/{note_id}")
def get_note(note_id: int, _: None = Depends(require_auth)):
    note = note_service.get(note_id)
    if not note:
        raise HTTPException(404, f"Note {note_id} not found")
    return {"id": note.id, "title": note.title, "question": note.question,
            "answer": note.answer, "sources": note.sources,
            "collection": note.collection, "role": note.role,
            "created_at": note.created_at}


@app.delete("/api/notes/{note_id}")
def delete_note(note_id: int, _: None = Depends(require_auth)):
    if not note_service.delete(note_id):
        raise HTTPException(404, f"Note {note_id} not found")
    return {"deleted": note_id}


# ── Frontend ───────────────────────────────────────────────────────────────────

_TEMPLATE = Path(__file__).parent / "templates" / "index.html"

@app.get("/", response_class=FileResponse)
def index():
    return FileResponse(_TEMPLATE)

