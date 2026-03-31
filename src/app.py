from dotenv import load_dotenv
load_dotenv()

import json
import os
import tempfile
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from src.config.prompts import ROLE_PROMPTS
from src.processors.factory import get_processor
from src.processors.rag_processor import DEFAULT_COLLECTION, RAGProcessor
from src.services.document_service import DocumentService


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

default_model = os.getenv("LLM_MODEL")


def _get_available_models() -> List[str]:
    """Fetch models from Ollama and ensure the default model is first."""
    models = rag_processor.get_available_models()
    if default_model in models:
        models.remove(default_model)
    return [default_model, *models] if default_model else models


# ── Status ────────────────────────────────────────────────────────────────────

@app.get("/api/status")
def status():
    return {
        "mode": rag_processor.mode,
        "collections": rag_processor.get_collections(),
        "files_map": {
            col: list(files.keys())
            for col, files in document_service.files_map.items()
        },
        "models": _get_available_models(),
        "default_model": default_model,
        "roles": list(ROLE_PROMPTS.keys()),
    }


# ── Storage mode ──────────────────────────────────────────────────────────────

class StorageModeRequest(BaseModel):
    mode: str


@app.post("/api/mode")
def set_mode(req: StorageModeRequest):
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
def get_collections():
    return {"collections": rag_processor.get_collections()}


class CollectionRequest(BaseModel):
    name: str


@app.post("/api/collections")
def create_collection(req: CollectionRequest):
    name = req.name.strip()
    if not name:
        raise HTTPException(400, "Collection name cannot be empty")
    document_service.create_collection(name)
    return {"created": name, "collections": rag_processor.get_collections()}


@app.delete("/api/collections/{name}")
def delete_collection(name: str):
    document_service.delete_collection(name)
    return {"deleted": name, "collections": rag_processor.get_collections()}


# ── Upload ────────────────────────────────────────────────────────────────────

@app.post("/api/upload")
async def upload_file(
    file: UploadFile = File(...),
    collection: str = Form(DEFAULT_COLLECTION),
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
def remove_file(collection: str, file_name: str):
    ok = document_service.remove_file(file_name, collection)
    if not ok:
        raise HTTPException(404, f"'{file_name}' not found in collection '{collection}'")
    return {"removed": file_name, "collection": collection}


# ── Clear all ─────────────────────────────────────────────────────────────────

@app.delete("/api/files")
def clear_all():
    document_service.clear_all()
    return {"cleared": True}


# ── Query ─────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    role: str = "default"
    model: Optional[str] = None
    collections: List[str] = []


@app.post("/api/query")
def query(req: QueryRequest):
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
def query_stream(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(400, "Empty question")
    if not document_service.has_documents():
        raise HTTPException(400, "No documents in context")

    cols = document_service.resolve_query_collections(req.collections)

    def generate():
        try:
            for chunk in rag_processor.query_stream(
                req.question.strip(), cols, req.role, req.model
            ):
                yield f"data: {json.dumps({'text': chunk})}\n\n"
        except ValueError as exc:
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Frontend ───────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_PAGE


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DocAnalyzer</title>
<style>
  :root {
    --bg: #111; --surface: #1a1a1e; --surface2: #222228; --border: #2a2a2e;
    --text: #e4e4e7; --muted: #888; --accent: #3b82f6; --accent2: #6366f1;
    --danger: #ef4444; --success: #22c55e; --radius: 8px;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: var(--bg); color: var(--text); height: 100vh;
         display: flex; flex-direction: column; overflow: hidden; }

  /* ── Header ── */
  header { padding: 10px 20px; border-bottom: 1px solid var(--border);
           display: flex; align-items: center; gap: 16px; flex-shrink: 0;
           background: var(--surface); }
  header h1 { font-size: 17px; font-weight: 700; letter-spacing: -0.3px; }
  .header-right { display: flex; gap: 10px; align-items: center; margin-left: auto; }
  header select { background: var(--bg); color: var(--text); border: 1px solid var(--border);
                  border-radius: var(--radius); padding: 5px 10px; font-size: 13px; }

  /* ── Toggle switch ── */
  .mode-toggle { display: flex; align-items: center; gap: 8px; font-size: 12px; color: var(--muted); }
  .mode-toggle span.active { color: var(--text); font-weight: 600; }
  .switch { position: relative; display: inline-block; width: 42px; height: 22px; }
  .switch input { opacity: 0; width: 0; height: 0; }
  .slider { position: absolute; cursor: pointer; inset: 0; background: var(--border);
            border-radius: 22px; transition: .25s; }
  .slider:before { content: ''; position: absolute; width: 16px; height: 16px;
                   left: 3px; top: 3px; background: #fff; border-radius: 50%; transition: .25s; }
  input:checked + .slider { background: var(--accent2); }
  input:checked + .slider:before { transform: translateX(20px); }

  /* ── Layout ── */
  .app { display: flex; flex: 1; overflow: hidden; }

  /* ── Sidebar ── */
  .sidebar { width: 280px; border-right: 1px solid var(--border); display: flex;
             flex-direction: column; flex-shrink: 0; background: var(--surface); overflow: hidden; }

  .sidebar-section { font-size: 11px; font-weight: 700; color: var(--muted);
                     text-transform: uppercase; letter-spacing: .6px;
                     padding: 10px 14px 6px; }

  .upload-zone { padding: 10px 12px; border-bottom: 1px solid var(--border); }
  .upload-btn { width: 100%; padding: 8px; background: var(--accent); color: #fff;
                border: none; border-radius: var(--radius); cursor: pointer; font-size: 13px;
                font-weight: 500; transition: opacity .15s; }
  .upload-btn:hover:not(:disabled) { opacity: .85; }
  .upload-btn:disabled { opacity: .4; cursor: not-allowed; }
  input[type=file] { display: none; }

  /* Collection selector for upload (persist mode) */
  .upload-target { display: flex; align-items: center; gap: 6px; margin-bottom: 8px;
                   font-size: 12px; color: var(--muted); }
  .upload-target select { flex: 1; background: var(--bg); color: var(--text);
                          border: 1px solid var(--border); border-radius: 6px;
                          padding: 4px 8px; font-size: 12px; }

  .file-list { flex: 1; overflow-y: auto; padding: 6px 8px; }
  .file-item { display: flex; justify-content: space-between; align-items: center;
               padding: 6px 10px; border-radius: 6px; font-size: 12px;
               background: var(--surface2); margin-bottom: 3px; gap: 6px; }
  .file-item span { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .file-item .col-badge { font-size: 10px; color: var(--muted); background: var(--border);
                          border-radius: 4px; padding: 1px 5px; flex-shrink: 0; }
  .icon-btn { background: none; border: none; cursor: pointer; padding: 2px 4px;
              line-height: 1; color: var(--muted); font-size: 15px; flex-shrink: 0; }
  .icon-btn:hover { color: var(--danger); }
  .empty-msg { color: var(--muted); font-size: 12px; padding: 14px; text-align: center; }
  .sidebar-footer { padding: 8px 12px; border-top: 1px solid var(--border); flex-shrink: 0; }
  .clear-btn { width: 100%; padding: 7px; background: transparent; color: var(--danger);
               border: 1px solid var(--danger); border-radius: var(--radius); cursor: pointer;
               font-size: 12px; transition: all .15s; }
  .clear-btn:hover { background: var(--danger); color: #fff; }

  /* Collections (persist mode) */
  .collections-header { display: flex; align-items: center; justify-content: space-between;
                        padding: 10px 14px 6px; }
  .collections-header span { font-size: 11px; font-weight: 700; color: var(--muted);
                              text-transform: uppercase; letter-spacing: .6px; }
  .new-col-btn { background: var(--accent); border: none; color: #fff; border-radius: 5px;
                 width: 22px; height: 22px; font-size: 16px; cursor: pointer; line-height: 1;
                 display: flex; align-items: center; justify-content: center; }
  .new-col-btn:hover { opacity: .85; }

  .col-list { padding: 4px 8px; border-bottom: 1px solid var(--border); max-height: 180px; overflow-y: auto; }
  .col-item { display: flex; align-items: center; gap: 8px; padding: 6px 8px;
              border-radius: 6px; font-size: 13px; cursor: pointer;
              background: var(--surface2); margin-bottom: 3px; }
  .col-item:hover { background: var(--border); }
  .col-item.active { background: color-mix(in srgb, var(--accent) 18%, transparent);
                     border: 1px solid color-mix(in srgb, var(--accent) 40%, transparent); }
  .col-item input[type=checkbox] { accent-color: var(--accent2); width: 14px; height: 14px;
                                    flex-shrink: 0; cursor: pointer; }
  .col-item .col-name { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .col-item .col-count { font-size: 10px; color: var(--muted); flex-shrink: 0; }

  .new-col-form { display: none; padding: 8px 12px; border-bottom: 1px solid var(--border);
                  gap: 6px; flex-direction: column; }
  .new-col-form.visible { display: flex; }
  .new-col-form input { background: var(--bg); color: var(--text); border: 1px solid var(--border);
                        border-radius: 6px; padding: 6px 10px; font-size: 13px; outline: none; }
  .new-col-form input:focus { border-color: var(--accent); }
  .new-col-form .row { display: flex; gap: 6px; }
  .btn-sm { padding: 5px 12px; border: none; border-radius: 6px; cursor: pointer;
            font-size: 12px; font-weight: 500; }
  .btn-primary { background: var(--accent); color: #fff; }
  .btn-ghost { background: var(--border); color: var(--text); }

  .files-section-header { padding: 8px 14px 4px; font-size: 11px; font-weight: 700;
                           color: var(--muted); text-transform: uppercase; letter-spacing: .6px;
                           display: flex; align-items: center; gap: 6px; }
  .files-section-header .in-col { color: var(--accent2); font-weight: 600;
                                   text-transform: none; letter-spacing: 0; font-size: 12px; }

  /* ── Chat ── */
  .chat { flex: 1; display: flex; flex-direction: column; overflow: hidden; }
  .messages { flex: 1; overflow-y: auto; padding: 20px; display: flex;
              flex-direction: column; gap: 14px; }
  .msg { max-width: 80%; padding: 11px 15px; border-radius: 12px; font-size: 14px;
         line-height: 1.65; white-space: pre-wrap; word-wrap: break-word; }
  .msg.user { align-self: flex-end; background: var(--accent); color: #fff;
              border-bottom-right-radius: 3px; }
  .msg.bot { align-self: flex-start; background: var(--surface);
             border-bottom-left-radius: 3px; }
  .msg.error { color: var(--danger); background: #2a1515; align-self: flex-start; }

  .input-bar { padding: 12px 16px; border-top: 1px solid var(--border);
               display: flex; gap: 8px; background: var(--surface); flex-shrink: 0; }
  .input-bar textarea { flex: 1; background: var(--bg); color: var(--text);
                        border: 1px solid var(--border); border-radius: var(--radius);
                        padding: 9px 13px; font-size: 14px; font-family: inherit;
                        resize: none; outline: none; min-height: 40px; max-height: 120px; }
  .input-bar textarea:focus { border-color: var(--accent); }
  .input-bar button { background: var(--accent); color: #fff; border: none;
                      border-radius: var(--radius); padding: 0 18px; cursor: pointer;
                      font-size: 14px; font-weight: 500; }
  .input-bar button:disabled { opacity: .4; cursor: not-allowed; }
  .input-bar button:hover:not(:disabled) { opacity: .88; }

  .cursor { display: inline-block; width: 2px; height: 1em; background: var(--text);
            vertical-align: text-bottom; margin-left: 2px;
            animation: blink .8s step-end infinite; }
  @keyframes blink { 50% { opacity: 0; } }

  /* ── Markdown rendering ── */
  .msg.bot h1, .msg.bot h2, .msg.bot h3 { font-weight: 700; margin: 10px 0 4px; line-height: 1.3; }
  .msg.bot h1 { font-size: 1.15em; }
  .msg.bot h2 { font-size: 1.05em; }
  .msg.bot h3 { font-size: 0.97em; }
  .msg.bot p { margin: 0 0 8px; }
  .msg.bot p:last-child { margin-bottom: 0; }
  .msg.bot ul, .msg.bot ol { margin: 4px 0 8px 18px; padding: 0; }
  .msg.bot li { margin-bottom: 3px; }
  .msg.bot strong { font-weight: 700; }
  .msg.bot em { font-style: italic; }
  .msg.bot code { font-family: 'JetBrains Mono', 'Fira Code', monospace; font-size: 0.88em;
                  background: #0d0d10; border: 1px solid var(--border);
                  border-radius: 4px; padding: 1px 5px; }
  .msg.bot pre { background: #0d0d10; border: 1px solid var(--border); border-radius: 8px;
                 padding: 12px 14px; overflow-x: auto; margin: 6px 0 10px; }
  .msg.bot pre code { background: none; border: none; padding: 0; font-size: 0.85em; line-height: 1.55; }
  .msg.bot blockquote { border-left: 3px solid var(--accent); padding-left: 10px;
                        color: var(--muted); margin: 6px 0; }
  .msg.bot hr { border: none; border-top: 1px solid var(--border); margin: 10px 0; }

  @media (max-width: 700px) {
    .app { flex-direction: column; }
    .sidebar { width: 100%; max-height: 220px; border-right: none;
               border-bottom: 1px solid var(--border); }
    .msg { max-width: 95%; }
  }
</style>
</head>
<body>

<header>
  <h1>DocAnalyzer</h1>
  <div class="mode-toggle">
    <span id="labelMemory" class="active">Memory</span>
    <label class="switch">
      <input type="checkbox" id="modeToggle" onchange="switchMode(this.checked)">
      <span class="slider"></span>
    </label>
    <span id="labelPersist">Persist</span>
  </div>
  <div class="header-right">
    <select id="role"></select>
    <select id="model"></select>
  </div>
</header>

<div class="app">
  <aside class="sidebar">

    <!-- MEMORY sidebar -->
    <div id="memorySidebar">
      <div class="sidebar-section">Documents</div>
      <div class="upload-zone">
        <input type="file" id="fileInputMemory" onchange="handleUpload(this, 'default')">
        <button class="upload-btn" onclick="document.getElementById('fileInputMemory').click()">
          Upload Document
        </button>
      </div>
      <div class="file-list" id="memoryFileList">
        <div class="empty-msg">No documents loaded</div>
      </div>
      <div class="sidebar-footer">
        <button class="clear-btn" onclick="clearAll()">Clear All</button>
      </div>
    </div>

    <!-- PERSIST sidebar -->
    <div id="persistSidebar" style="display:none; display:none; flex-direction:column; flex:1; overflow:hidden;">

      <div class="collections-header">
        <span>Collections</span>
        <button class="new-col-btn" onclick="toggleNewColForm()" title="New collection">+</button>
      </div>

      <div class="new-col-form" id="newColForm">
        <input type="text" id="newColInput" placeholder="Collection name..." maxlength="64"
               onkeydown="if(event.key==='Enter')createCollection()">
        <div class="row">
          <button class="btn-sm btn-primary" onclick="createCollection()">Create</button>
          <button class="btn-sm btn-ghost" onclick="toggleNewColForm()">Cancel</button>
        </div>
      </div>

      <div class="col-list" id="colList"></div>

      <div class="files-section-header">
        Files in <span class="in-col" id="activeColLabel">—</span>
      </div>
      <div class="upload-zone">
        <input type="file" id="fileInputPersist" onchange="handleUploadPersist(this)">
        <button class="upload-btn" id="uploadBtnPersist"
                onclick="document.getElementById('fileInputPersist').click()">
          Upload to Collection
        </button>
      </div>
      <div class="file-list" id="persistFileList">
        <div class="empty-msg">Select a collection</div>
      </div>
      <div class="sidebar-footer">
        <button class="clear-btn" onclick="clearAll()">Clear All</button>
      </div>

    </div>
  </aside>

  <main class="chat">
    <div class="messages" id="messages"></div>
    <div class="input-bar">
      <textarea id="questionInput" placeholder="Ask a question..." rows="1"
        onkeydown="if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();sendQuery()}"></textarea>
      <button id="sendBtn" onclick="sendQuery()" disabled>Send</button>
    </div>
  </main>
</div>

<script>
const API = '';

let state = {
  mode: 'memory',
  collections: [],
  queryCollections: new Set(),  // checked for query (persist)
  activeCollection: null,       // selected for upload (persist)
  filesMap: {},                 // {collection: [filenames]}
  models: [],
  roles: [],
  default_model: null,
};

// ── Init ─────────────────────────────────────────────────────────────────────

async function init() {
  const res = await fetch(API + '/api/status');
  const data = await res.json();
  state.mode = data.mode;
  state.collections = data.collections || [];
  state.filesMap = data.files_map || {};
  state.models = data.models;
  state.roles = data.roles;
  state.default_model = data.default_model;

  document.getElementById('role').innerHTML =
    state.roles.map(r => `<option value="${r}">${r}</option>`).join('');
  document.getElementById('model').innerHTML =
    state.models.map(m =>
      `<option value="${m}"${m === state.default_model ? ' selected' : ''}>${m}</option>`
    ).join('');

  document.getElementById('modeToggle').checked = state.mode === 'persist';
  updateModeLabels();

  if (state.mode === 'persist') {
    showPersistSidebar();
    // default: check all existing collections
    state.queryCollections = new Set(state.collections);
    if (state.collections.length > 0) state.activeCollection = state.collections[0];
    renderCollections();
  } else {
    showMemorySidebar();
    renderMemoryFiles();
  }
  updateSendBtn();
}

// ── Mode ─────────────────────────────────────────────────────────────────────

function updateModeLabels() {
  document.getElementById('labelMemory').className = state.mode === 'memory' ? 'active' : '';
  document.getElementById('labelPersist').className = state.mode === 'persist' ? 'active' : '';
}

async function switchMode(isPersist) {
  const newMode = isPersist ? 'persist' : 'memory';
  const res = await fetch(API + '/api/mode', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({mode: newMode}),
  });
  if (!res.ok) { alert('Failed to switch mode'); document.getElementById('modeToggle').checked = !isPersist; return; }
  const data = await res.json();
  state.mode = data.mode;
  state.collections = data.collections || [];
  state.filesMap = data.files_map || {};
  state.queryCollections = new Set(state.collections);
  state.activeCollection = state.collections.length > 0 ? state.collections[0] : null;
  updateModeLabels();
  if (isPersist) { showPersistSidebar(); renderCollections(); }
  else { showMemorySidebar(); renderMemoryFiles(); }
  updateSendBtn();
}

function showMemorySidebar() {
  document.getElementById('memorySidebar').style.display = '';
  const ps = document.getElementById('persistSidebar');
  ps.style.display = 'none';
}

function showPersistSidebar() {
  document.getElementById('memorySidebar').style.display = 'none';
  const ps = document.getElementById('persistSidebar');
  ps.style.display = 'flex';
  ps.style.flexDirection = 'column';
  ps.style.flex = '1';
  ps.style.overflow = 'hidden';
}

// ── Memory sidebar ────────────────────────────────────────────────────────────

function renderMemoryFiles() {
  const files = state.filesMap['default'] || [];
  const el = document.getElementById('memoryFileList');
  if (!files.length) { el.innerHTML = '<div class="empty-msg">No documents loaded</div>'; return; }
  el.innerHTML = files.map(f =>
    `<div class="file-item">
       <span title="${f}">${f}</span>
       <button class="icon-btn" onclick="removeFile('default','${escHtml(f)}')" title="Remove">&times;</button>
     </div>`
  ).join('');
}

// ── Collections sidebar ───────────────────────────────────────────────────────

function renderCollections() {
  const el = document.getElementById('colList');
  if (!state.collections.length) {
    el.innerHTML = '<div class="empty-msg" style="padding:10px">No collections yet</div>';
    document.getElementById('activeColLabel').textContent = '—';
    document.getElementById('persistFileList').innerHTML = '<div class="empty-msg">No collections yet</div>';
    return;
  }
  el.innerHTML = state.collections.map(col => {
    const checked = state.queryCollections.has(col);
    const active = col === state.activeCollection;
    const count = (state.filesMap[col] || []).length;
    return `<div class="col-item${active ? ' active' : ''}" id="colitem-${escId(col)}">
      <input type="checkbox" ${checked ? 'checked' : ''} onclick="toggleQueryCol('${escHtml(col)}',this.checked)">
      <span class="col-name" onclick="setActiveCol('${escHtml(col)}')" title="${escHtml(col)}">${escHtml(col)}</span>
      <span class="col-count">${count} file${count !== 1 ? 's' : ''}</span>
      <button class="icon-btn" onclick="deleteCollection('${escHtml(col)}')" title="Delete collection">&times;</button>
    </div>`;
  }).join('');

  // Render files for active collection
  renderPersistFiles();
  document.getElementById('activeColLabel').textContent = state.activeCollection || '—';
  const uploadBtn = document.getElementById('uploadBtnPersist');
  uploadBtn.disabled = !state.activeCollection;
}

function renderPersistFiles() {
  const el = document.getElementById('persistFileList');
  if (!state.activeCollection) { el.innerHTML = '<div class="empty-msg">Select a collection</div>'; return; }
  const files = state.filesMap[state.activeCollection] || [];
  if (!files.length) { el.innerHTML = '<div class="empty-msg">No files in this collection</div>'; return; }
  el.innerHTML = files.map(f =>
    `<div class="file-item">
       <span title="${escHtml(f)}">${escHtml(f)}</span>
       <button class="icon-btn" onclick="removeFile('${escHtml(state.activeCollection)}','${escHtml(f)}')" title="Remove">&times;</button>
     </div>`
  ).join('');
}

function setActiveCol(col) {
  state.activeCollection = col;
  renderCollections();
}

function toggleQueryCol(col, checked) {
  if (checked) state.queryCollections.add(col);
  else state.queryCollections.delete(col);
  updateSendBtn();
}

function toggleNewColForm() {
  const form = document.getElementById('newColForm');
  form.classList.toggle('visible');
  if (form.classList.contains('visible')) document.getElementById('newColInput').focus();
  else document.getElementById('newColInput').value = '';
}

async function createCollection() {
  const input = document.getElementById('newColInput');
  const name = input.value.trim();
  if (!name) return;
  const res = await fetch(API + '/api/collections', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({name}),
  });
  if (!res.ok) { const e = await res.json(); addMsg('error', e.detail || 'Failed'); return; }
  const data = await res.json();
  state.collections = data.collections;
  if (!state.filesMap[name]) state.filesMap[name] = [];
  state.queryCollections.add(name);
  state.activeCollection = name;
  input.value = '';
  toggleNewColForm();
  renderCollections();
  updateSendBtn();
}

async function deleteCollection(col) {
  if (!confirm(`Delete collection "${col}" and all its documents?`)) return;
  const res = await fetch(API + '/api/collections/' + encodeURIComponent(col), {method: 'DELETE'});
  if (!res.ok) { addMsg('error', 'Failed to delete collection'); return; }
  const data = await res.json();
  state.collections = data.collections;
  delete state.filesMap[col];
  state.queryCollections.delete(col);
  if (state.activeCollection === col) state.activeCollection = state.collections[0] || null;
  renderCollections();
  updateSendBtn();
}

// ── Upload ────────────────────────────────────────────────────────────────────

async function handleUpload(input, collection) {
  const file = input.files[0];
  if (!file) return;
  const btn = input.closest('.upload-zone').querySelector('.upload-btn');
  btn.disabled = true;
  btn.textContent = 'Uploading...';
  const fd = new FormData();
  fd.append('file', file);
  fd.append('collection', collection);
  try {
    const res = await fetch(API + '/api/upload', {method: 'POST', body: fd});
    if (!res.ok) { const e = await res.json(); throw new Error(e.detail || 'Upload failed'); }
    const data = await res.json();
    if (!state.filesMap[data.collection]) state.filesMap[data.collection] = [];
    state.filesMap[data.collection].push(data.file);
    if (state.mode === 'memory') renderMemoryFiles();
    else renderCollections();
    addMsg('bot', `Added "${data.file}" to "${data.collection}" (${data.chunks} chunks)`);
    updateSendBtn();
  } catch (err) { addMsg('error', err.message); }
  finally { btn.disabled = false; btn.textContent = state.mode === 'persist' ? 'Upload to Collection' : 'Upload Document'; input.value = ''; }
}

async function handleUploadPersist(input) {
  if (!state.activeCollection) { addMsg('error', 'Select a collection first'); input.value = ''; return; }
  await handleUpload(input, state.activeCollection);
}

// ── Remove file ───────────────────────────────────────────────────────────────

async function removeFile(collection, name) {
  const res = await fetch(
    API + '/api/collections/' + encodeURIComponent(collection) + '/files/' + encodeURIComponent(name),
    {method: 'DELETE'}
  );
  if (!res.ok) { addMsg('error', 'Failed to remove file'); return; }
  state.filesMap[collection] = (state.filesMap[collection] || []).filter(f => f !== name);
  if (state.mode === 'memory') renderMemoryFiles();
  else renderCollections();
  updateSendBtn();
}

// ── Clear all ─────────────────────────────────────────────────────────────────

async function clearAll() {
  if (!confirm('Clear all documents?')) return;
  await fetch(API + '/api/files', {method: 'DELETE'});
  state.filesMap = {};
  state.collections = [];
  state.queryCollections = new Set();
  state.activeCollection = null;
  if (state.mode === 'memory') renderMemoryFiles();
  else renderCollections();
  document.getElementById('messages').innerHTML = '';
  updateSendBtn();
}

// ── Query ─────────────────────────────────────────────────────────────────────

function getQueryCollections() {
  if (state.mode === 'memory') return [];  // backend resolves to all
  return [...state.queryCollections];
}

async function sendQuery() {
  const input = document.getElementById('questionInput');
  const q = input.value.trim();
  if (!q) return;

  const cols = getQueryCollections();
  if (state.mode === 'persist' && cols.length === 0) {
    addMsg('error', 'Select at least one collection to query');
    return;
  }

  input.value = '';
  input.style.height = 'auto';
  addMsg('user', q);

  const msgId = addMsg('bot', '');
  const msgEl = document.getElementById(msgId);
  const messagesEl = document.getElementById('messages');
  document.getElementById('sendBtn').disabled = true;

  // Show blinking cursor while waiting for first token
  msgEl.innerHTML = '<span class="cursor"></span>';

  let fullText = '';
  let firstChunk = true;

  try {
    const res = await fetch(API + '/api/query/stream', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        question: q,
        role: document.getElementById('role').value,
        model: document.getElementById('model').value,
        collections: cols,
      }),
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Query failed');
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, {stream: true});
      const lines = buffer.split('\\n');
      buffer = lines.pop();
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const payload = line.slice(6).trim();
        if (payload === '[DONE]') break;
        let parsed;
        try { parsed = JSON.parse(payload); } catch { continue; }
        if (parsed.error) throw new Error(parsed.error);
        if (parsed.text) {
          firstChunk = false;
          fullText += parsed.text;
          // Render markdown progressively, keep cursor at end while streaming
          msgEl.innerHTML = renderMarkdown(fullText) + '<span class="cursor"></span>';
          messagesEl.scrollTop = messagesEl.scrollHeight;
        }
      }
    }

    // Final render without cursor
    if (firstChunk) {
      msgEl.textContent = '(no response)';
    } else {
      msgEl.innerHTML = renderMarkdown(fullText);
    }

  } catch (err) {
    msgEl.className = 'msg error';
    msgEl.textContent = err.message;
  } finally {
    updateSendBtn();
    input.focus();
  }
}

// ── Markdown renderer ────────────────────────────────────────────────────────

function renderMarkdown(text) {
  function esc(s) {
    return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  }
  function inline(s) {
    // inline code first (protect from bold/italic processing)
    s = s.replace(/`([^`]+)`/g, (_, c) => `<code>${esc(c)}</code>`);
    s = s.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    s = s.replace(/__(.+?)__/g, '<strong>$1</strong>');
    s = s.replace(/\*(.+?)\*/g, '<em>$1</em>');
    s = s.replace(/_(.+?)_/g, '<em>$1</em>');
    return s;
  }

  const lines = text.split('\\n');
  const out = [];
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];

    // Fenced code block
    if (line.startsWith('```')) {
      const codeLines = [];
      i++;
      while (i < lines.length && !lines[i].startsWith('```')) {
        codeLines.push(esc(lines[i]));
        i++;
      }
      out.push(`<pre><code>${codeLines.join('\\n')}</code></pre>`);
      i++;
      continue;
    }

    // Headings
    let m;
    if ((m = line.match(/^(#{1,3}) (.+)/))) {
      const lvl = m[1].length;
      out.push(`<h${lvl}>${inline(esc(m[2]))}</h${lvl}>`);
      i++; continue;
    }

    // Horizontal rule
    if (/^[-*_]{3,}$/.test(line.trim())) {
      out.push('<hr>'); i++; continue;
    }

    // Blockquote
    if (line.startsWith('> ')) {
      const bqLines = [];
      while (i < lines.length && lines[i].startsWith('> ')) {
        bqLines.push(inline(esc(lines[i].slice(2))));
        i++;
      }
      out.push(`<blockquote>${bqLines.join('<br>')}</blockquote>`);
      continue;
    }

    // Unordered list
    if (/^[-*+] /.test(line)) {
      const items = [];
      while (i < lines.length && /^[-*+] /.test(lines[i])) {
        items.push(`<li>${inline(esc(lines[i].replace(/^[-*+] /, '')))}</li>`);
        i++;
      }
      out.push(`<ul>${items.join('')}</ul>`);
      continue;
    }

    // Ordered list
    if (/^\d+[.)]\s/.test(line)) {
      const items = [];
      while (i < lines.length && /^\d+[.)]\s/.test(lines[i])) {
        items.push(`<li>${inline(esc(lines[i].replace(/^\d+[.)]\s/, '')))}</li>`);
        i++;
      }
      out.push(`<ol>${items.join('')}</ol>`);
      continue;
    }

    // Empty line
    if (line.trim() === '') { i++; continue; }

    // Paragraph
    out.push(`<p>${inline(esc(line))}</p>`);
    i++;
  }
  return out.join('');
}

// ── Utils ─────────────────────────────────────────────────────────────────────

function updateSendBtn() {
  const hasFiles = Object.values(state.filesMap).some(f => f.length > 0);
  const hasQueryTarget = state.mode === 'memory'
    ? hasFiles
    : [...state.queryCollections].some(col => (state.filesMap[col] || []).length > 0);
  document.getElementById('sendBtn').disabled = !hasQueryTarget;
}

let msgCounter = 0;
function addMsg(type, text) {
  const el = document.getElementById('messages');
  const id = 'msg-' + (++msgCounter);
  const div = document.createElement('div');
  div.className = 'msg ' + type;
  div.id = id;
  div.textContent = text;
  el.appendChild(div);
  el.scrollTop = el.scrollHeight;
  return id;
}

function escHtml(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;').replace(/'/g,'&#39;');
}
function escId(s) { return s.replace(/[^a-zA-Z0-9_-]/g, '_'); }

document.getElementById('questionInput').addEventListener('input', function() {
  this.style.height = 'auto';
  this.style.height = Math.min(this.scrollHeight, 120) + 'px';
});

init();
</script>
</body>
</html>"""
