from dotenv import load_dotenv
load_dotenv()

import json
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
import os
import tempfile
import shutil
from src.processors.factory import ProcessorFactory
from src.processors.rag_processor import RAGProcessor
from src.config.prompts import ROLE_PROMPTS


app = FastAPI(title="DocAnalyzer")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_processor = RAGProcessor()

# {file_name: [document_ids]}
processed_files_map: dict[str, list[str]] = {}

available_models = rag_processor.get_available_models()
default_model = os.getenv("LLM_MODEL")
if default_model not in available_models:
    available_models.insert(0, default_model)
else:
    available_models.remove(default_model)
    available_models.insert(0, default_model)


# ── API ────────────────────────────────────────────────────────────────────────

@app.get("/api/status")
def status():
    return {
        "files": list(processed_files_map.keys()),
        "models": available_models,
        "default_model": default_model,
        "roles": list(ROLE_PROMPTS.keys()),
    }


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    name = file.filename
    if name in processed_files_map:
        raise HTTPException(400, f"'{name}' is already in context")

    suffix = os.path.splitext(name)[1]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        content = await file.read()
        tmp.write(content)
        tmp.close()

        processor = ProcessorFactory.get_processor(tmp.name)
        chunks = processor.process(tmp.name)

        if not chunks:
            raise HTTPException(422, f"No content extracted from '{name}'")

        ids = rag_processor.add_document(chunks)
        processed_files_map[name] = ids
        return {"file": name, "chunks": len(chunks)}
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(422, str(e))
    except Exception as e:
        raise HTTPException(500, f"Error processing '{name}': {e}")
    finally:
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)


@app.delete("/api/files/{file_name}")
def remove_file(file_name: str):
    if file_name not in processed_files_map:
        raise HTTPException(404, f"'{file_name}' not found")
    ids = processed_files_map[file_name]
    ok = rag_processor.remove_document(ids)
    if ok:
        del processed_files_map[file_name]
        return {"removed": file_name}
    raise HTTPException(500, f"Failed to remove '{file_name}'")


@app.delete("/api/files")
def clear_all():
    global processed_files_map
    processed_files_map = {}
    rag_processor._clean_db()
    return {"cleared": True}


class QueryRequest(BaseModel):
    question: str
    role: str = "default"
    model: str | None = None


@app.post("/api/query")
def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(400, "Empty question")
    if not processed_files_map:
        raise HTTPException(400, "No documents in context")
    try:
        answer = rag_processor.query(req.question.strip(), req.role, req.model)
        return {"answer": answer}
    except ValueError as e:
        raise HTTPException(422, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/query/stream")
def query_stream(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(400, "Empty question")
    if not processed_files_map:
        raise HTTPException(400, "No documents in context")

    def generate():
        try:
            for chunk in rag_processor.query_stream(req.question.strip(), req.role, req.model):
                yield f"data: {json.dumps({'text': chunk})}\n\n"
        except ValueError as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
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


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DocAnalyzer</title>
<style>
  :root { --bg: #111; --surface: #1a1a1e; --border: #2a2a2e; --text: #e4e4e7;
          --muted: #888; --accent: #3b82f6; --danger: #ef4444; --radius: 8px; }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: var(--bg); color: var(--text); height: 100vh; display: flex;
         flex-direction: column; }

  header { padding: 12px 20px; border-bottom: 1px solid var(--border);
           display: flex; align-items: center; gap: 16px; flex-shrink: 0; }
  header h1 { font-size: 18px; font-weight: 600; }
  header .controls { display: flex; gap: 8px; margin-left: auto; align-items: center; }
  header select { background: var(--surface); color: var(--text); border: 1px solid var(--border);
                  border-radius: var(--radius); padding: 6px 10px; font-size: 13px; }

  .app { display: flex; flex: 1; overflow: hidden; }

  /* Sidebar */
  .sidebar { width: 280px; border-right: 1px solid var(--border); display: flex;
             flex-direction: column; flex-shrink: 0; }
  .sidebar-header { padding: 12px 16px; border-bottom: 1px solid var(--border);
                    font-size: 13px; font-weight: 600; color: var(--muted); text-transform: uppercase; }
  .upload-zone { padding: 16px; border-bottom: 1px solid var(--border); }
  .upload-zone input[type=file] { display: none; }
  .upload-btn { width: 100%; padding: 10px; background: var(--accent); color: #fff;
                border: none; border-radius: var(--radius); cursor: pointer; font-size: 13px;
                font-weight: 500; }
  .upload-btn:hover { opacity: 0.9; }
  .upload-btn:disabled { opacity: 0.5; cursor: not-allowed; }
  .file-list { flex: 1; overflow-y: auto; padding: 8px; }
  .file-item { display: flex; justify-content: space-between; align-items: center;
               padding: 8px 10px; border-radius: 6px; font-size: 13px;
               background: var(--surface); margin-bottom: 4px; }
  .file-item button { background: none; border: none; color: var(--danger); cursor: pointer;
                      font-size: 16px; line-height: 1; padding: 0 4px; }
  .file-item button:hover { opacity: 0.7; }
  .sidebar-footer { padding: 8px 16px; border-top: 1px solid var(--border); }
  .clear-btn { width: 100%; padding: 8px; background: transparent; color: var(--danger);
               border: 1px solid var(--danger); border-radius: var(--radius); cursor: pointer;
               font-size: 12px; }
  .clear-btn:hover { background: var(--danger); color: #fff; }
  .empty-msg { color: var(--muted); font-size: 13px; padding: 16px; text-align: center; }

  /* Chat */
  .chat { flex: 1; display: flex; flex-direction: column; overflow: hidden; }
  .messages { flex: 1; overflow-y: auto; padding: 20px; display: flex;
              flex-direction: column; gap: 16px; }
  .msg { max-width: 80%; padding: 12px 16px; border-radius: 12px; font-size: 14px;
         line-height: 1.6; white-space: pre-wrap; word-wrap: break-word; }
  .msg.user { align-self: flex-end; background: var(--accent); color: #fff;
              border-bottom-right-radius: 4px; }
  .msg.bot { align-self: flex-start; background: var(--surface);
             border-bottom-left-radius: 4px; }
  .msg.bot p { margin: 0 0 8px 0; }
  .msg.bot p:last-child { margin-bottom: 0; }
  .msg.error { color: var(--danger); background: #2a1515; }

  .input-bar { padding: 12px 20px; border-top: 1px solid var(--border); display: flex; gap: 8px; }
  .input-bar textarea { flex: 1; background: var(--surface); color: var(--text);
                        border: 1px solid var(--border); border-radius: var(--radius);
                        padding: 10px 14px; font-size: 14px; font-family: inherit;
                        resize: none; outline: none; min-height: 42px; max-height: 120px; }
  .input-bar textarea:focus { border-color: var(--accent); }
  .input-bar button { background: var(--accent); color: #fff; border: none;
                      border-radius: var(--radius); padding: 0 20px; cursor: pointer;
                      font-size: 14px; font-weight: 500; }
  .input-bar button:disabled { opacity: 0.4; cursor: not-allowed; }
  .input-bar button:hover:not(:disabled) { opacity: 0.9; }

  .spinner { display: inline-block; width: 16px; height: 16px; border: 2px solid var(--muted);
             border-top-color: var(--accent); border-radius: 50%;
             animation: spin 0.6s linear infinite; vertical-align: middle; margin-right: 6px; }
  @keyframes spin { to { transform: rotate(360deg); } }
  .cursor { display: inline-block; width: 2px; height: 1em; background: var(--text);
            vertical-align: text-bottom; margin-left: 2px;
            animation: blink 0.8s step-end infinite; }
  @keyframes blink { 50% { opacity: 0; } }

  @media (max-width: 700px) {
    .app { flex-direction: column; }
    .sidebar { width: 100%; max-height: 200px; border-right: none; border-bottom: 1px solid var(--border); }
    .msg { max-width: 95%; }
  }
</style>
</head>
<body>

<header>
  <h1>DocAnalyzer</h1>
  <div class="controls">
    <select id="role"></select>
    <select id="model"></select>
  </div>
</header>

<div class="app">
  <aside class="sidebar">
    <div class="sidebar-header">Documents</div>
    <div class="upload-zone">
      <input type="file" id="fileInput"
        accept=".pdf,.doc,.docx,.txt,.rtf,.py,.js,.java,.c,.cpp,.html,.css,.php,.go,.ts,.rb,.json,.xml,.md,.yaml,.yml,.xlsx,.xls,.csv,.ods">
      <button class="upload-btn" id="uploadBtn" onclick="document.getElementById('fileInput').click()">
        Upload Document
      </button>
    </div>
    <div class="file-list" id="fileList">
      <div class="empty-msg">No documents loaded</div>
    </div>
    <div class="sidebar-footer">
      <button class="clear-btn" id="clearBtn" onclick="clearAll()">Clear All</button>
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
let state = { files: [], models: [], roles: [] };

async function init() {
  const res = await fetch(API + '/api/status');
  state = await res.json();
  renderFiles();
  const roleEl = document.getElementById('role');
  roleEl.innerHTML = state.roles.map(r => `<option value="${r}">${r}</option>`).join('');
  const modelEl = document.getElementById('model');
  modelEl.innerHTML = state.models.map(m => `<option value="${m}"${m===state.default_model?' selected':''}>${m}</option>`).join('');
  document.getElementById('sendBtn').disabled = state.files.length === 0;
}

function renderFiles() {
  const el = document.getElementById('fileList');
  if (!state.files.length) {
    el.innerHTML = '<div class="empty-msg">No documents loaded</div>';
    document.getElementById('sendBtn').disabled = true;
    return;
  }
  el.innerHTML = state.files.map(f =>
    `<div class="file-item"><span>${f}</span><button onclick="removeFile('${f}')">&times;</button></div>`
  ).join('');
  document.getElementById('sendBtn').disabled = false;
}

document.getElementById('fileInput').addEventListener('change', async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const btn = document.getElementById('uploadBtn');
  btn.disabled = true;
  btn.textContent = 'Uploading...';
  const fd = new FormData();
  fd.append('file', file);
  try {
    const res = await fetch(API + '/api/upload', { method: 'POST', body: fd });
    if (!res.ok) { const err = await res.json(); throw new Error(err.detail || 'Upload failed'); }
    const data = await res.json();
    state.files.push(data.file);
    renderFiles();
    addMsg('bot', `Added "${data.file}" (${data.chunks} chunks)`);
  } catch (err) {
    addMsg('error', err.message);
  } finally {
    btn.disabled = false;
    btn.textContent = 'Upload Document';
    e.target.value = '';
  }
});

async function removeFile(name) {
  try {
    const res = await fetch(API + '/api/files/' + encodeURIComponent(name), { method: 'DELETE' });
    if (!res.ok) throw new Error('Failed');
    state.files = state.files.filter(f => f !== name);
    renderFiles();
  } catch (err) { addMsg('error', err.message); }
}

async function clearAll() {
  if (!confirm('Clear all documents?')) return;
  await fetch(API + '/api/files', { method: 'DELETE' });
  state.files = [];
  renderFiles();
  document.getElementById('messages').innerHTML = '';
}

async function sendQuery() {
  const input = document.getElementById('questionInput');
  const q = input.value.trim();
  if (!q || !state.files.length) return;

  input.value = '';
  input.style.height = 'auto';
  addMsg('user', q);

  const msgId = addMsg('bot', '');
  const msgEl = document.getElementById(msgId);
  const messagesEl = document.getElementById('messages');
  document.getElementById('sendBtn').disabled = true;

  // Show blinking cursor while waiting for first token
  const cursor = document.createElement('span');
  cursor.className = 'cursor';
  msgEl.appendChild(cursor);

  let fullText = '';
  let firstChunk = true;

  try {
    const res = await fetch(API + '/api/query/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        question: q,
        role: document.getElementById('role').value,
        model: document.getElementById('model').value,
      })
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Query failed');
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop(); // hold incomplete line

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const payload = line.slice(6).trim();
        if (payload === '[DONE]') break;
        let parsed;
        try { parsed = JSON.parse(payload); } catch { continue; }
        if (parsed.error) throw new Error(parsed.error);
        if (parsed.text) {
          if (firstChunk) { cursor.remove(); firstChunk = false; }
          fullText += parsed.text;
          msgEl.textContent = fullText;
          messagesEl.scrollTop = messagesEl.scrollHeight;
        }
      }
    }

    // If LLM returned nothing at all
    if (firstChunk) { cursor.remove(); msgEl.textContent = '(no response)'; }

  } catch (err) {
    cursor.remove();
    msgEl.className = 'msg error';
    msgEl.textContent = err.message;
  } finally {
    document.getElementById('sendBtn').disabled = state.files.length === 0;
    input.focus();
  }
}

let msgCounter = 0;
function addMsg(type, html) {
  const el = document.getElementById('messages');
  const id = 'msg-' + (++msgCounter);
  const div = document.createElement('div');
  div.className = 'msg ' + type;
  div.id = id;
  div.innerHTML = html;
  el.appendChild(div);
  el.scrollTop = el.scrollHeight;
  return id;
}

function replaceMsg(id, type, text) {
  const el = document.getElementById(id);
  if (el) { el.className = 'msg ' + type; el.textContent = text; }
  document.getElementById('messages').scrollTop = document.getElementById('messages').scrollHeight;
}

// Auto-resize textarea
document.getElementById('questionInput').addEventListener('input', function() {
  this.style.height = 'auto';
  this.style.height = Math.min(this.scrollHeight, 120) + 'px';
});

init();
</script>
</body>
</html>"""
