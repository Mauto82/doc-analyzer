import json
import pytest
from unittest.mock import MagicMock, patch


def _seed_file(client):
    """Upload a dummy file so has_documents() returns True."""
    client.post("/api/upload", files={"file": ("seed.txt", b"content", "text/plain")})


# ── Streaming endpoint ─────────────────────────────────────────────────────────

def test_stream_empty_question(app_client):
    client, mock_rag, _ = app_client
    res = client.post("/api/query/stream", json={"question": "  "})
    assert res.status_code == 400


def test_stream_no_documents(app_client):
    client, mock_rag, _ = app_client
    res = client.post("/api/query/stream", json={"question": "hello?"})
    assert res.status_code == 400
    assert "No documents" in res.json()["detail"]


def _text_events(*tokens):
    """Build structured dict events as query_stream now yields."""
    return iter([{"type": "text", "text": t} for t in tokens])


def test_stream_yields_tokens(app_client):
    client, mock_rag, _ = app_client
    _seed_file(client)

    mock_rag.query_stream.return_value = _text_events("Hello", " world", "!")

    with client.stream("POST", "/api/query/stream",
                       json={"question": "What is this?", "role": "default"}) as resp:
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        raw = resp.read().decode()

    payloads = [
        json.loads(line[6:])
        for line in raw.splitlines()
        if line.startswith("data: ") and line[6:].strip() != "[DONE]"
    ]
    texts = [p["text"] for p in payloads if p.get("type") == "text"]
    assert "".join(texts) == "Hello world!"


def test_stream_sources_event_forwarded(app_client):
    client, mock_rag, _ = app_client
    _seed_file(client)

    sources = [{"file": "doc.pdf", "page": 0, "score": 0.9}]
    mock_rag.query_stream.return_value = iter([
        {"type": "sources", "sources": sources},
        {"type": "text", "text": "answer"},
    ])

    with client.stream("POST", "/api/query/stream",
                       json={"question": "test"}) as resp:
        raw = resp.read().decode()

    payloads = [
        json.loads(line[6:])
        for line in raw.splitlines()
        if line.startswith("data: ") and line[6:].strip() != "[DONE]"
    ]
    source_events = [p for p in payloads if p.get("type") == "sources"]
    assert len(source_events) == 1
    assert source_events[0]["sources"] == sources


def test_stream_done_sentinel(app_client):
    client, mock_rag, _ = app_client
    _seed_file(client)
    mock_rag.query_stream.return_value = _text_events("ok")

    with client.stream("POST", "/api/query/stream",
                       json={"question": "test"}) as resp:
        raw = resp.read().decode()

    assert "data: [DONE]" in raw


def test_stream_passes_role_and_model(app_client):
    client, mock_rag, _ = app_client
    _seed_file(client)
    mock_rag.query_stream.return_value = _text_events("answer")

    with client.stream("POST", "/api/query/stream",
                       json={"question": "q", "role": "legal",
                             "model": "llama3"}) as resp:
        resp.read()

    mock_rag.query_stream.assert_called_once()
    _, kwargs = mock_rag.query_stream.call_args
    assert kwargs.get("role") == "legal" or mock_rag.query_stream.call_args[0][2] == "legal"
    assert "llama3" in str(mock_rag.query_stream.call_args)


def test_stream_error_propagated(app_client):
    client, mock_rag, _ = app_client
    _seed_file(client)

    def failing_stream(*args, **kwargs):
        raise RuntimeError("embedding failed")
        yield  # make it a generator

    mock_rag.query_stream.side_effect = failing_stream

    with client.stream("POST", "/api/query/stream",
                       json={"question": "test"}) as resp:
        raw = resp.read().decode()

    # The error should be delivered as an SSE error event
    payloads = [
        json.loads(line[6:])
        for line in raw.splitlines()
        if line.startswith("data: ") and line[6:].strip() != "[DONE]"
    ]
    assert any("error" in p for p in payloads)
