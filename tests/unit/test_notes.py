"""Tests for NoteService and /api/notes endpoints."""
import pytest
import os
import tempfile
from src.models.note import Note
from src.services.note_service import NoteService


# ── NoteService unit tests ─────────────────────────────────────────────────────

@pytest.fixture
def svc(tmp_path):
    return NoteService(db_path=str(tmp_path / "notes.db"))


def test_save_and_get(svc):
    note = Note(question="What is RAG?", answer="Retrieval Augmented Generation.")
    saved = svc.save(note)
    assert saved.id > 0
    fetched = svc.get(saved.id)
    assert fetched.question == "What is RAG?"
    assert fetched.answer == "Retrieval Augmented Generation."


def test_save_persists_sources(svc):
    sources = [{"file": "doc.pdf", "page": 0, "score": 0.95}]
    note = Note(question="q", answer="a", sources=sources)
    saved = svc.save(note)
    fetched = svc.get(saved.id)
    assert fetched.sources == sources


def test_list_ordered_most_recent_first(svc):
    svc.save(Note(question="first", answer="a"))
    svc.save(Note(question="second", answer="b"))
    notes = svc.list()
    assert notes[0].question == "second"
    assert notes[1].question == "first"


def test_list_empty(svc):
    assert svc.list() == []


def test_get_nonexistent_returns_none(svc):
    assert svc.get(9999) is None


def test_delete(svc):
    saved = svc.save(Note(question="q", answer="a"))
    assert svc.delete(saved.id) is True
    assert svc.get(saved.id) is None


def test_delete_nonexistent_returns_false(svc):
    assert svc.delete(9999) is False


def test_title_truncated_at_80(svc):
    long_q = "x" * 100
    note = Note(question=long_q, answer="a")
    assert len(note.title) <= 81  # 80 chars + ellipsis
    assert note.title.endswith("…")


def test_title_short_no_ellipsis():
    note = Note(question="Short question", answer="a")
    assert note.title == "Short question"


def test_created_at_auto_set():
    note = Note(question="q", answer="a")
    assert note.created_at != ""


# ── API endpoint tests ─────────────────────────────────────────────────────────

@pytest.fixture
def notes_client(app_client, tmp_path):
    """app_client with a fresh NoteService injected."""
    client, mock_rag, mock_get_proc = app_client
    import src.app
    src.app.note_service = NoteService(db_path=str(tmp_path / "test_notes.db"))
    return client


def test_api_save_note(notes_client):
    res = notes_client.post("/api/notes", json={
        "question": "What is RAG?",
        "answer": "Retrieval Augmented Generation.",
        "sources": [],
    })
    assert res.status_code == 200
    data = res.json()
    assert data["id"] > 0
    assert "created_at" in data


def test_api_list_notes(notes_client):
    notes_client.post("/api/notes", json={"question": "q1", "answer": "a1"})
    notes_client.post("/api/notes", json={"question": "q2", "answer": "a2"})
    res = notes_client.get("/api/notes")
    assert res.status_code == 200
    assert len(res.json()["notes"]) == 2


def test_api_get_note(notes_client):
    saved = notes_client.post("/api/notes", json={"question": "q", "answer": "a"}).json()
    res = notes_client.get(f"/api/notes/{saved['id']}")
    assert res.status_code == 200
    assert res.json()["question"] == "q"


def test_api_get_note_not_found(notes_client):
    res = notes_client.get("/api/notes/9999")
    assert res.status_code == 404


def test_api_delete_note(notes_client):
    saved = notes_client.post("/api/notes", json={"question": "q", "answer": "a"}).json()
    res = notes_client.delete(f"/api/notes/{saved['id']}")
    assert res.status_code == 200
    assert notes_client.get(f"/api/notes/{saved['id']}").status_code == 404


def test_api_delete_note_not_found(notes_client):
    res = notes_client.delete("/api/notes/9999")
    assert res.status_code == 404


def test_api_note_stores_sources(notes_client):
    sources = [{"file": "report.pdf", "page": 2, "score": 0.87}]
    saved = notes_client.post("/api/notes", json={
        "question": "q", "answer": "a", "sources": sources
    }).json()
    fetched = notes_client.get(f"/api/notes/{saved['id']}").json()
    assert fetched["sources"] == sources
