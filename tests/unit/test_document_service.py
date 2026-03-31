"""
Unit tests for DocumentService.

These tests verify state management, delegation to RAGProcessor,
and edge cases — all without a real Qdrant or Ollama instance.
"""
import pytest
from unittest.mock import MagicMock, call

from src.models.document import Document
from src.services.document_service import DocumentService


@pytest.fixture
def mock_rag():
    rag = MagicMock()
    rag.mode = "memory"
    rag.add_document.return_value = ["id1", "id2"]
    rag.remove_document.return_value = True
    rag.get_collections.return_value = []
    return rag


@pytest.fixture
def service(mock_rag):
    return DocumentService(mock_rag)


@pytest.fixture
def chunks():
    return [Document(page_content="chunk1", metadata={})]


# ── has_documents ─────────────────────────────────────────────────────────────

def test_has_documents_empty(service):
    assert service.has_documents() is False


def test_has_documents_after_add(service, mock_rag, chunks):
    service.add_file("doc.txt", chunks, "default")
    assert service.has_documents() is True


# ── is_duplicate ──────────────────────────────────────────────────────────────

def test_is_duplicate_false_when_empty(service):
    assert service.is_duplicate("doc.txt", "default") is False


def test_is_duplicate_true_after_add(service, mock_rag, chunks):
    service.add_file("doc.txt", chunks, "default")
    assert service.is_duplicate("doc.txt", "default") is True


def test_is_duplicate_different_collection(service, mock_rag, chunks):
    service.add_file("doc.txt", chunks, "col-a")
    assert service.is_duplicate("doc.txt", "col-b") is False


# ── add_file ──────────────────────────────────────────────────────────────────

def test_add_file_delegates_to_rag(service, mock_rag, chunks):
    n = service.add_file("doc.txt", chunks, "mycol")
    mock_rag.add_document.assert_called_once_with(chunks, collection_name="mycol")
    assert n == 2  # len(["id1", "id2"])


def test_add_file_updates_files_map(service, mock_rag, chunks):
    service.add_file("doc.txt", chunks, "mycol")
    assert "doc.txt" in service.files_map["mycol"]
    assert service.files_map["mycol"]["doc.txt"] == ["id1", "id2"]


# ── remove_file ───────────────────────────────────────────────────────────────

def test_remove_file_delegates_to_rag(service, mock_rag, chunks):
    service.add_file("doc.txt", chunks, "default")
    ok = service.remove_file("doc.txt", "default")
    assert ok is True
    mock_rag.remove_document.assert_called_once_with(
        ["id1", "id2"], collection_name="default"
    )


def test_remove_file_updates_files_map(service, mock_rag, chunks):
    service.add_file("doc.txt", chunks, "default")
    service.remove_file("doc.txt", "default")
    assert "doc.txt" not in service.files_map.get("default", {})


def test_remove_nonexistent_file_returns_false(service):
    ok = service.remove_file("ghost.txt", "default")
    assert ok is False


def test_remove_file_rag_failure_returns_false(service, mock_rag, chunks):
    mock_rag.remove_document.return_value = False
    service.add_file("doc.txt", chunks, "default")
    ok = service.remove_file("doc.txt", "default")
    assert ok is False
    # File should remain in map since removal failed
    assert "doc.txt" in service.files_map["default"]


# ── resolve_query_collections ─────────────────────────────────────────────────

def test_resolve_uses_explicit_list(service):
    assert service.resolve_query_collections(["col-a"]) == ["col-a"]


def test_resolve_falls_back_to_all_loaded(service, mock_rag, chunks):
    service.add_file("a.txt", chunks, "col-a")
    mock_rag.add_document.return_value = ["id3"]
    service.add_file("b.txt", chunks, "col-b")
    result = service.resolve_query_collections([])
    assert set(result) == {"col-a", "col-b"}


# ── create / delete collection ────────────────────────────────────────────────

def test_create_collection(service, mock_rag):
    service.create_collection("new-col")
    mock_rag.create_collection.assert_called_once_with("new-col")
    assert "new-col" in service.files_map


def test_delete_collection(service, mock_rag, chunks):
    service.add_file("doc.txt", chunks, "target")
    service.delete_collection("target")
    mock_rag.delete_collection.assert_called_once_with("target")
    assert "target" not in service.files_map


# ── clear_all ─────────────────────────────────────────────────────────────────

def test_clear_all_calls_rag_reset(service, mock_rag, chunks):
    service.add_file("doc.txt", chunks, "default")
    service.clear_all()
    mock_rag.reset.assert_called_once()
    assert service.files_map == {}


# ── switch_mode ───────────────────────────────────────────────────────────────

def test_switch_to_persist_rebuilds_map(service, mock_rag):
    mock_rag.rebuild_files_map.return_value = {"col": {"file.txt": ["id1"]}}
    service.switch_mode("persist")
    mock_rag.set_mode.assert_called_once_with("persist")
    assert service.files_map == {"col": {"file.txt": ["id1"]}}


def test_switch_to_memory_clears_map(service, mock_rag, chunks):
    service.add_file("doc.txt", chunks, "default")
    service.switch_mode("memory")
    mock_rag.set_mode.assert_called_once_with("memory")
    assert service.files_map == {}
