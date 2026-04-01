"""
Shared fixtures for unit tests that need a FastAPI TestClient
with all external dependencies mocked out.
"""
import importlib
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


def _make_mock_rag():
    mock = MagicMock()
    mock.mode = "memory"
    mock.get_collections.return_value = []
    mock.add_document.return_value = ["id1", "id2"]
    mock.remove_document.return_value = True
    mock.query.return_value = "Test response"
    return mock


def _make_mock_processor(chunks=None):
    chunk = MagicMock()
    chunk.page_content = "chunk content"
    chunk.metadata = {}
    mock = MagicMock()
    mock.process.return_value = chunks if chunks is not None else [chunk]
    return mock


@pytest.fixture
def app_client(monkeypatch):
    """
    TestClient with ollama and qdrant patched out.
    Yields (client, mock_rag, mock_get_processor).
    """
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("OLLAMA_HOST", "localhost")
    monkeypatch.setenv("OLLAMA_PORT", "11434")
    monkeypatch.setenv("EMBEDDING_MODEL", "test-embed-model")

    with patch("src.processors.rag_processor.ollama.Client"), \
         patch("src.processors.rag_processor.QdrantClient") as mock_qdrant_cls:

        mock_qdrant_cls.return_value = MagicMock(
            get_collections=MagicMock(return_value=MagicMock(collections=[]))
        )

        import src.app
        importlib.reload(src.app)

        mock_rag = _make_mock_rag()
        src.app.rag_processor = mock_rag
        src.app.document_service._rag = mock_rag
        src.app.document_service._files_map = {}

        mock_get_processor = MagicMock(return_value=_make_mock_processor())

        with patch.object(src.app, "get_processor", mock_get_processor):
            yield TestClient(src.app.app), mock_rag, mock_get_processor
