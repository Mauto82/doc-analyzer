import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def mock_env(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("OLLAMA_HOST", "localhost")
    monkeypatch.setenv("OLLAMA_PORT", "11434")
    monkeypatch.setenv("EMBEDDING_MODEL", "test-embed-model")


@pytest.fixture
def client():
    with patch("src.processors.rag_processor.ollama.Client") as mock_ollama, \
         patch("src.processors.rag_processor.QdrantClient") as mock_qdrant:

        mock_qclient = MagicMock()
        mock_qdrant.return_value = mock_qclient
        mock_qclient.get_collections.return_value = MagicMock(collections=[])
        mock_ollama.return_value = MagicMock(
            list=MagicMock(return_value=MagicMock(models=[]))
        )

        import importlib
        import src.app
        importlib.reload(src.app)
        # Reset service state between tests
        src.app.document_service._files_map = {}

        yield TestClient(src.app.app)


def test_index(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "DocAnalyzer" in res.text


def test_status(client):
    res = client.get("/api/status")
    assert res.status_code == 200
    data = res.json()
    assert "files_map" in data
    assert "models" in data
    assert "roles" in data
    assert "mode" in data
    assert "collections" in data


def test_query_empty_question(client):
    res = client.post("/api/query", json={"question": ""})
    assert res.status_code == 400


def test_query_no_context(client):
    res = client.post("/api/query", json={"question": "test"})
    assert res.status_code == 400
    assert "No documents" in res.json()["detail"]
