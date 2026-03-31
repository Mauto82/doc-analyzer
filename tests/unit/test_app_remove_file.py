import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


@pytest.fixture
def app_with_mocks(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("OLLAMA_HOST", "localhost")
    monkeypatch.setenv("OLLAMA_PORT", "11434")
    monkeypatch.setenv("EMBEDDING_MODEL", "test-embed-model")

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

        # Inject a fully-mocked RAGProcessor and reset service state
        mock_rag = MagicMock()
        mock_rag.mode = "memory"
        mock_rag.get_collections.return_value = []
        mock_rag.add_document.return_value = ["id1", "id2"]
        mock_rag.remove_document.return_value = True
        mock_rag.query.return_value = "Test response"

        src.app.rag_processor = mock_rag
        src.app.document_service._rag = mock_rag
        src.app.document_service._files_map = {}

        mock_factory = MagicMock()
        mock_processor = MagicMock()
        mock_chunk = MagicMock()
        mock_chunk.page_content = "chunk"
        mock_chunk.metadata = {}
        mock_processor.process.return_value = [mock_chunk]
        mock_factory.return_value = mock_processor

        with patch.object(src.app, "get_processor", mock_factory):
            yield TestClient(src.app.app), mock_rag


def test_add_and_remove_file(app_with_mocks):
    client, mock_rag = app_with_mocks

    res = client.post(
        "/api/upload",
        files={"file": ("test.txt", b"test content", "text/plain")},
    )
    assert res.status_code == 200
    assert res.json()["file"] == "test.txt"
    assert res.json()["collection"] == "default"

    status = client.get("/api/status").json()
    assert "test.txt" in status["files_map"].get("default", [])

    res = client.delete("/api/collections/default/files/test.txt")
    assert res.status_code == 200
    mock_rag.remove_document.assert_called_once_with(
        ["id1", "id2"], collection_name="default"
    )

    status = client.get("/api/status").json()
    assert "test.txt" not in status["files_map"].get("default", [])


def test_remove_nonexistent_file(app_with_mocks):
    client, mock_rag = app_with_mocks
    res = client.delete("/api/collections/default/files/nonexistent.txt")
    assert res.status_code == 404
    mock_rag.remove_document.assert_not_called()


def test_query_after_remove(app_with_mocks):
    client, mock_rag = app_with_mocks

    client.post(
        "/api/upload",
        files={"file": ("test1.txt", b"content1", "text/plain")},
    )
    mock_rag.add_document.return_value = ["id3", "id4"]
    client.post(
        "/api/upload",
        files={"file": ("test2.txt", b"content2", "text/plain")},
    )

    client.delete("/api/collections/default/files/test1.txt")

    res = client.post("/api/query", json={"question": "test", "role": "default"})
    assert res.status_code == 200
    assert res.json()["answer"] == "Test response"
