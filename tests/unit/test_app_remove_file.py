import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
import io


@pytest.fixture
def app_with_mocks(monkeypatch):
    monkeypatch.setenv('LLM_MODEL', 'test-model')
    monkeypatch.setenv('OLLAMA_HOST', 'localhost')
    monkeypatch.setenv('OLLAMA_PORT', '11434')
    monkeypatch.setenv('EMBEDDING_MODEL', 'test-embed-model')

    with patch('src.processors.rag_processor.ollama.Client') as mock_ollama, \
         patch('src.processors.rag_processor.QdrantClient') as mock_qdrant:
        mock_qclient = MagicMock()
        mock_qdrant.return_value = mock_qclient
        mock_qclient.get_collections.return_value = MagicMock(collections=[])
        mock_ollama_inst = MagicMock()
        mock_ollama.return_value = mock_ollama_inst
        mock_ollama_inst.list.return_value = MagicMock(models=[])

        import importlib
        import src.app
        importlib.reload(src.app)
        src.app.processed_files_map = {}

        # Mock processor and rag_processor for upload
        mock_rag = MagicMock()
        mock_rag.add_document.return_value = ['id1', 'id2']
        mock_rag.remove_document.return_value = True
        mock_rag.query.return_value = "Test response"
        src.app.rag_processor = mock_rag

        mock_factory = MagicMock()
        mock_processor = MagicMock()
        mock_processor.process.return_value = [MagicMock(page_content="chunk")]
        mock_factory.get_processor.return_value = mock_processor

        with patch.object(src.app, 'ProcessorFactory', mock_factory):
            yield TestClient(src.app.app), mock_rag


def test_add_and_remove_file(app_with_mocks):
    client, mock_rag = app_with_mocks

    # Upload
    res = client.post("/api/upload", files={"file": ("test.txt", b"test content", "text/plain")})
    assert res.status_code == 200
    assert res.json()["file"] == "test.txt"

    # Check status
    status = client.get("/api/status").json()
    assert "test.txt" in status["files"]

    # Remove
    res = client.delete("/api/files/test.txt")
    assert res.status_code == 200
    mock_rag.remove_document.assert_called_once_with(['id1', 'id2'])

    # Check removed
    status = client.get("/api/status").json()
    assert "test.txt" not in status["files"]


def test_remove_nonexistent_file(app_with_mocks):
    client, mock_rag = app_with_mocks
    res = client.delete("/api/files/nonexistent.txt")
    assert res.status_code == 404
    mock_rag.remove_document.assert_not_called()


def test_query_after_remove(app_with_mocks):
    client, mock_rag = app_with_mocks

    # Upload two files
    client.post("/api/upload", files={"file": ("test1.txt", b"content1", "text/plain")})
    mock_rag.add_document.return_value = ['id3', 'id4']
    client.post("/api/upload", files={"file": ("test2.txt", b"content2", "text/plain")})

    # Remove one
    client.delete("/api/files/test1.txt")

    # Query should still work
    res = client.post("/api/query", json={"question": "test", "role": "default"})
    assert res.status_code == 200
    assert res.json()["answer"] == "Test response"
