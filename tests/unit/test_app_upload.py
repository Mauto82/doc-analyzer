import pytest
from unittest.mock import MagicMock, patch


def test_upload_success(app_client):
    client, mock_rag, mock_get_proc = app_client
    res = client.post(
        "/api/upload",
        files={"file": ("doc.txt", b"hello world", "text/plain")},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["file"] == "doc.txt"
    assert data["collection"] == "default"
    assert data["chunks"] == 2  # mock_rag.add_document returns ["id1","id2"]


def test_upload_to_named_collection(app_client):
    client, mock_rag, _ = app_client
    res = client.post(
        "/api/upload",
        files={"file": ("doc.txt", b"hello", "text/plain")},
        data={"collection": "legal"},
    )
    assert res.status_code == 200
    assert res.json()["collection"] == "legal"


def test_upload_duplicate_rejected(app_client):
    client, mock_rag, _ = app_client
    # First upload succeeds
    client.post("/api/upload", files={"file": ("dup.txt", b"a", "text/plain")})
    # Second upload of same file to same collection should 400
    res = client.post("/api/upload", files={"file": ("dup.txt", b"b", "text/plain")})
    assert res.status_code == 400
    assert "already in collection" in res.json()["detail"]


def test_upload_unsupported_type(app_client):
    client, mock_rag, mock_get_proc = app_client
    import src.app
    mock_get_proc.side_effect = ValueError("Unsupported file type")
    res = client.post(
        "/api/upload",
        files={"file": ("file.xyz", b"data", "application/octet-stream")},
    )
    assert res.status_code == 422


def test_upload_empty_content(app_client):
    client, mock_rag, mock_get_proc = app_client
    # processor returns no chunks
    mock_proc = MagicMock()
    mock_proc.process.return_value = []
    mock_get_proc.return_value = mock_proc

    res = client.post(
        "/api/upload",
        files={"file": ("empty.txt", b"", "text/plain")},
    )
    assert res.status_code == 422
    assert "No content extracted" in res.json()["detail"]


def test_upload_processor_exception(app_client):
    client, mock_rag, mock_get_proc = app_client
    mock_proc = MagicMock()
    mock_proc.process.side_effect = RuntimeError("corrupt file")
    mock_get_proc.return_value = mock_proc

    res = client.post(
        "/api/upload",
        files={"file": ("bad.txt", b"x", "text/plain")},
    )
    assert res.status_code == 500
    assert "Error processing" in res.json()["detail"]


def test_upload_tracks_file_in_status(app_client):
    client, mock_rag, _ = app_client
    client.post("/api/upload", files={"file": ("tracked.txt", b"x", "text/plain")})
    status = client.get("/api/status").json()
    assert "tracked.txt" in status["files_map"].get("default", [])
