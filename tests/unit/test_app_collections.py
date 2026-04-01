import pytest
from unittest.mock import MagicMock


# ── Collections ────────────────────────────────────────────────────────────────

def test_get_collections_empty(app_client):
    client, mock_rag, _ = app_client
    mock_rag.get_collections.return_value = []
    res = client.get("/api/collections")
    assert res.status_code == 200
    assert res.json() == {"collections": []}


def test_create_collection(app_client):
    client, mock_rag, _ = app_client
    mock_rag.get_collections.return_value = ["legal"]
    res = client.post("/api/collections", json={"name": "legal"})
    assert res.status_code == 200
    data = res.json()
    assert data["created"] == "legal"
    assert "legal" in data["collections"]


def test_create_collection_empty_name(app_client):
    client, mock_rag, _ = app_client
    res = client.post("/api/collections", json={"name": "   "})
    assert res.status_code == 400
    assert "empty" in res.json()["detail"].lower()


def test_create_collection_strips_whitespace(app_client):
    client, mock_rag, _ = app_client
    mock_rag.get_collections.return_value = ["legal"]
    res = client.post("/api/collections", json={"name": "  legal  "})
    assert res.status_code == 200
    assert res.json()["created"] == "legal"


def test_delete_collection(app_client):
    client, mock_rag, _ = app_client
    mock_rag.get_collections.return_value = []
    res = client.delete("/api/collections/legal")
    assert res.status_code == 200
    data = res.json()
    assert data["deleted"] == "legal"
    mock_rag.delete_collection.assert_called_once_with("legal")


# ── Storage mode ───────────────────────────────────────────────────────────────

def test_set_mode_memory(app_client):
    client, mock_rag, _ = app_client
    mock_rag.mode = "memory"
    mock_rag.get_collections.return_value = []
    res = client.post("/api/mode", json={"mode": "memory"})
    assert res.status_code == 200
    assert res.json()["mode"] == "memory"


def test_set_mode_persist(app_client):
    client, mock_rag, _ = app_client
    mock_rag.mode = "persist"
    mock_rag.get_collections.return_value = ["default"]
    res = client.post("/api/mode", json={"mode": "persist"})
    assert res.status_code == 200
    assert res.json()["mode"] == "persist"


def test_set_mode_invalid(app_client):
    client, mock_rag, _ = app_client
    import src.app
    src.app.document_service._rag.set_mode = MagicMock(
        side_effect=ValueError("Mode must be 'memory' or 'persist'")
    )
    res = client.post("/api/mode", json={"mode": "banana"})
    assert res.status_code == 400
    assert "Mode must be" in res.json()["detail"]


# ── Prompts hot-reload ─────────────────────────────────────────────────────────

def test_reload_prompts(app_client):
    client, mock_rag, _ = app_client
    import src.app
    from unittest.mock import patch
    with patch.object(src.app.prompt_registry, "reload") as mock_reload, \
         patch.object(src.app.prompt_registry, "as_api_list",
                      return_value=[{"key": "default", "name": "General"}]):
        res = client.post("/api/prompts/reload")
    assert res.status_code == 200
    mock_reload.assert_called_once()
    assert res.json()["roles"][0]["key"] == "default"
