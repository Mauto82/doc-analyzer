"""
Tests for POST /api/import — URL-based content ingestion endpoint.
WebProcessor is mocked so no real network calls are made.
"""
import pytest
from unittest.mock import MagicMock, patch

from src.models.document import Document


def _make_chunk(text="chunk content"):
    doc = Document(page_content=text, metadata={})
    return doc


# ── Helpers ───────────────────────────────────────────────────────────────────

def _post_import(client, url, collection="default"):
    return client.post(
        "/api/import",
        json={"url": url, "collection": collection},
    )


# ── Success cases ─────────────────────────────────────────────────────────────

class TestImportSuccess:
    def test_import_webpage_success(self, app_client):
        client, mock_rag, _ = app_client
        chunks = [_make_chunk(), _make_chunk()]
        mock_processor = MagicMock()
        mock_processor.process_url.return_value = chunks

        with patch("src.app.WebProcessor", return_value=mock_processor):
            res = _post_import(client, "https://example.com/docs")

        assert res.status_code == 200
        data = res.json()
        assert data["url"] == "https://example.com/docs"
        assert data["collection"] == "default"
        assert data["chunks"] == 2

    def test_import_to_named_collection(self, app_client):
        client, mock_rag, _ = app_client
        mock_processor = MagicMock()
        mock_processor.process_url.return_value = [_make_chunk()]

        with patch("src.app.WebProcessor", return_value=mock_processor):
            res = _post_import(client, "https://example.com/page", collection="legal")

        assert res.status_code == 200
        assert res.json()["collection"] == "legal"

    def test_import_file_name_set_to_url(self, app_client):
        """Each chunk's metadata.file_name must be the URL (for dedup tracking)."""
        client, mock_rag, _ = app_client
        chunk = _make_chunk()
        mock_processor = MagicMock()
        mock_processor.process_url.return_value = [chunk]

        url = "https://example.com/api-reference"
        with patch("src.app.WebProcessor", return_value=mock_processor):
            _post_import(client, url)

        assert chunk.metadata["file_name"] == url

    def test_import_appears_in_status(self, app_client):
        client, mock_rag, _ = app_client
        mock_processor = MagicMock()
        mock_processor.process_url.return_value = [_make_chunk()]

        with patch("src.app.WebProcessor", return_value=mock_processor):
            _post_import(client, "https://example.com/page")

        status = client.get("/api/status").json()
        assert "https://example.com/page" in status["files_map"].get("default", [])


# ── Validation / error cases ──────────────────────────────────────────────────

class TestImportErrors:
    def test_empty_url_rejected(self, app_client):
        client, *_ = app_client
        res = _post_import(client, "   ")
        assert res.status_code == 400
        assert "empty" in res.json()["detail"].lower()

    def test_duplicate_url_rejected(self, app_client):
        client, mock_rag, _ = app_client
        mock_processor = MagicMock()
        mock_processor.process_url.return_value = [_make_chunk()]
        url = "https://example.com/dup"

        with patch("src.app.WebProcessor", return_value=mock_processor):
            _post_import(client, url)  # first import
            res = _post_import(client, url)  # duplicate

        assert res.status_code == 400
        assert "already in collection" in res.json()["detail"]

    def test_no_content_extracted_returns_422(self, app_client):
        client, *_ = app_client
        mock_processor = MagicMock()
        mock_processor.process_url.return_value = []

        with patch("src.app.WebProcessor", return_value=mock_processor):
            res = _post_import(client, "https://empty.example.com")

        assert res.status_code == 422
        assert "No content extracted" in res.json()["detail"]

    def test_value_error_from_processor_returns_422(self, app_client):
        client, *_ = app_client
        mock_processor = MagicMock()
        mock_processor.process_url.side_effect = ValueError("No transcript available")

        with patch("src.app.WebProcessor", return_value=mock_processor):
            res = _post_import(client, "https://www.youtube.com/watch?v=abc12345678")

        assert res.status_code == 422
        assert "No transcript available" in res.json()["detail"]

    def test_unexpected_exception_returns_500(self, app_client):
        client, *_ = app_client
        mock_processor = MagicMock()
        mock_processor.process_url.side_effect = RuntimeError("network timeout")

        with patch("src.app.WebProcessor", return_value=mock_processor):
            res = _post_import(client, "https://flaky.example.com")

        assert res.status_code == 500
        assert "Error importing" in res.json()["detail"]

    def test_fetch_failure_propagates_as_422(self, app_client):
        client, *_ = app_client
        mock_processor = MagicMock()
        mock_processor.process_url.side_effect = ValueError("Could not fetch content from")

        with patch("src.app.WebProcessor", return_value=mock_processor):
            res = _post_import(client, "https://unreachable.example.com")

        assert res.status_code == 422

    def test_missing_url_field_returns_422(self, app_client):
        client, *_ = app_client
        res = client.post("/api/import", json={"collection": "default"})
        assert res.status_code == 422  # pydantic validation error


# ── Auth ──────────────────────────────────────────────────────────────────────

class TestImportAuth:
    def test_unauthenticated_returns_401(self, app_client):
        client, *_ = app_client
        res = client.post(
            "/api/import",
            json={"url": "https://example.com"},
            auth=None,
        )
        # TestClient uses default auth from app_client fixture; override with bad creds
        import importlib
        import src.app
        from fastapi.testclient import TestClient
        bare_client = TestClient(src.app.app, raise_server_exceptions=False)
        r = bare_client.post("/api/import", json={"url": "https://example.com"})
        assert r.status_code == 401
