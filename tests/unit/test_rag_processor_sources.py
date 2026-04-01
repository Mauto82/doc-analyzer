"""Tests for structured sources in query_stream."""
import pytest
from unittest.mock import MagicMock
from src.processors.rag_processor import RAGProcessor, DEFAULT_COLLECTION
from src.models.document import Document


@pytest.fixture
def rag():
    mock_ollama = MagicMock()
    embed_resp = MagicMock()
    embed_resp.embeddings = [[0.1] * 1024]
    mock_ollama.embed.return_value = embed_resp
    chunk = MagicMock()
    chunk.message.content = "answer token"
    mock_ollama.chat.return_value = iter([chunk])

    mock_qdrant = MagicMock()
    mock_qdrant.get_collections.return_value = MagicMock(collections=[])

    return RAGProcessor(
        ollama_client=mock_ollama,
        qdrant_client=mock_qdrant,
        model_name="test-model",
        embedding_model="test-embed",
        qdrant_path="./data/test",
    )


def _make_point(file_name="doc.pdf", page=0, score=0.9, content="ctx"):
    p = MagicMock()
    p.payload = {"page_content": content, "file_name": file_name, "page": page}
    p.score = score
    return p


def test_stream_emits_sources_event_first(rag):
    point = _make_point("report.pdf", page=1, score=0.88)
    rag.qdrant.query_points.return_value = MagicMock(points=[point])

    events = list(rag.query_stream("question", [DEFAULT_COLLECTION]))
    assert events[0]["type"] == "sources"
    sources = events[0]["sources"]
    assert len(sources) == 1
    assert sources[0]["file"] == "report.pdf"
    assert sources[0]["page"] == 1
    assert sources[0]["score"] == 0.88


def test_stream_sources_score_rounded(rag):
    point = _make_point(score=0.876543)
    rag.qdrant.query_points.return_value = MagicMock(points=[point])

    events = list(rag.query_stream("q", [DEFAULT_COLLECTION]))
    assert events[0]["sources"][0]["score"] == 0.877


def test_stream_text_events_after_sources(rag):
    point = _make_point()
    rag.qdrant.query_points.return_value = MagicMock(points=[point])

    events = list(rag.query_stream("q", [DEFAULT_COLLECTION]))
    text_events = [e for e in events if e["type"] == "text"]
    assert len(text_events) == 1
    assert text_events[0]["text"] == "answer token"


def test_stream_no_results_no_sources_event(rag):
    rag.qdrant.query_points.return_value = MagicMock(points=[])

    events = list(rag.query_stream("q", [DEFAULT_COLLECTION]))
    types = [e["type"] for e in events]
    assert "sources" not in types
    assert "text" in types
    assert "couldn't find" in events[0]["text"].lower()


def test_stream_falls_back_to_source_field(rag):
    """If file_name is absent in payload, fall back to source field."""
    p = MagicMock()
    p.payload = {"page_content": "ctx", "source": "/path/to/file.txt"}
    p.score = 0.7
    rag.qdrant.query_points.return_value = MagicMock(points=[p])

    events = list(rag.query_stream("q", [DEFAULT_COLLECTION]))
    assert events[0]["sources"][0]["file"] == "/path/to/file.txt"


def test_query_method_still_returns_string(rag):
    point = _make_point()
    rag.qdrant.query_points.return_value = MagicMock(points=[point])

    result = rag.query("question", [DEFAULT_COLLECTION])
    assert result == "answer token"
    assert isinstance(result, str)
