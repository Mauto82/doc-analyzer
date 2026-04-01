"""
Additional RAGProcessor tests covering paths not in test_rag_processor.py:
- _embed_single truncation retry logic
- reset()
- get_collections()
- query_stream() generator
- rebuild_files_map()
- set_mode() no-op when mode unchanged
"""
import pytest
from unittest.mock import MagicMock, call, patch
from src.processors.rag_processor import RAGProcessor, DEFAULT_COLLECTION
from src.models.document import Document


@pytest.fixture
def rag(mock_ollama_client, mock_qdrant_client):
    return RAGProcessor(
        ollama_client=mock_ollama_client,
        qdrant_client=mock_qdrant_client,
        model_name="test-model",
        embedding_model="test-embed-model",
        qdrant_path="./data/test",
    )


@pytest.fixture
def mock_ollama_client():
    client = MagicMock()
    embed_resp = MagicMock()
    embed_resp.embeddings = [[0.1] * 1024]
    client.embed.return_value = embed_resp
    chunk = MagicMock()
    chunk.message.content = "streamed token"
    client.chat.return_value = iter([chunk])
    client.list.return_value = MagicMock(models=[])
    return client


@pytest.fixture
def mock_qdrant_client():
    client = MagicMock()
    client.get_collections.return_value = MagicMock(collections=[])
    return client


# ── _embed_single ──────────────────────────────────────────────────────────────

def test_embed_single_success(rag):
    result = rag._embed_single("some text")
    assert result == [0.1] * 1024
    rag.ollama_client.embed.assert_called_once_with(
        model="test-embed-model", input=["some text"]
    )


def test_embed_single_truncates_on_context_length_error(rag):
    """On context length error the text is reduced and retried."""
    embed_resp = MagicMock()
    embed_resp.embeddings = [[0.5] * 1024]

    call_count = 0

    def embed_side_effect(model, input):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise Exception("context length exceeded")
        return embed_resp

    rag.ollama_client.embed.side_effect = embed_side_effect
    rag._truncation_factor = 0.5
    rag._max_truncation_attempts = 5

    result = rag._embed_single("x" * 100)
    assert result == [0.5] * 1024
    assert call_count == 2


def test_embed_single_raises_on_non_context_error(rag):
    """Non-context-length errors are re-raised immediately."""
    rag.ollama_client.embed.side_effect = ConnectionError("network down")
    with pytest.raises(ConnectionError, match="network down"):
        rag._embed_single("text")


def test_embed_single_raises_after_max_attempts(rag):
    rag._max_truncation_attempts = 3
    rag._truncation_factor = 0.9
    rag.ollama_client.embed.side_effect = Exception("input length too long")

    with pytest.raises(RuntimeError, match="Failed to embed"):
        rag._embed_single("x" * 200)


def test_embed_single_raises_when_text_becomes_empty(rag):
    rag._max_truncation_attempts = 5
    rag._truncation_factor = 0.0001  # will reduce to empty almost immediately
    rag.ollama_client.embed.side_effect = Exception("context length exceeded")

    with pytest.raises(RuntimeError, match="empty string"):
        rag._embed_single("ab")


# ── get_collections ────────────────────────────────────────────────────────────

def test_get_collections_empty(rag):
    assert rag.get_collections() == []


def test_get_collections_returns_names(rag):
    col = MagicMock()
    col.name = "legal"
    rag.qdrant.get_collections.return_value = MagicMock(collections=[col])
    assert rag.get_collections() == ["legal"]


# ── reset() ───────────────────────────────────────────────────────────────────

def test_reset_memory_mode_creates_new_client(rag):
    with patch("src.processors.rag_processor.QdrantClient") as mock_cls:
        mock_cls.return_value = MagicMock(
            get_collections=MagicMock(return_value=MagicMock(collections=[]))
        )
        rag.reset()
    mock_cls.assert_called_once_with(":memory:")


def test_reset_persist_mode_deletes_collections(rag):
    rag._mode = "persist"
    col_a = MagicMock()
    col_a.name = "col_a"
    col_b = MagicMock()
    col_b.name = "col_b"
    rag.qdrant.get_collections.return_value = MagicMock(collections=[col_a, col_b])

    rag.reset()

    assert rag.qdrant.delete_collection.call_count == 2


# ── set_mode no-op ─────────────────────────────────────────────────────────────

def test_set_mode_noop_when_same(rag):
    assert rag.mode == "memory"
    original_qdrant = rag.qdrant
    rag.set_mode("memory")  # no change
    assert rag.qdrant is original_qdrant  # client not replaced


# ── query_stream() ────────────────────────────────────────────────────────────

def test_query_stream_yields_tokens(rag):
    point = MagicMock()
    point.payload = {"page_content": "context text"}
    point.score = 0.9
    rag.qdrant.query_points.return_value = MagicMock(points=[point])

    chunk1 = MagicMock()
    chunk1.message.content = "Hello"
    chunk2 = MagicMock()
    chunk2.message.content = " world"
    rag.ollama_client.chat.return_value = iter([chunk1, chunk2])

    events = list(rag.query_stream("question", [DEFAULT_COLLECTION]))
    # First event is sources, then text tokens
    text_tokens = [e["text"] for e in events if e.get("type") == "text"]
    assert text_tokens == ["Hello", " world"]


def test_query_stream_no_results_yields_message(rag):
    rag.qdrant.query_points.return_value = MagicMock(points=[])

    events = list(rag.query_stream("question", [DEFAULT_COLLECTION]))
    text_events = [e for e in events if e.get("type") == "text"]
    assert len(text_events) == 1
    assert "couldn't find" in text_events[0]["text"].lower()


def test_query_stream_invalid_role_raises(rag):
    with pytest.raises(ValueError, match="Invalid role"):
        list(rag.query_stream("q", [DEFAULT_COLLECTION], role="nonexistent"))


def test_query_stream_skips_empty_content(rag):
    """Chunks with empty content string are not yielded."""
    point = MagicMock()
    point.payload = {"page_content": "ctx"}
    point.score = 0.9
    rag.qdrant.query_points.return_value = MagicMock(points=[point])

    c1 = MagicMock()
    c1.message.content = ""
    c2 = MagicMock()
    c2.message.content = "answer"
    rag.ollama_client.chat.return_value = iter([c1, c2])

    events = list(rag.query_stream("q", [DEFAULT_COLLECTION]))
    text_tokens = [e["text"] for e in events if e.get("type") == "text"]
    assert text_tokens == ["answer"]


# ── rebuild_files_map() ────────────────────────────────────────────────────────

def test_rebuild_files_map_single_collection(rag):
    col = MagicMock()
    col.name = "default"
    rag.qdrant.get_collections.return_value = MagicMock(collections=[col])

    point = MagicMock()
    point.id = "abc-123"
    point.payload = {"file_name": "doc.pdf", "page_content": "text"}
    # scroll returns (points, next_offset); None offset signals end
    rag.qdrant.scroll.return_value = ([point], None)

    result = rag.rebuild_files_map()
    assert result == {"default": {"doc.pdf": ["abc-123"]}}


def test_rebuild_files_map_falls_back_to_source(rag):
    col = MagicMock()
    col.name = "default"
    rag.qdrant.get_collections.return_value = MagicMock(collections=[col])

    point = MagicMock()
    point.id = "xyz"
    point.payload = {"source": "/path/to/file.txt", "page_content": "text"}
    rag.qdrant.scroll.return_value = ([point], None)

    result = rag.rebuild_files_map()
    assert result == {"default": {"/path/to/file.txt": ["xyz"]}}


def test_rebuild_files_map_pagination(rag):
    col = MagicMock()
    col.name = "col"
    rag.qdrant.get_collections.return_value = MagicMock(collections=[col])

    p1 = MagicMock()
    p1.id = "1"
    p1.payload = {"file_name": "a.pdf"}
    p2 = MagicMock()
    p2.id = "2"
    p2.payload = {"file_name": "b.pdf"}

    # First scroll returns page token, second returns None
    rag.qdrant.scroll.side_effect = [([p1], "cursor"), ([p2], None)]

    result = rag.rebuild_files_map()
    assert "a.pdf" in result["col"]
    assert "b.pdf" in result["col"]
    assert rag.qdrant.scroll.call_count == 2
