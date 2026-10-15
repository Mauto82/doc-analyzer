"""Tests for deep mode (map → recursive reduce) in RAGProcessor."""
import pytest
from unittest.mock import MagicMock
from src.processors.rag_processor import RAGProcessor, DEFAULT_COLLECTION
from src.models.document import Document
from src.services.query_router import Route


def _point(content, file_name="doc.pdf", chunk_index=None, page=None):
    p = MagicMock()
    p.payload = {"page_content": content, "file_name": file_name, "page": page}
    if chunk_index is not None:
        p.payload["chunk_index"] = chunk_index
    return p


def _reply(text):
    resp = MagicMock()
    resp.message.content = text
    return resp


def _token(text):
    chunk = MagicMock()
    chunk.message.content = text
    return chunk


def _make_rag(points, map_reply=lambda prompt: "- fact", block_chars=100, **kwargs):
    """
    Build a RAGProcessor whose ollama.chat answers non-streamed calls with
    map_reply(prompt) and streamed calls with a single 'final' token.
    """
    mock_ollama = MagicMock()
    mock_ollama.prompts = []

    def chat(model, messages, stream=False):
        prompt = messages[0]["content"]
        mock_ollama.prompts.append(prompt)
        if stream:
            return iter([_token("final")])
        return _reply(map_reply(prompt))

    mock_ollama.chat.side_effect = chat

    mock_qdrant = MagicMock()
    mock_qdrant.scroll.return_value = (points, None)

    return RAGProcessor(
        ollama_client=mock_ollama,
        qdrant_client=mock_qdrant,
        model_name="test-model",
        embedding_model="test-embed",
        qdrant_path="./data/test",
        deep_block_chars=block_chars,
        deep_max_workers=2,
        **{"router_embeddings": False, **kwargs},
    )


def _map_prompts(rag):
    return [p for p in rag.ollama_client.prompts if p.startswith("You are reading one block")]


def test_add_document_stores_chunk_index():
    rag = _make_rag([])
    rag.ollama_client.embed.return_value = MagicMock(embeddings=[[0.1] * 1024])
    rag.qdrant.get_collections.return_value = MagicMock(collections=[])
    rag.add_document([Document("a", {}), Document("b", {})])

    points = rag.qdrant.upsert.call_args.kwargs["points"]
    assert [p.payload["chunk_index"] for p in points] == [0, 1]


def test_blocks_follow_chunk_order_and_size():
    points = [
        _point("C" * 60, chunk_index=2),
        _point("A" * 60, chunk_index=0),
        _point("B" * 30, chunk_index=1),
    ]
    rag = _make_rag(points, block_chars=100)
    blocks = rag._build_blocks([DEFAULT_COLLECTION])
    assert [text for _, text in blocks] == ["A" * 60 + "\n" + "B" * 30, "C" * 60]


def test_blocks_never_span_files():
    points = [_point("x", file_name="a.pdf"), _point("y", file_name="b.pdf")]
    rag = _make_rag(points, block_chars=1000)
    blocks = rag._build_blocks([DEFAULT_COLLECTION])
    assert sorted(blocks) == [("a.pdf", "x"), ("b.pdf", "y")]


def test_legacy_chunks_fall_back_to_page_order():
    points = [_point("second", page=1), _point("first", page=0)]
    rag = _make_rag(points, block_chars=1000)
    assert rag._build_blocks([DEFAULT_COLLECTION]) == [("doc.pdf", "first\nsecond")]


def test_deep_stream_maps_every_block_then_answers():
    points = [_point("A" * 80, chunk_index=0), _point("B" * 80, chunk_index=1)]
    rag = _make_rag(points, block_chars=100)

    events = list(rag.query_stream_deep("q?", [DEFAULT_COLLECTION]))

    assert len(_map_prompts(rag)) == 2
    progress = [e for e in events if e["type"] == "progress"]
    assert progress[0] == {"type": "progress", "stage": "map", "done": 0, "total": 2}
    assert progress[-1]["done"] == 2
    assert [e["text"] for e in events if e["type"] == "text"] == ["final"]
    sources = next(e for e in events if e["type"] == "sources")["sources"]
    assert sources == [{"file": "doc.pdf", "page": None, "score": 1.0, "blocks": 2}]


def test_deep_stream_skips_irrelevant_blocks():
    points = [_point("MARKER", chunk_index=0), _point("noise", chunk_index=1)]
    rag = _make_rag(
        points,
        map_reply=lambda p: "- yes" if "MARKER" in p else "NONE.",
        block_chars=10,
    )
    events = list(rag.query_stream_deep("q?", [DEFAULT_COLLECTION]))

    sources = next(e for e in events if e["type"] == "sources")["sources"]
    assert sources[0]["blocks"] == 1
    assert sources[0]["score"] == 0.5
    final_prompt = rag.ollama_client.prompts[-1]
    assert "- yes" in final_prompt and "NONE" not in final_prompt


def test_deep_stream_nothing_relevant():
    rag = _make_rag([_point("noise")], map_reply=lambda p: "NONE")
    events = list(rag.query_stream_deep("q?", [DEFAULT_COLLECTION]))
    assert events[-1]["type"] == "text"
    assert "nothing relevant" in events[-1]["text"]
    assert not any(e["type"] == "sources" for e in events)


def test_deep_stream_no_documents():
    rag = _make_rag([])
    events = list(rag.query_stream_deep("q?", [DEFAULT_COLLECTION]))
    assert events == [{"type": "text", "text": (
        "I couldn't find any documents in the selected collections."
    )}]


def test_deep_stream_reduces_when_notes_too_long():
    points = [_point(str(i) * 80, chunk_index=i) for i in range(4)]  # one block each

    def reply(prompt):
        if prompt.startswith("The following are notes"):
            return "merged"
        return "n" * 40  # each map note ~40 chars → 4 notes overflow a 100-char block

    rag = _make_rag(points, map_reply=reply, block_chars=100)
    events = list(rag.query_stream_deep("q?", [DEFAULT_COLLECTION]))

    stages = {e["stage"] for e in events if e["type"] == "progress"}
    assert stages == {"map", "reduce"}
    assert "merged" in rag.ollama_client.prompts[-1]


def test_deep_stream_rejects_too_many_blocks():
    points = [_point("x" * 10, chunk_index=i) for i in range(5)]
    rag = _make_rag(points, block_chars=10, deep_max_blocks=3)
    with pytest.raises(ValueError, match="too large for deep mode"):
        list(rag.query_stream_deep("q?", [DEFAULT_COLLECTION]))


def test_deep_stream_invalid_role():
    rag = _make_rag([_point("x")])
    with pytest.raises(ValueError, match="Invalid role"):
        list(rag.query_stream_deep("q?", [DEFAULT_COLLECTION], role="nope"))


def test_query_deep_returns_joined_text():
    rag = _make_rag([_point("x")])
    assert rag.query("q?", deep=True) == "final"


@pytest.mark.parametrize("note,empty", [
    ("NONE", True), ("none.", True), ("**NONE**", True), ("", True),
    ("- None of the clauses apply after 2024", False), ("- fact", False),
])
def test_is_empty_note(note, empty):
    assert RAGProcessor._is_empty_note(note) is empty


# ── Auto routing ───────────────────────────────────────────────────────────────

def test_auto_routes_targeted_question_to_standard():
    rag = _make_rag([_point("x")])
    rag.query_stream = MagicMock(return_value=iter([{"type": "text", "text": "std"}]))

    events = list(rag.query_stream_auto("Chi è il fornitore?", [DEFAULT_COLLECTION]))

    assert events[0] == {"type": "route", "mode": "standard", "reason": "targeted question"}
    assert events[1] == {"type": "text", "text": "std"}
    assert _map_prompts(rag) == []


def test_auto_routes_summary_to_deep():
    rag = _make_rag([_point("x")])
    events = list(rag.query_stream_auto("Riassumi il documento", [DEFAULT_COLLECTION]))

    assert events[0] == {"type": "route", "mode": "deep", "reason": "summary"}
    assert len(_map_prompts(rag)) == 1
    assert events[-1] == {"type": "text", "text": "final"}


def test_auto_falls_back_to_standard_when_too_large():
    points = [_point("x" * 10, chunk_index=i) for i in range(5)]
    rag = _make_rag(points, block_chars=10, deep_max_blocks=3)
    rag.query_stream = MagicMock(return_value=iter([{"type": "text", "text": "std"}]))

    events = list(rag.query_stream_auto("Riassumi il documento", [DEFAULT_COLLECTION]))

    assert events[0]["mode"] == "standard"
    assert "too large" in events[0]["reason"]
    assert _map_prompts(rag) == []


def test_auto_invalid_role():
    rag = _make_rag([_point("x")])
    with pytest.raises(ValueError, match="Invalid role"):
        list(rag.query_stream_auto("q", [DEFAULT_COLLECTION], role="nope"))


def test_query_mode_auto():
    rag = _make_rag([_point("x")])
    assert rag.query("Riassumi il documento", mode="auto") == "final"


def test_query_invalid_mode():
    rag = _make_rag([_point("x")])
    with pytest.raises(ValueError, match="Invalid mode"):
        rag.query("q", mode="turbo")


def test_auto_uses_embedding_router_when_no_rule_fires():
    rag = _make_rag([_point("x")], router_embeddings=True)
    rag._embedding_router = MagicMock()
    rag._embedding_router.classify.return_value = Route("deep", "similar to list")
    rag._embed_single = MagicMock(return_value=[0.1, 0.2])

    events = list(rag.query_stream_auto("Che scadenze ci sono?", [DEFAULT_COLLECTION]))

    rag._embedding_router.classify.assert_called_once_with([0.1, 0.2])
    assert events[0] == {"type": "route", "mode": "deep", "reason": "similar to list"}
    assert len(_map_prompts(rag)) == 1


def test_auto_reuses_question_embedding_for_standard():
    rag = _make_rag([_point("x")], router_embeddings=True)
    rag._embedding_router = MagicMock()
    rag._embedding_router.classify.return_value = None
    rag._embed_single = MagicMock(return_value=[0.1, 0.2])
    rag.query_stream = MagicMock(return_value=iter([]))

    events = list(rag.query_stream_auto("Chi è il fornitore?", [DEFAULT_COLLECTION]))

    assert events[0]["reason"] == "targeted question"
    assert rag._embed_single.call_count == 1
    assert rag.query_stream.call_args.kwargs["query_embedding"] == [0.1, 0.2]


def test_auto_skips_embedding_router_when_rule_fires():
    rag = _make_rag([_point("x")], router_embeddings=True)
    rag._embedding_router = MagicMock()

    list(rag.query_stream_auto("Riassumi il documento", [DEFAULT_COLLECTION]))

    rag._embedding_router.classify.assert_not_called()


def test_query_stream_uses_given_embedding():
    rag = _make_rag([])
    rag.qdrant.query_points.return_value = MagicMock(points=[])

    list(rag.query_stream("q", [DEFAULT_COLLECTION], query_embedding=[0.3]))

    rag.ollama_client.embed.assert_not_called()
    assert rag.qdrant.query_points.call_args.kwargs["query"] == [0.3]
