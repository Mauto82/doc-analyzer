import pytest
from unittest.mock import MagicMock, patch
from src.processors.rag_processor import RAGProcessor, DEFAULT_COLLECTION
from src.models.document import Document


@pytest.fixture
def rag_processor():
    mock_ollama = MagicMock()

    embed_response = MagicMock()
    embed_response.embeddings = [[0.1] * 1024]
    mock_ollama.embed.return_value = embed_response

    mock_stream_chunk = MagicMock()
    mock_stream_chunk.message.content = "Test response"
    mock_ollama.chat.return_value = iter([mock_stream_chunk])

    model = MagicMock()
    model.model = "test-model"
    mock_ollama.list.return_value = MagicMock(models=[model])

    mock_qdrant = MagicMock()
    mock_qdrant.get_collections.return_value = MagicMock(collections=[])

    processor = RAGProcessor(
        ollama_client=mock_ollama,
        qdrant_client=mock_qdrant,
        model_name="test-model",
        embedding_model="test-embed-model",
        qdrant_path="./data/test",
    )
    yield processor


@pytest.fixture
def sample_chunks():
    return [
        Document(page_content="Test content 1", metadata={"source": "test1.pdf"}),
        Document(page_content="Test content 2", metadata={"source": "test2.docx"}),
        Document(page_content="Test content 3", metadata={"source": "test3.doc"}),
    ]


def test_init(rag_processor):
    assert rag_processor.model_name == 'test-model'
    assert rag_processor.embedding_model == 'test-embed-model'
    assert rag_processor.qdrant is not None
    assert rag_processor.ollama_client is not None
    assert rag_processor.mode == 'memory'


def test_missing_model_env(monkeypatch):
    monkeypatch.delenv("LLM_MODEL", raising=False)
    with pytest.raises(ValueError, match="LLM_MODEL environment variable is not set"):
        RAGProcessor.from_env()


def test_add_document_no_chunks(rag_processor):
    with pytest.raises(ValueError, match="No document chunks provided"):
        rag_processor.add_document([])


def test_add_document(rag_processor, sample_chunks):
    embed_response = MagicMock()
    embed_response.embeddings = [[0.1] * 1024] * len(sample_chunks)
    rag_processor.ollama_client.embed.return_value = embed_response

    ids = rag_processor.add_document(sample_chunks)
    assert len(ids) == 3
    rag_processor.qdrant.upsert.assert_called_once()


def test_add_document_to_named_collection(rag_processor, sample_chunks):
    embed_response = MagicMock()
    embed_response.embeddings = [[0.1] * 1024] * len(sample_chunks)
    rag_processor.ollama_client.embed.return_value = embed_response

    ids = rag_processor.add_document(sample_chunks, collection_name="my-col")
    assert len(ids) == 3
    rag_processor.qdrant.upsert.assert_called_once_with(
        collection_name="my-col", points=unittest_any()
    )


def test_query_returns_answer(rag_processor):
    point = MagicMock()
    point.payload = {"page_content": "Test content 1"}
    point.score = 0.9
    query_response = MagicMock()
    query_response.points = [point]
    rag_processor.qdrant.query_points.return_value = query_response

    response = rag_processor.query("test question", collections=[DEFAULT_COLLECTION])
    assert response == "Test response"
    rag_processor.qdrant.query_points.assert_called_once()
    rag_processor.ollama_client.chat.assert_called_once()


def test_query_default_collection_when_none(rag_processor):
    point = MagicMock()
    point.payload = {"page_content": "Test content"}
    point.score = 0.9
    query_response = MagicMock()
    query_response.points = [point]
    rag_processor.qdrant.query_points.return_value = query_response

    # collections=None should fall back to DEFAULT_COLLECTION
    response = rag_processor.query("test question")
    assert response == "Test response"


def test_query_with_model(rag_processor):
    point = MagicMock()
    point.payload = {"page_content": "Test content"}
    point.score = 0.9
    query_response = MagicMock()
    query_response.points = [point]
    rag_processor.qdrant.query_points.return_value = query_response

    mock_chunk = MagicMock()
    mock_chunk.message.content = "answer"
    rag_processor.ollama_client.chat.return_value = iter([mock_chunk])

    rag_processor.query("test question", collections=[DEFAULT_COLLECTION], model="another-model")
    call_args = rag_processor.ollama_client.chat.call_args
    assert call_args.kwargs['model'] == 'another-model'


def test_query_no_results(rag_processor):
    query_response = MagicMock()
    query_response.points = []
    rag_processor.qdrant.query_points.return_value = query_response

    response = rag_processor.query("test question", collections=[DEFAULT_COLLECTION])
    assert "couldn't find relevant information" in response


def test_query_invalid_role(rag_processor):
    with pytest.raises(ValueError, match="Invalid role"):
        rag_processor.query("test question", collections=[DEFAULT_COLLECTION], role="invalid_role")


def test_query_multi_collection(rag_processor):
    """Results from multiple collections are merged and sorted by score."""
    p1 = MagicMock()
    p1.payload = {"page_content": "Content A"}
    p1.score = 0.8

    p2 = MagicMock()
    p2.payload = {"page_content": "Content B"}
    p2.score = 0.95

    def side_effect(collection_name, **kwargs):
        result = MagicMock()
        result.points = [p1] if collection_name == "col1" else [p2]
        return result

    rag_processor.qdrant.query_points.side_effect = side_effect

    mock_chunk = MagicMock()
    mock_chunk.message.content = "merged answer"
    rag_processor.ollama_client.chat.return_value = iter([mock_chunk])

    response = rag_processor.query("test", collections=["col1", "col2"])
    assert response == "merged answer"
    assert rag_processor.qdrant.query_points.call_count == 2


def test_set_mode_persist(rag_processor, tmp_path):
    rag_processor.qdrant_path = str(tmp_path / "qdrant")
    with patch("src.processors.rag_processor.QdrantClient") as mock_qdrant:
        mock_qclient = MagicMock()
        mock_qdrant.return_value = mock_qclient
        mock_qclient.get_collections.return_value = MagicMock(collections=[])
        rag_processor.set_mode("persist")
        assert rag_processor.mode == "persist"


def test_set_mode_invalid(rag_processor):
    with pytest.raises(ValueError, match="Mode must be"):
        rag_processor.set_mode('invalid')


def test_get_available_models(rag_processor):
    models = rag_processor.get_available_models()
    assert models == ['test-model']

    rag_processor.ollama_client.list.side_effect = Exception("Connection failed")
    models = rag_processor.get_available_models()
    assert models == ['test-model']


def test_create_and_delete_collection(rag_processor):
    rag_processor.qdrant.get_collections.return_value = MagicMock(collections=[])
    rag_processor.create_collection("new-col")
    rag_processor.qdrant.create_collection.assert_called_once()

    rag_processor.delete_collection("new-col")
    rag_processor.qdrant.delete_collection.assert_called_once_with(collection_name="new-col")


# Helper for flexible assertion
class unittest_any:
    def __eq__(self, other): return True
