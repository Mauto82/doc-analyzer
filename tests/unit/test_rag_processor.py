import pytest
from unittest.mock import MagicMock, patch
from src.processors.rag_processor import RAGProcessor, COLLECTION_NAME
from src.models.document import Document
import os


@pytest.fixture
def rag_processor(monkeypatch):
    monkeypatch.setenv('OLLAMA_HOST', 'localhost')
    monkeypatch.setenv('OLLAMA_PORT', '11434')
    monkeypatch.setenv('LLM_MODEL', 'test-model')
    monkeypatch.setenv('EMBEDDING_MODEL', 'test-embed-model')

    with patch('src.processors.rag_processor.ollama.Client') as mock_ollama, \
         patch('src.processors.rag_processor.QdrantClient') as mock_qdrant:

        # Setup ollama mock
        mock_client = MagicMock()
        mock_ollama.return_value = mock_client

        embed_response = MagicMock()
        embed_response.embeddings = [[0.1] * 1024]
        mock_client.embed.return_value = embed_response

        chat_response = MagicMock()
        chat_response.message.content = "Test response"
        mock_client.chat.return_value = chat_response

        model = MagicMock()
        model.model = "test-model"
        list_response = MagicMock()
        list_response.models = [model]
        mock_client.list.return_value = list_response

        # Setup qdrant mock
        mock_qclient = MagicMock()
        mock_qdrant.return_value = mock_qclient
        collections_response = MagicMock()
        collections_response.collections = []
        mock_qclient.get_collections.return_value = collections_response

        processor = RAGProcessor()
        yield processor


@pytest.fixture
def sample_chunks():
    return [
        Document(page_content="Test content 1", metadata={"source": "test1.pdf"}),
        Document(page_content="Test content 2", metadata={"source": "test2.docx"}),
        Document(page_content="Test content 3", metadata={"source": "test3.doc"}),
    ]


@pytest.fixture
def more_chunks():
    return [
        Document(page_content="Additional content 1", metadata={"source": "test4.pdf"}),
        Document(page_content="Additional content 2", metadata={"source": "test5.docx"}),
    ]


def test_init(rag_processor):
    assert rag_processor.model_name == 'test-model'
    assert rag_processor.embedding_model == 'test-embed-model'
    assert rag_processor.qdrant is not None
    assert rag_processor.ollama_client is not None


def test_missing_model_env(monkeypatch):
    monkeypatch.delenv('LLM_MODEL', raising=False)
    with pytest.raises(ValueError, match="LLM_MODEL environment variable is not set"):
        with patch('src.processors.rag_processor.ollama.Client'), \
             patch('src.processors.rag_processor.QdrantClient'):
            RAGProcessor()


def test_process_document_no_chunks(rag_processor):
    with pytest.raises(ValueError, match="No document chunks provided"):
        rag_processor.process_document([])


def test_add_document_no_chunks(rag_processor):
    with pytest.raises(ValueError, match="No document chunks provided"):
        rag_processor.add_document([])


def test_add_document(rag_processor, sample_chunks):
    # Make embed return enough embeddings for all chunks
    embed_response = MagicMock()
    embed_response.embeddings = [[0.1] * 1024] * len(sample_chunks)
    rag_processor.ollama_client.embed.return_value = embed_response

    ids = rag_processor.add_document(sample_chunks)
    assert len(ids) == 3
    rag_processor.qdrant.upsert.assert_called_once()


def test_process_and_query(rag_processor, sample_chunks):
    # Setup query results
    point = MagicMock()
    point.payload = {"page_content": "Test content 1"}
    query_response = MagicMock()
    query_response.points = [point]
    rag_processor.qdrant.query_points.return_value = query_response

    response = rag_processor.query("test question")
    assert response == "Test response"
    rag_processor.qdrant.query_points.assert_called_once()
    rag_processor.ollama_client.chat.assert_called_once()


def test_query_with_model(rag_processor):
    point = MagicMock()
    point.payload = {"page_content": "Test content"}
    query_response = MagicMock()
    query_response.points = [point]
    rag_processor.qdrant.query_points.return_value = query_response

    rag_processor.query("test question", model="another-model")
    call_args = rag_processor.ollama_client.chat.call_args
    assert call_args.kwargs['model'] == 'another-model'


def test_query_no_results(rag_processor):
    query_response = MagicMock()
    query_response.points = []
    rag_processor.qdrant.query_points.return_value = query_response

    response = rag_processor.query("test question")
    assert "couldn't find relevant information" in response


def test_query_invalid_role(rag_processor):
    with pytest.raises(ValueError, match="Invalid role"):
        rag_processor.query("test question", role="invalid_role")


def test_process_document_with_clean_flag(rag_processor, sample_chunks):
    embed_response = MagicMock()
    embed_response.embeddings = [[0.1] * 1024] * len(sample_chunks)
    rag_processor.ollama_client.embed.return_value = embed_response

    # Process with clean_db=True (default)
    rag_processor.process_document(sample_chunks)
    rag_processor.qdrant.delete_collection.assert_called_once_with(
        collection_name=COLLECTION_NAME
    )

    rag_processor.qdrant.reset_mock()
    collections_response = MagicMock()
    collections_response.collections = []
    rag_processor.qdrant.get_collections.return_value = collections_response

    # Process with clean_db=False
    rag_processor.process_document(sample_chunks, clean_db=False)
    rag_processor.qdrant.delete_collection.assert_not_called()


def test_get_available_models(rag_processor):
    models = rag_processor.get_available_models()
    assert models == ['test-model']

    # Test error handling
    rag_processor.ollama_client.list.side_effect = Exception("Connection failed")
    models = rag_processor.get_available_models()
    assert models == ['test-model']
