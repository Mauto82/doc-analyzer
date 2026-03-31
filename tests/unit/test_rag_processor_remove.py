import pytest
from unittest.mock import MagicMock, patch
from src.processors.rag_processor import RAGProcessor
from src.models.document import Document


@pytest.fixture
def rag_processor(monkeypatch):
    monkeypatch.setenv('OLLAMA_HOST', 'localhost')
    monkeypatch.setenv('OLLAMA_PORT', '11434')
    monkeypatch.setenv('LLM_MODEL', 'test-model')
    monkeypatch.setenv('EMBEDDING_MODEL', 'test-embed-model')

    with patch('src.processors.rag_processor.ollama.Client') as mock_ollama, \
         patch('src.processors.rag_processor.QdrantClient') as mock_qdrant:

        mock_client = MagicMock()
        mock_ollama.return_value = mock_client

        embed_response = MagicMock()
        embed_response.embeddings = [[0.1] * 1024] * 3
        mock_client.embed.return_value = embed_response

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


def test_remove_document(rag_processor, sample_chunks):
    ids = rag_processor.add_document(sample_chunks)
    assert len(ids) == 3

    result = rag_processor.remove_document(ids[:2])
    assert result is True
    rag_processor.qdrant.delete.assert_called_once()


def test_remove_document_empty_ids(rag_processor):
    assert rag_processor.remove_document([]) is False
    assert rag_processor.remove_document(None) is False


def test_remove_document_error(rag_processor):
    rag_processor.qdrant.delete.side_effect = Exception("Delete failed")
    result = rag_processor.remove_document(['id1'])
    assert result is False
