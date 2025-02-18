import pytest
from src.processors.rag_processor import RAGProcessor
from langchain.schema import Document
import os

@pytest.fixture
def rag_processor():
    return RAGProcessor()

@pytest.fixture
def sample_chunks():
    return [
        Document(page_content="Test content 1", metadata={"source": "test1.pdf"}),
        Document(page_content="Test content 2", metadata={"source": "test2.docx"}),
        Document(page_content="Test content 3", metadata={"source": "test3.doc"})
    ]

def test_init(rag_processor):
    assert rag_processor.model_name == os.getenv('DEEPSEEK_MODEL', 'deepseek-r1:14b')
    assert rag_processor.chroma_path == os.getenv('CHROMA_DB_PATH', './data/chroma')
    assert rag_processor.embeddings is not None
    assert rag_processor.vectordb is None

def test_query_no_chunks(rag_processor):
    with pytest.raises(ValueError) as excinfo:
        rag_processor.query("test question", [])
    assert "No document chunks provided" in str(excinfo.value)

def test_query_with_chunks(rag_processor, sample_chunks, mocker):
    # Mock embeddings
    mock_embeddings = mocker.patch('langchain_community.embeddings.ollama.OllamaEmbeddings')
    mock_embeddings_instance = mocker.MagicMock()
    mock_embeddings_instance.embed_documents.return_value = [[0.1] * 384] * len(sample_chunks)
    mock_embeddings_instance.embed_query.return_value = [0.1] * 384
    mock_embeddings.return_value = mock_embeddings_instance
    rag_processor.embeddings = mock_embeddings_instance

    # Mock Chroma - CORRETTO IL PERCORSO DEL MOCK
    mock_chroma_class = mocker.patch('langchain_community.vectorstores.chroma.Chroma.from_documents')
    mock_chroma_instance = mocker.MagicMock()
    mock_chroma_class.return_value = mock_chroma_instance
    mock_chroma_instance.as_retriever.return_value.get_relevant_documents.return_value = sample_chunks

    # Mock ollama client
    mock_ollama = mocker.patch('ollama.Client')
    mock_ollama_instance = mocker.MagicMock()
    mock_ollama.return_value = mock_ollama_instance
    mock_ollama_instance.chat.return_value = {
        'message': {'content': 'Test response'}
    }

    response = rag_processor.query("test question", sample_chunks)
    assert response == "Test response"

    # Verify that the chunks were used to create embeddings
    mock_chroma_class.assert_called_once()

    # Verify that the retriever was used
    mock_chroma_instance.as_retriever.return_value.get_relevant_documents.assert_called_once_with("test question")

def test_query_persistence(rag_processor, sample_chunks, mocker):
    # Mock embeddings
    mock_embeddings = mocker.patch('langchain_community.embeddings.ollama.OllamaEmbeddings')
    mock_embeddings_instance = mocker.MagicMock()
    mock_embeddings_instance.embed_documents.return_value = [[0.1] * 384] * len(sample_chunks)
    mock_embeddings_instance.embed_query.return_value = [0.1] * 384
    mock_embeddings.return_value = mock_embeddings_instance
    rag_processor.embeddings = mock_embeddings_instance

    # Test with persistence enabled
    mocker.patch.dict(os.environ, {'PERSIST_VECTORDB': 'true'})

    # Mock Chroma - CORRETTO IL PERCORSO DEL MOCK
    mock_chroma_class = mocker.patch('langchain_community.vectorstores.chroma.Chroma.from_documents')
    mock_chroma_instance = mocker.MagicMock()
    mock_chroma_class.return_value = mock_chroma_instance
    mock_chroma_instance.as_retriever.return_value.get_relevant_documents.return_value = sample_chunks

    mock_ollama = mocker.patch('ollama.Client')
    mock_ollama_instance = mocker.MagicMock()
    mock_ollama.return_value = mock_ollama_instance
    mock_ollama_instance.chat.return_value = {
        'message': {'content': 'Test response'}
    }

    rag_processor.query("test question", sample_chunks)

    # Verify that persist was called
    mock_chroma_instance.persist.assert_called_once()