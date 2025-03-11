import pytest
from src.processors.rag_processor import RAGProcessor
from langchain.schema import Document
import os

@pytest.fixture
def rag_processor(monkeypatch):
    # Ensure we don't try to connect to Ollama during tests
    monkeypatch.setenv('OLLAMA_HOST', 'localhost')
    monkeypatch.setenv('OLLAMA_PORT', '11434')
    return RAGProcessor()

@pytest.fixture
def sample_chunks():
    return [
        Document(page_content="Test content 1", metadata={"source": "test1.pdf"}),
        Document(page_content="Test content 2", metadata={"source": "test2.docx"}),
        Document(page_content="Test content 3", metadata={"source": "test3.doc"})
    ]

@pytest.fixture
def more_chunks():
    return [
        Document(page_content="Additional content 1", metadata={"source": "test4.pdf"}),
        Document(page_content="Additional content 2", metadata={"source": "test5.docx"}),
    ]

def test_init(rag_processor):
    assert rag_processor.model_name == os.getenv('LLM_MODEL', 'deepseek-r1:14b')
    assert rag_processor.chroma_path == os.getenv('CHROMA_DB_PATH', './data/chroma')
    assert rag_processor.embeddings is not None

def test_process_document_no_chunks(rag_processor):
    with pytest.raises(ValueError) as excinfo:
        rag_processor.process_document([])
    assert "No document chunks provided" in str(excinfo.value)

def test_add_document_no_chunks(rag_processor):
    with pytest.raises(ValueError) as excinfo:
        rag_processor.add_document([])
    assert "No document chunks provided" in str(excinfo.value)

def test_query_without_document(rag_processor, mocker):
    # Mock _ensure_db per evitare connessioni reali
    mocker.patch.object(rag_processor, '_ensure_db')

    # Imposta vectordb a None per simulare nessun documento caricato
    rag_processor.vectordb = None

    # Mock embeddings per evitare chiamate reali a Ollama
    mocker.patch.object(rag_processor, 'embeddings')

    with pytest.raises(ValueError) as excinfo:
        rag_processor.query("test question")
    assert "Please upload a document before asking questions" in str(excinfo.value)

def test_process_and_query(rag_processor, sample_chunks, mocker):
    # Mock embeddings
    mock_embeddings = mocker.patch.object(rag_processor, 'embeddings')
    mock_embeddings.embed_documents.return_value = [[0.1] * 384] * len(sample_chunks)
    mock_embeddings.embed_query.return_value = [0.1] * 384

    # Mock Chroma instance
    mock_chroma = mocker.MagicMock()
    mock_retriever = mocker.MagicMock()
    mock_chroma.as_retriever.return_value = mock_retriever
    mock_retriever.get_relevant_documents.return_value = sample_chunks

    # Mock the _ensure_db method to set our mock vectordb
    mocker.patch.object(rag_processor, '_ensure_db')
    rag_processor.vectordb = mock_chroma

    # Mock ollama client
    mock_ollama_client = mocker.patch('ollama.Client')
    mock_client_instance = mocker.MagicMock()
    mock_ollama_client.return_value = mock_client_instance
    mock_client_instance.chat.return_value = {
        'message': {'content': 'Test response'}
    }

    # Test query
    response = rag_processor.query("test question")
    assert response == "Test response"

    # Verify the retriever was used
    mock_retriever.get_relevant_documents.assert_called_once_with("test question")

def test_add_document(rag_processor, sample_chunks, more_chunks, mocker):
    # Mock embeddings
    mock_embeddings = mocker.patch.object(rag_processor, 'embeddings')

    # Mock Chroma instance
    mock_chroma = mocker.MagicMock()

    # Mock the _ensure_db method to set our mock vectordb
    mocker.patch.object(rag_processor, '_ensure_db')
    rag_processor.vectordb = mock_chroma

    # Add first set of documents
    rag_processor.add_document(sample_chunks)
    mock_chroma.add_documents.assert_called_once_with(sample_chunks)
    mock_chroma.add_documents.reset_mock()

    # Add more documents
    rag_processor.add_document(more_chunks)
    mock_chroma.add_documents.assert_called_once_with(more_chunks)

def test_process_document_with_clean_flag(rag_processor, sample_chunks, mocker):
    # Mock embeddings and vectordb
    mocker.patch.object(rag_processor, 'embeddings')
    mock_chroma = mocker.MagicMock()

    # Mock the database methods
    mocker.patch.object(rag_processor, '_ensure_db')
    mocker.patch.object(rag_processor, '_clean_db')
    rag_processor.vectordb = mock_chroma

    # Process with clean_db=True (default)
    rag_processor.process_document(sample_chunks)
    rag_processor._clean_db.assert_called_once()
    mock_chroma.add_documents.assert_called_once()

    # Reset mocks
    rag_processor._clean_db.reset_mock()
    mock_chroma.add_documents.reset_mock()

    # Process with clean_db=False
    rag_processor.process_document(sample_chunks, clean_db=False)
    rag_processor._clean_db.assert_not_called()
    mock_chroma.add_documents.assert_called_once()

def test_persistence(sample_chunks, mocker):
    # Set PERSIST_VECTORDB to true BEFORE creating RAGProcessor
    mocker.patch.dict(os.environ, {
        'PERSIST_VECTORDB': 'true',
        'OLLAMA_HOST': 'localhost',
        'OLLAMA_PORT': '11434'
    })

    # Create mocked RAGProcessor
    rag_processor = RAGProcessor()

    # Mock embeddings
    mocker.patch.object(rag_processor, 'embeddings')

    # Create mock vectordb
    mock_chroma = mocker.MagicMock()
    mocker.patch.object(rag_processor, '_ensure_db')
    rag_processor.vectordb = mock_chroma

    # Mock _clean_db to avoid issues
    mocker.patch.object(rag_processor, '_clean_db')

    # Process document
    rag_processor.process_document(sample_chunks)

    # Verify persist was called
    mock_chroma.persist.assert_called()

    # Add more documents
    mock_chroma.persist.reset_mock()
    rag_processor.add_document(sample_chunks)

    # Verify persist was called again
    mock_chroma.persist.assert_called_once()