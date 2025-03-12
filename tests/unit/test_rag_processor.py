import pytest
from src.processors.rag_processor import RAGProcessor
from langchain.schema import Document
import os

@pytest.fixture
def rag_processor(monkeypatch):
    # Set required environment variables for tests
    monkeypatch.setenv('OLLAMA_HOST', 'localhost')
    monkeypatch.setenv('OLLAMA_PORT', '11434')
    monkeypatch.setenv('LLM_MODEL', 'test-model')  # Explicitly set model for tests
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
    assert rag_processor.model_name == 'test-model'  # Should match our test environment value
    assert rag_processor.chroma_path == os.getenv('CHROMA_DB_PATH', './data/chroma')
    assert rag_processor.embeddings is not None

def test_missing_model_env():
    # Test that RAGProcessor raises an error when LLM_MODEL is not set
    with pytest.raises(ValueError) as excinfo:
        # Temporarily clear LLM_MODEL from environment
        if 'LLM_MODEL' in os.environ:
            old_model = os.environ['LLM_MODEL']
            del os.environ['LLM_MODEL']
            try:
                RAGProcessor()
            finally:
                # Restore environment
                os.environ['LLM_MODEL'] = old_model
        else:
            # If LLM_MODEL wasn't in environment to begin with
            RAGProcessor()

    assert "LLM_MODEL environment variable is not set" in str(excinfo.value)

def test_process_document_no_chunks(rag_processor):
    with pytest.raises(ValueError) as excinfo:
        rag_processor.process_document([])
    assert "No document chunks provided" in str(excinfo.value)

def test_add_document_no_chunks(rag_processor):
    with pytest.raises(ValueError) as excinfo:
        rag_processor.add_document([])
    assert "No document chunks provided" in str(excinfo.value)

def test_query_without_document(rag_processor, mocker):
    # Mock _ensure_db to prevent real connections
    mocker.patch.object(rag_processor, '_ensure_db')

    # Set vectordb to None to simulate no document loaded
    rag_processor.vectordb = None

    # Mock embeddings to avoid real calls to Ollama
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

    # Test query with default model
    response = rag_processor.query("test question")
    assert response == "Test response"
    # Verify the retriever was used
    mock_retriever.get_relevant_documents.assert_called_once_with("test question")
    # Verify the default model was used
    mock_client_instance.chat.assert_called_with(
        model='test-model',
        messages=[{'role': 'user', 'content': mocker.ANY}]
    )
    mock_client_instance.chat.reset_mock()

    # Test query with specified model
    response = rag_processor.query("test question", model="another-model")
    assert response == "Test response"
    # Verify the specified model was used
    mock_client_instance.chat.assert_called_with(
        model='another-model',
        messages=[{'role': 'user', 'content': mocker.ANY}]
    )

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

def test_persistence(monkeypatch, sample_chunks, mocker):
    # Set environment variables BEFORE creating RAGProcessor
    monkeypatch.setenv('PERSIST_VECTORDB', 'true')
    monkeypatch.setenv('OLLAMA_HOST', 'localhost')
    monkeypatch.setenv('OLLAMA_PORT', '11434')
    monkeypatch.setenv('LLM_MODEL', 'test-model')

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

def test_get_available_models(rag_processor, mocker):
    # Mock ollama client
    mock_ollama_client = mocker.patch('ollama.Client')
    mock_client_instance = mocker.MagicMock()
    mock_ollama_client.return_value = mock_client_instance

    # Set up the mock response for list()
    mock_client_instance.list.return_value = {
        'models': [
            {'name': 'model1'},
            {'name': 'model2'},
            {'name': 'test-model'}
        ]
    }

    models = rag_processor.get_available_models()
    assert models == ['model1', 'model2', 'test-model']

    # Test error handling
    mock_client_instance.list.side_effect = Exception("Connection failed")
    models = rag_processor.get_available_models()
    assert models == ['test-model']