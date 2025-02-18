import pytest
from unittest.mock import Mock, patch
from src.app import process_document, query_document
from src.processors.factory import ProcessorFactory
from langchain.schema import Document

def test_process_document_with_no_file():
    """Test process_document when no file is provided"""
    result = process_document(None)
    assert result == "Please upload a document"

def test_process_document_success():
    """Test successful document processing"""
    # Mock file object and processor
    mock_file = Mock()
    mock_processor = Mock()
    mock_processor.process.return_value = [
        Document(page_content="chunk1", metadata={}),
        Document(page_content="chunk2", metadata={})
    ]

    # Mock both the factory and RAGProcessor
    with patch.object(ProcessorFactory, 'get_processor', return_value=mock_processor), \
         patch('src.app.rag_processor.process_document') as mock_rag:
        result = process_document(mock_file)

    assert result == "Document processed successfully. You can now ask questions."
    mock_processor.process.assert_called_once_with(mock_file)
    mock_rag.assert_called_once()

def test_process_document_invalid_file():
    """Test processing with invalid file type"""
    mock_file = Mock()

    with patch.object(ProcessorFactory, 'get_processor', side_effect=ValueError("Invalid file type")):
        result = process_document(mock_file)

    assert result == "Invalid file type"

def test_query_document_empty_question():
    """Test query with empty question"""
    result = query_document("")
    assert result == "Please enter a question"

def test_query_document_no_document():
    """Test query when no document has been processed"""
    with patch('src.app.rag_processor.query', side_effect=ValueError("Please upload a document before asking questions")):
        result = query_document("What is this about?")

    assert result == "Please upload a document before asking questions"

def test_query_document_success():
    """Test successful document query"""
    test_question = "What is this about?"
    expected_answer = "This is about testing."

    with patch('src.app.rag_processor.query', return_value=expected_answer):
        result = query_document(test_question)

    assert result == expected_answer