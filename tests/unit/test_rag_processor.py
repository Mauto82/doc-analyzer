import pytest
from unittest.mock import patch, Mock, MagicMock
from processors.rag_processor import RAGProcessor
from langchain.schema import Document

def test_query_with_empty_chunks():
    processor = RAGProcessor()
    with pytest.raises(ValueError) as exc_info:
        processor.query("Test question", [])
    assert "No document chunks provided" in str(exc_info)

def test_successful_query(mock_ollama_client, mock_vectordb):
    with patch('processors.rag_processor.Chroma') as mock_chroma, \
         patch('processors.rag_processor.ollama.Client') as mock_client:

        mock_client.return_value = mock_ollama_client
        mock_chroma.from_documents.return_value = mock_vectordb
        mock_vectordb.as_retriever.return_value.get_relevant_documents.return_value = [
            Document(page_content="Test content")
        ]

        processor = RAGProcessor()
        result = processor.query("Test question", [Document(page_content="Test content")])

        assert result == "Test response"