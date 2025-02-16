from fastapi.testclient import TestClient
from app import app, process_and_query
import pytest
from unittest.mock import Mock
import io

client = TestClient(app)

def test_process_and_query_no_file():
    result = process_and_query(None, "test question")
    assert "Please upload a PDF file" == result

def test_process_and_query_with_invalid_file():
    # Simula un file caricato
    mock_file = io.BytesIO(b"test content")
    mock_file.name = "test.txt"

    result = process_and_query(mock_file, "test question")
    assert "Please upload a PDF file" in result

def test_successful_query(mocker):
    # Simula un PDF
    mock_file = io.BytesIO(b"%PDF-")
    mock_file.name = "test.pdf"

    mocker.patch('processors.pdf_processor.PDFProcessor.process', return_value=["test chunk"])
    mocker.patch('processors.rag_processor.RAGProcessor.query', return_value="test response")

    result = process_and_query(mock_file, "test question")
    assert result == "test response"