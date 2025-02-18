import pytest
from fastapi.testclient import TestClient
from src.app import app
import tempfile
import os
from src.app import process_and_query

client = TestClient(app)

def create_test_file(content, suffix):
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp.write(content if isinstance(content, bytes) else content.encode())
    temp.close()
    return temp.name

@pytest.fixture
def mock_pdf_file():
    file_path = create_test_file(b"%PDF-1.4\nTest PDF content", ".pdf")
    yield file_path
    try:
        os.unlink(file_path)
    except FileNotFoundError:
        pass  # Il file è già stato eliminato dal processore

@pytest.fixture
def mock_docx_file():
    file_path = create_test_file(b"PK\x03\x04\nTest DOCX content", ".docx")
    yield file_path
    try:
        os.unlink(file_path)
    except FileNotFoundError:
        pass  # Il file è già stato eliminato dal processore

@pytest.fixture
def mock_doc_file():
    file_path = create_test_file(b"\xD0\xCF\x11\xe0\nTest DOC content", ".doc")
    yield file_path
    try:
        os.unlink(file_path)
    except FileNotFoundError:
        pass  # Il file è già stato eliminato dal processore

def test_process_and_query_no_file():
    response = process_and_query(None, "test question")
    assert response == "Please upload a document file"

def test_process_and_query_pdf(mock_pdf_file, mocker):
    # Mock the necessary components
    mocker.patch('src.processors.pdf_processor.PyMuPDFLoader.load', return_value=[])
    mocker.patch('src.processors.rag_processor.RAGProcessor.query', return_value="Test response")

    with open(mock_pdf_file, 'rb') as f:
        response = process_and_query(f, "test question")
    assert response == "Test response"

def test_process_and_query_docx(mock_docx_file, mocker):
    # Mock the necessary components
    mocker.patch('docx.Document', return_value=mocker.Mock(paragraphs=[]))
    mocker.patch('src.processors.rag_processor.RAGProcessor.query', return_value="Test response")

    with open(mock_docx_file, 'rb') as f:
        response = process_and_query(f, "test question")
    assert response == "Test response"

def test_process_and_query_doc(mock_doc_file, mocker):
    # Mock the necessary components
    mocker.patch('textract.process', return_value=b"Test content")
    mocker.patch('src.processors.rag_processor.RAGProcessor.query', return_value="Test response")

    with open(mock_doc_file, 'rb') as f:
        response = process_and_query(f, "test question")
    assert response == "Test response"

def test_process_and_query_invalid_file():
    with tempfile.NamedTemporaryFile(suffix=".txt") as tmp:
        tmp.write(b"Test content")
        tmp.seek(0)
        response = process_and_query(tmp, "test question")
        assert "Please upload a PDF, DOC, or DOCX file" in response

def test_process_and_query_error(mock_pdf_file, mocker):
    # Mock to raise an exception
    mocker.patch('src.processors.pdf_processor.PyMuPDFLoader.load',
                 side_effect=Exception("Test error"))

    with open(mock_pdf_file, 'rb') as f:
        response = process_and_query(f, "test question")
    assert "An error occurred: Test error" in response