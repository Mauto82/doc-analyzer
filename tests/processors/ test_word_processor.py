import pytest
from src.processors.word_processor import WordProcessor
from langchain.schema import Document
import os
import tempfile
from unittest.mock import patch, Mock

@pytest.fixture
def word_processor():
    return WordProcessor()

def test_init(word_processor):
    assert word_processor.text_splitter is not None

def create_temp_file(content, suffix):
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp.write(content if isinstance(content, bytes) else content.encode())
    temp.close()
    return temp.name

@pytest.mark.asyncio
async def test_process_docx(word_processor):
    with patch('docx.Document') as mock_docx:
        # Mock document with paragraphs
        mock_doc = Mock()
        mock_doc.paragraphs = [Mock(text="Test paragraph 1"), Mock(text="Test paragraph 2")]
        mock_docx.return_value = mock_doc

        # Create temp file
        file_path = create_temp_file("dummy content", ".docx")
        try:
            chunks = word_processor.process(file_path)
            assert len(chunks) > 0
            assert all(isinstance(chunk, Document) for chunk in chunks)
        finally:
            os.unlink(file_path)

@pytest.mark.asyncio
async def test_process_doc(word_processor):
    with patch('textract.process') as mock_textract:
        mock_textract.return_value = b"Test content from doc file"

        # Create temp file
        file_path = create_temp_file("dummy content", ".doc")
        try:
            chunks = word_processor.process(file_path)
            assert len(chunks) > 0
            assert all(isinstance(chunk, Document) for chunk in chunks)
        finally:
            os.unlink(file_path)

def test_get_suffix(word_processor):
    class MockFile:
        name = "test.docx"

    assert word_processor._get_suffix(MockFile()) == ".docx"

    class MockFileNoName:
        pass

    assert word_processor._get_suffix(MockFileNoName()) == ".docx"