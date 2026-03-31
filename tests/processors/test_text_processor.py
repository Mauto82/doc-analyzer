import pytest
from src.processors.text_processor import TextProcessor
from src.models.document import Document
import os
import tempfile

@pytest.fixture
def text_processor():
    return TextProcessor()

def test_init(text_processor):
    assert text_processor.text_splitter is not None

def create_temp_file(content, suffix='.txt'):
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp.write(content if isinstance(content, bytes) else content.encode())
    temp.close()
    return temp.name

def test_process_txt_file(text_processor):
    # Create a temporary text file
    test_content = "This is a test file.\nIt has multiple lines.\nThird line for testing."
    file_path = create_temp_file(test_content)

    try:
        chunks = text_processor.process(file_path)
        # Verify results
        assert len(chunks) > 0
        assert all(isinstance(chunk, Document) for chunk in chunks)
        # Verify content (all content should be in a single chunk for this small text)
        assert test_content in chunks[0].page_content
    finally:
        os.unlink(file_path)

def test_process_utf8_with_special_chars(text_processor):
    test_content = "Accented: àèìòù — symbols: €£"
    file_path = create_temp_file(test_content)
    try:
        chunks = text_processor.process(file_path)
        assert len(chunks) > 0
        assert "àèìòù" in chunks[0].page_content
    finally:
        os.unlink(file_path)


def test_process_empty_file(text_processor):
    file_path = create_temp_file("")
    try:
        # Empty file → no chunks (splitter returns nothing for empty text)
        chunks = text_processor.process(file_path)
        assert isinstance(chunks, list)
    finally:
        os.unlink(file_path)