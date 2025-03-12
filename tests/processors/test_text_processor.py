import pytest
from src.processors.text_processor import TextProcessor
from langchain.schema import Document
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

def test_process_txt_fileobj(text_processor):
    # Test with file-like object that has name attribute
    test_content = "Test content in a file object."
    file_path = create_temp_file(test_content)

    class MockFile:
        def __init__(self, path):
            self.name = path

    mock_file = MockFile(file_path)

    try:
        chunks = text_processor.process(mock_file)
        assert len(chunks) > 0
        assert test_content in chunks[0].page_content
    finally:
        # Remove file if exists
        if os.path.exists(file_path):
            os.unlink(file_path)

def test_process_txt_content(text_processor):
    # Test with content directly provided
    class ContentObject:
        def __init__(self, content):
            self.content = content

        def read(self):
            return self.content

    test_content = "Direct content test"
    content_obj = ContentObject(test_content.encode())

    chunks = text_processor.process(content_obj)
    assert len(chunks) > 0
    assert test_content in chunks[0].page_content