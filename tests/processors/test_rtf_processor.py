import pytest
from src.processors.rtf_processor import RtfProcessor
from langchain.schema import Document
import os
import tempfile
from unittest.mock import patch

@pytest.fixture
def rtf_processor():
    return RtfProcessor()

def test_init(rtf_processor):
    assert rtf_processor.text_splitter is not None

def create_temp_file(content, suffix='.rtf'):
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp.write(content if isinstance(content, bytes) else content.encode())
    temp.close()
    return temp.name

@pytest.mark.asyncio
async def test_process_rtf_file(rtf_processor):
    # Mock textract to avoid dependency on actual RTF files in tests
    with patch('textract.process') as mock_textract:
        test_content = "This is a test RTF document content."
        mock_textract.return_value = test_content.encode('utf-8')

        # Create a temporary RTF file
        file_path = create_temp_file(b"{\\rtf1\\ansi Test RTF content}")

        try:
            chunks = rtf_processor.process(file_path)
            # Verify results
            assert len(chunks) > 0
            assert all(isinstance(chunk, Document) for chunk in chunks)
            # Verify content
            assert test_content in chunks[0].page_content
        finally:
            if os.path.exists(file_path):
                os.unlink(file_path)

@pytest.mark.asyncio
async def test_process_rtf_fileobj(rtf_processor):
    # Test with file-like object that has name attribute
    with patch('textract.process') as mock_textract:
        test_content = "Test content from file object."
        mock_textract.return_value = test_content.encode('utf-8')

        # Create a mock file
        class MockFile:
            def __init__(self, path):
                self.name = path

        file_path = create_temp_file(b"{\\rtf1\\ansi Test content}")
        mock_file = MockFile(file_path)

        try:
            chunks = rtf_processor.process(mock_file)
            assert len(chunks) > 0
            assert test_content in chunks[0].page_content
        finally:
            if os.path.exists(file_path):
                os.unlink(file_path)

@pytest.mark.asyncio
async def test_process_rtf_content(rtf_processor):
    # Test with content directly provided
    with patch('textract.process') as mock_textract:
        test_content = "Direct RTF content test"
        mock_textract.return_value = test_content.encode('utf-8')

        class ContentObject:
            def __init__(self, content):
                self.content = content

            def read(self):
                return self.content

        content_obj = ContentObject(b"{\\rtf1\\ansi Direct content}")

        chunks = rtf_processor.process(content_obj)
        assert len(chunks) > 0
        assert test_content in chunks[0].page_content