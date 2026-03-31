import pytest
from src.processors.rtf_processor import RtfProcessor
from src.models.document import Document
import os
import tempfile
from unittest.mock import patch, MagicMock

@pytest.fixture
def rtf_processor():
    return RtfProcessor()

def create_temp_file(content, suffix='.rtf'):
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp.write(content if isinstance(content, bytes) else content.encode())
    temp.close()
    return temp.name

def _mock_textract(test_content):
    """Create a mock textract module with process() returning the given content."""
    mock = MagicMock()
    mock.process.return_value = test_content.encode('utf-8')
    return mock

@pytest.mark.asyncio
async def test_process_rtf_file(rtf_processor):
    test_content = "This is a test RTF document content."
    mock = _mock_textract(test_content)

    with patch.object(
        __import__('src.processors.rtf_processor', fromlist=['rtf_processor']),
        'textract', mock,
    ):
        file_path = create_temp_file(b"{\\rtf1\\ansi Test RTF content}")
        try:
            chunks = rtf_processor.process(file_path)
            assert len(chunks) > 0
            assert all(isinstance(chunk, Document) for chunk in chunks)
            assert test_content in chunks[0].page_content
        finally:
            if os.path.exists(file_path):
                os.unlink(file_path)

@pytest.mark.asyncio
async def test_process_rtf_fileobj(rtf_processor):
    test_content = "Test content from file object."
    mock = _mock_textract(test_content)

    with patch.object(
        __import__('src.processors.rtf_processor', fromlist=['rtf_processor']),
        'textract', mock,
    ):
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
    test_content = "Direct RTF content test"
    mock = _mock_textract(test_content)

    with patch.object(
        __import__('src.processors.rtf_processor', fromlist=['rtf_processor']),
        'textract', mock,
    ):
        class ContentObject:
            def __init__(self, content):
                self.content = content

            def read(self):
                return self.content

        content_obj = ContentObject(b"{\\rtf1\\ansi Direct content}")

        chunks = rtf_processor.process(content_obj)
        assert len(chunks) > 0
        assert test_content in chunks[0].page_content
