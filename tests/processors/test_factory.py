import pytest
from pathlib import Path
from processors.factory import ProcessorFactory
from processors.pdf_processor import PDFProcessor
from unittest.mock import Mock

def test_get_processor_pdf():
    """Test factory returns PDF processor for .pdf files"""
    mock_file = Mock()
    mock_file.name = "test.pdf"
    processor = ProcessorFactory.get_processor(mock_file)
    assert isinstance(processor, PDFProcessor)

def test_get_processor_unsupported():
    """Test factory raises error for unsupported file types"""
    mock_file = Mock()
    mock_file.name = "test.doc"
    with pytest.raises(ValueError) as exc_info:
        ProcessorFactory.get_processor(mock_file)
    assert "Please upload a PDF file" in str(exc_info.value)