import pytest
from pathlib import Path
from src.processors.factory import ProcessorFactory
from src.processors.pdf_processor import PDFProcessor
from src.processors.word_processor import WordProcessor

def test_get_processor_pdf():
    # Test PDF processor
    processor = ProcessorFactory.get_processor("test.pdf")
    assert isinstance(processor, PDFProcessor)

def test_get_processor_doc():
    # Test DOC processor
    processor = ProcessorFactory.get_processor("test.doc")
    assert isinstance(processor, WordProcessor)

def test_get_processor_docx():
    # Test DOCX processor
    processor = ProcessorFactory.get_processor("test.docx")
    assert isinstance(processor, WordProcessor)

def test_get_processor_invalid():
    # Test invalid file type
    with pytest.raises(ValueError) as excinfo:
        ProcessorFactory.get_processor("test.txt")
    assert "Please upload a PDF, DOC, or DOCX file" in str(excinfo.value)

def test_get_processor_with_path_object():
    # Test with Path object
    processor = ProcessorFactory.get_processor(Path("test.pdf"))
    assert isinstance(processor, PDFProcessor)

def test_get_processor_with_file_object():
    # Test with file-like object having name attribute
    class MockFile:
        name = "test.docx"

    processor = ProcessorFactory.get_processor(MockFile())
    assert isinstance(processor, WordProcessor)