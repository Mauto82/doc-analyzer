import pytest
from pathlib import Path
from src.processors.factory import ProcessorFactory
from src.processors.pdf_processor import PDFProcessor
from src.processors.word_processor import WordProcessor
from src.processors.text_processor import TextProcessor
from src.processors.rtf_processor import RtfProcessor
from src.processors.code_processor import CodeProcessor

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

def test_get_processor_txt():
    # Test TXT processor
    processor = ProcessorFactory.get_processor("test.txt")
    assert isinstance(processor, TextProcessor)

def test_get_processor_rtf():
    # Test RTF processor
    processor = ProcessorFactory.get_processor("test.rtf")
    assert isinstance(processor, RtfProcessor)

def test_get_processor_code_files():
    # Test various code file extensions
    code_extensions = ['.py', '.js', '.java', '.c', '.cpp', '.php']
    for ext in code_extensions:
        processor = ProcessorFactory.get_processor(f"test{ext}")
        assert isinstance(processor, CodeProcessor)

def test_get_processor_invalid():
    # Test invalid file type
    with pytest.raises(ValueError) as excinfo:
        ProcessorFactory.get_processor("test.invalid")
    assert "Please upload a supported file type" in str(excinfo.value)

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

def test_get_processor_with_rtf_file_object():
    # Test with file-like object having .rtf extension
    class MockFile:
        name = "test.rtf"

    processor = ProcessorFactory.get_processor(MockFile())
    assert isinstance(processor, RtfProcessor)

def test_get_processor_with_code_file_object():
    # Test with file-like object having code file extension
    class MockFile:
        name = "test.py"

    processor = ProcessorFactory.get_processor(MockFile())
    assert isinstance(processor, CodeProcessor)