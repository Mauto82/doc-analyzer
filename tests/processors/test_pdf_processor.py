import io
import os
import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from src.processors.pdf_processor import PDFProcessor
from src.models.document import Document


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_page(text=""):
    page = MagicMock()
    page.get_text.return_value = text
    return page


def _make_fitz_doc(pages):
    doc = MagicMock()
    doc.__iter__ = MagicMock(return_value=iter(pages))
    doc.__enter__ = MagicMock(return_value=doc)
    doc.__exit__ = MagicMock(return_value=False)
    doc.close = MagicMock()
    return doc


@pytest.fixture
def processor():
    return PDFProcessor()


# ── _extract_text ──────────────────────────────────────────────────────────────

def test_extract_text_with_content(processor):
    page = _make_page("Hello PDF")
    text = processor._extract_text(page)
    assert text == "Hello PDF"


def test_extract_text_empty_no_ocr(processor):
    """When page has no text and OCR is unavailable, return empty string."""
    page = _make_page("")
    with patch("src.processors.pdf_processor.HAS_OCR", False):
        text = processor._extract_text(page)
    assert text == ""


def test_extract_text_empty_with_ocr(processor):
    """When page has no text, OCR is called and its result returned."""
    page = _make_page("")
    pix = MagicMock()
    pix.tobytes.return_value = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
    page.get_pixmap.return_value = pix

    with patch("src.processors.pdf_processor.HAS_OCR", True), \
         patch("src.processors.pdf_processor.Image") as mock_image, \
         patch("src.processors.pdf_processor.pytesseract") as mock_tess:
        mock_tess.image_to_string.return_value = "OCR result"
        mock_image.open.return_value = MagicMock()

        text = processor._extract_text(page)

    assert text == "OCR result"


# ── process() with string path ─────────────────────────────────────────────────

def test_process_string_path(processor, tmp_path):
    pdf_path = str(tmp_path / "test.pdf")

    pages = [_make_page("Page one content"), _make_page("Page two content")]
    fitz_doc = _make_fitz_doc(pages)

    with patch("src.processors.pdf_processor.fitz.open", return_value=fitz_doc):
        result = processor.process(pdf_path)

    assert len(result) >= 1
    assert all(isinstance(d, Document) for d in result)


def test_process_string_path_empty_pages(processor, tmp_path):
    """PDF with no extractable text returns empty list."""
    pdf_path = str(tmp_path / "empty.pdf")

    pages = [_make_page(""), _make_page("")]
    fitz_doc = _make_fitz_doc(pages)

    with patch("src.processors.pdf_processor.fitz.open", return_value=fitz_doc), \
         patch("src.processors.pdf_processor.HAS_OCR", False):
        result = processor.process(pdf_path)

    assert result == []


# ── process() with file-like objects ──────────────────────────────────────────

def test_process_file_like_with_name(processor, tmp_path):
    real_path = str(tmp_path / "file.pdf")
    open(real_path, "w").close()

    pages = [_make_page("content")]
    fitz_doc = _make_fitz_doc(pages)

    mock_file = MagicMock()
    mock_file.name = real_path

    with patch("src.processors.pdf_processor.fitz.open", return_value=fitz_doc):
        result = processor.process(mock_file)

    assert len(result) >= 1
    # No temp file should be created
    assert not any(
        f for f in os.listdir(tmp_path) if f != "file.pdf"
    )


def test_process_file_like_without_name(processor):
    """File-like object without .name — processor creates and cleans up a temp file."""
    content = b"%PDF-1.4 fake content"
    file_obj = io.BytesIO(content)

    pages = [_make_page("extracted")]
    fitz_doc = _make_fitz_doc(pages)

    with patch("src.processors.pdf_processor.fitz.open", return_value=fitz_doc) as mock_open:
        result = processor.process(file_obj)

    # fitz.open was called with a temp file path (string), not the BytesIO object
    call_arg = mock_open.call_args[0][0]
    assert isinstance(call_arg, str)
    # Temp file should be cleaned up after processing
    assert not os.path.exists(call_arg)


# ── Multi-page PDF ─────────────────────────────────────────────────────────────

def test_process_multi_page(processor, tmp_path):
    pdf_path = str(tmp_path / "multi.pdf")

    pages = [
        _make_page("Chapter one " * 50),
        _make_page("Chapter two " * 50),
        _make_page("Chapter three " * 50),
    ]
    fitz_doc = _make_fitz_doc(pages)

    with patch("src.processors.pdf_processor.fitz.open", return_value=fitz_doc):
        result = processor.process(pdf_path)

    assert len(result) >= 3
    sources = {d.metadata["source"] for d in result}
    assert pdf_path in sources
