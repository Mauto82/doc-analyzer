from pathlib import Path
from typing import Union

from .base.document_processor import DocumentProcessor
from .pdf_processor import PDFProcessor
from .word_processor import WordProcessor
from .text_processor import TextProcessor
from .rtf_processor import RtfProcessor
from .code_processor import CodeProcessor
from .table_processor import TableProcessor

# ── Registry ──────────────────────────────────────────────────────────────────
# Extension → processor class mapping. Adding a new processor = one new line.
# OCP: open for extension (add entry), closed for modification (no if/elif).

_PROCESSOR_MAP: dict[str, type[DocumentProcessor]] = {
    ".pdf":  PDFProcessor,
    ".doc":  WordProcessor,
    ".docx": WordProcessor,
    ".txt":  TextProcessor,
    ".rtf":  RtfProcessor,
    ".xlsx": TableProcessor,
    ".xls":  TableProcessor,
    ".csv":  TableProcessor,
    ".ods":  TableProcessor,
    ".json": TableProcessor,
}

_UNSUPPORTED_MESSAGE = (
    "Please upload a supported file type: PDF, DOC, DOCX, TXT, RTF, "
    "Excel, CSV, ODS, JSON, Dockerfile, Markdown, YAML, or code file "
    "(e.g., .py, .js, .java, etc.)"
)


def get_processor(file_path: Union[str, Path]) -> DocumentProcessor:
    """
    Return the appropriate DocumentProcessor for the given file path.

    Lookup order:
    1. Exact extension match in the registry
    2. CodeProcessor fallback (handles 40+ languages + Dockerfiles without extension)

    Raises:
        ValueError: if no processor supports the file type.
    """
    path = Path(file_path) if isinstance(file_path, str) else file_path
    # Handle file-like objects that expose a .name attribute
    if hasattr(path, "name"):
        path = Path(path.name)

    ext = path.suffix.lower()

    processor_cls = _PROCESSOR_MAP.get(ext)
    if processor_cls:
        return processor_cls()

    if CodeProcessor.is_code_file(path):
        return CodeProcessor()

    raise ValueError(_UNSUPPORTED_MESSAGE)


# ── Backward-compatible facade ────────────────────────────────────────────────

class ProcessorFactory:
    """Thin facade kept for backward compatibility. Prefer get_processor()."""

    @staticmethod
    def get_processor(file_path: Union[str, Path]) -> DocumentProcessor:
        return get_processor(file_path)
