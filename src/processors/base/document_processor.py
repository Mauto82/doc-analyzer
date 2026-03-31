from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import List
import os
import tempfile

from src.models.document import Document

# Default chunking config — processors inherit these unless overridden via env.
DEFAULT_CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))


class DocumentProcessor(ABC):
    """
    Base class for all document processors.

    Subclasses implement process() to extract text from a specific file format
    and return a list of Document chunks ready for embedding.
    """

    @abstractmethod
    def process(self, file_path: str) -> List[Document]:
        """
        Process a document and return a list of Document chunks.

        Args:
            file_path: Absolute path to the file on disk.

        Returns:
            Non-empty list of Document chunks with content and metadata.
        """

    @staticmethod
    @contextmanager
    def _temp_path(suffix: str = ""):
        """
        Context manager that creates a named temporary file, yields its path,
        and guarantees cleanup on exit — even on exception.

        Usage:
            with self._temp_path(suffix=".pdf") as path:
                write_content_to(path)
                process(path)
        """
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp_path = tmp.name
        tmp.close()
        try:
            yield tmp_path
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
