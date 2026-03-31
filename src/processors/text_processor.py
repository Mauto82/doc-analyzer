from src.models.document import Document
from src.utils.text_splitter import RecursiveCharacterTextSplitter
from .base.document_processor import DocumentProcessor, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP


class TextProcessor(DocumentProcessor):
    """Plain-text file (.txt, .md, .yaml, …) processor."""

    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=DEFAULT_CHUNK_SIZE,
            chunk_overlap=DEFAULT_CHUNK_OVERLAP,
        )

    def process(self, file_path: str) -> list[Document]:
        with open(file_path, encoding="utf-8", errors="replace") as f:
            text = f.read()

        doc = Document(page_content=text, metadata={"source": file_path})
        return self.text_splitter.split_documents([doc])
