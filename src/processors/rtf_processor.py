from src.models.document import Document
from src.utils.text_splitter import RecursiveCharacterTextSplitter
from .base.document_processor import DocumentProcessor, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP

try:
    import textract
except ImportError:
    textract = None


class RtfProcessor(DocumentProcessor):
    """RTF (Rich Text Format) processor."""

    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=DEFAULT_CHUNK_SIZE,
            chunk_overlap=DEFAULT_CHUNK_OVERLAP,
        )

    def process(self, file_path: str) -> list[Document]:
        if textract is None:
            raise ImportError(
                "textract is required to process .rtf files. "
                "Install it with: pip install textract"
            )

        text = textract.process(file_path).decode("utf-8")
        doc = Document(page_content=text, metadata={"source": file_path})
        return self.text_splitter.split_documents([doc])
