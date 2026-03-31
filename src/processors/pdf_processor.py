import fitz
from src.models.document import Document
from src.utils.text_splitter import RecursiveCharacterTextSplitter
from .base.document_processor import DocumentProcessor
import tempfile
import os
import io

try:
    import pytesseract
    from PIL import Image
    HAS_OCR = True
except ImportError:
    HAS_OCR = False


class PDFProcessor(DocumentProcessor):
    """PDF document processor with OCR fallback for scanned/vectorized PDFs."""

    OCR_DPI = 300

    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(os.getenv('CHUNK_SIZE', '1000')),
            chunk_overlap=int(os.getenv('CHUNK_OVERLAP', '200'))
        )

    def _extract_text(self, page):
        """Extract text from a page, falling back to OCR if no text found."""
        text = page.get_text().strip()
        if text:
            return text

        # No text — try OCR
        if not HAS_OCR:
            return ""

        pix = page.get_pixmap(dpi=self.OCR_DPI)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        text = pytesseract.image_to_string(img).strip()
        return text

    def process(self, file_obj):
        if isinstance(file_obj, str):
            file_path = file_obj
            created_tmp = False
        elif hasattr(file_obj, 'name') and os.path.exists(file_obj.name):
            file_path = file_obj.name
            created_tmp = False
        else:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
            content = file_obj.read() if hasattr(file_obj, 'read') else file_obj
            if isinstance(content, str):
                content = content.encode('utf-8')
            tmp.write(content)
            tmp.close()
            file_path = tmp.name
            created_tmp = True

        try:
            doc = fitz.open(file_path)
            documents = []
            for page_num, page in enumerate(doc):
                text = self._extract_text(page)
                if text:
                    documents.append(Document(
                        page_content=text,
                        metadata={"source": file_path, "page": page_num},
                    ))
            doc.close()

            return self.text_splitter.split_documents(documents)
        finally:
            if created_tmp and os.path.exists(file_path):
                os.unlink(file_path)
