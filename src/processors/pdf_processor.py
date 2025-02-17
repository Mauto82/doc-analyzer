from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from .base.document_processor import DocumentProcessor
import tempfile
import os

class PDFProcessor(DocumentProcessor):
    """PDF document processor implementation"""

    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(os.getenv('CHUNK_SIZE', '1000')),
            chunk_overlap=int(os.getenv('CHUNK_OVERLAP', '200'))
        )

    def process(self, file_obj):
        # Handle both string paths and file-like objects from Gradio
        if isinstance(file_obj, str):
            file_path = file_obj
        else:
            # Create a temporary file for the uploaded content
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                # If file_obj has a 'name' attribute (from Gradio), use it directly
                if hasattr(file_obj, 'name'):
                    file_path = file_obj.name
                else:
                    # Otherwise, write the content to a temporary file
                    content = file_obj.read() if hasattr(file_obj, 'read') else file_obj
                    if isinstance(content, str):
                        content = content.encode('utf-8')
                    tmp_file.write(content)
                    file_path = tmp_file.name

        try:
            # Load the PDF
            loader = PyMuPDFLoader(file_path)
            documents = loader.load()

            # Split into chunks
            chunks = self.text_splitter.split_documents(documents)

            return chunks
        finally:
            # Clean up the temporary file only if we created one
            if 'tmp_file' in locals():
                os.unlink(file_path)