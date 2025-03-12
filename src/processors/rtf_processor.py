from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from .base.document_processor import DocumentProcessor
import tempfile
import os
import textract

class RtfProcessor(DocumentProcessor):
    """RTF (Rich Text Format) file processor implementation"""

    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(os.getenv('CHUNK_SIZE', '1000')),
            chunk_overlap=int(os.getenv('CHUNK_OVERLAP', '200'))
        )

    def process(self, file_obj):
        # Flag to track if we created a temporary file
        created_tmp_file = False

        # Handle both string paths and file-like objects
        if isinstance(file_obj, str):
            file_path = file_obj
        else:
            # If file_obj has a 'name' attribute and the file exists, use it directly
            if hasattr(file_obj, 'name') and os.path.exists(file_obj.name):
                file_path = file_obj.name
            else:
                # Create a temporary file for the uploaded content
                with tempfile.NamedTemporaryFile(delete=False, suffix='.rtf') as tmp_file:
                    created_tmp_file = True
                    content = file_obj.read() if hasattr(file_obj, 'read') else file_obj
                    if isinstance(content, str):
                        content = content.encode('utf-8')
                    tmp_file.write(content)
                    file_path = tmp_file.name

        try:
            # Extract text using textract
            text = textract.process(file_path).decode('utf-8')

            # Create a single document with the extracted text
            doc = Document(page_content=text, metadata={"source": file_path})

            # Split into chunks
            chunks = self.text_splitter.split_documents([doc])

            return chunks
        finally:
            # Clean up temporary file only if we created it
            if created_tmp_file and os.path.exists(file_path):
                os.unlink(file_path)