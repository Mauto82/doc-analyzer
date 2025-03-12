from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from .base.document_processor import DocumentProcessor
import tempfile
import os
from pathlib import Path

class CodeProcessor(DocumentProcessor):
    """
Code file processor implementation for various programming languages.
Handles code files like .py, .js, .java, .c, .cpp, .php, etc.
"""

    # List of supported file extensions for code files
    SUPPORTED_EXTENSIONS = [
        '.py', '.js', '.ts', '.java', '.c', '.cpp', '.h', '.hpp',
        '.cs', '.php', '.go', '.rb', '.rs', '.swift', '.kt',
        '.sh', '.bash', '.ps1', '.sql', '.r', '.scala', '.dart',
        '.html', '.css', '.scss', '.less', '.json', '.xml', '.yaml', '.yml',
        '.lua', '.pl', '.pm', '.groovy', '.tsx', '.jsx', '.vb', '.f90',
        '.clj', '.ex', '.exs'
    ]

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
                # Determine the file extension for the temp file
                if hasattr(file_obj, 'name'):
                    suffix = Path(file_obj.name).suffix.lower()
                else:
                    # Default to .txt if we can't determine
                    suffix = '.txt'

                # Create a temporary file with the appropriate extension
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                    created_tmp_file = True
                    content = file_obj.read() if hasattr(file_obj, 'read') else file_obj
                    if isinstance(content, str):
                        content = content.encode('utf-8')
                    tmp_file.write(content)
                    file_path = tmp_file.name

        try:
            # Determine the file language for metadata based on extension
            extension = Path(file_path).suffix.lower()
            language = self._get_language_from_extension(extension)

            # Read the code file with encoding error handling
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                code_content = f.read()

            # Create a single document with the extracted text
            # Include language and extension in metadata for potential syntax highlighting
            doc = Document(
                page_content=code_content,
                metadata={
                    "source": file_path,
                    "language": language,
                    "extension": extension
                }
            )

            # Split into chunks
            chunks = self.text_splitter.split_documents([doc])

            return chunks
        finally:
            # Clean up temporary file only if we created it
            if created_tmp_file and os.path.exists(file_path):
                os.unlink(file_path)

    def _get_language_from_extension(self, extension):
        """
Map file extension to programming language name.
Used for metadata.
        """
        extension_to_language = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.java': 'java',
            '.c': 'c',
            '.cpp': 'cpp',
            '.h': 'c-header',
            '.hpp': 'cpp-header',
            '.cs': 'csharp',
            '.php': 'php',
            '.go': 'go',
            '.rb': 'ruby',
            '.rs': 'rust',
            '.swift': 'swift',
            '.kt': 'kotlin',
            '.sh': 'bash',
            '.bash': 'bash',
            '.ps1': 'powershell',
            '.sql': 'sql',
            '.r': 'r',
            '.scala': 'scala',
            '.dart': 'dart',
            '.html': 'html',
            '.css': 'css',
            '.scss': 'scss',
            '.less': 'less',
            '.json': 'json',
            '.xml': 'xml',
            '.yaml': 'yaml',
            '.yml': 'yaml',
            '.lua': 'lua',
            '.pl': 'perl',
            '.pm': 'perl-module',
            '.groovy': 'groovy',
            '.tsx': 'typescript-react',
            '.jsx': 'javascript-react',
            '.vb': 'visual-basic',
            '.f90': 'fortran',
            '.clj': 'clojure',
            '.ex': 'elixir',
            '.exs': 'elixir-script'
        }

        return extension_to_language.get(extension, 'unknown')

    @classmethod
    def is_code_file(cls, file_path):
        """
Check if a file is a supported code file based on its extension.

Args:
file_path: Path or file-like object with a name attribute

Returns:
bool: True if the file is a supported code file
        """
        if isinstance(file_path, str):
            extension = Path(file_path).suffix.lower()
        elif hasattr(file_path, 'name'):
            extension = Path(file_path.name).suffix.lower()
        else:
            extension = Path(file_path).suffix.lower()

        return extension in cls.SUPPORTED_EXTENSIONS