from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from .base.document_processor import DocumentProcessor
import tempfile
import os
from pathlib import Path
import pandas as pd

class TableProcessor(DocumentProcessor):
    """
    Processor for tabular data files like Excel, CSV, ODS and JSON.
    Simply converts them to plain text format exactly like TextProcessor.
    """

    # List of supported file extensions for tabular data
    SUPPORTED_EXTENSIONS = [
        '.xlsx', '.xls',  # Excel
        '.csv',           # CSV
        '.ods',           # OpenOffice Calc
        '.json'           # JSON
    ]

    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(os.getenv('CHUNK_SIZE', '1000')),
            chunk_overlap=int(os.getenv('CHUNK_OVERLAP', '200'))
        )

    def process(self, file_obj):
        """
        Process a tabular data file and return document chunks

        Args:
        file_obj: File object or path to the file

        Returns:
        List of Document objects containing the chunked content
        """
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
                    # Default to .csv if we can't determine
                    suffix = '.csv'

                # Create a temporary file with the appropriate extension
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                    created_tmp_file = True
                    content = file_obj.read() if hasattr(file_obj, 'read') else file_obj
                    if isinstance(content, str):
                        content = content.encode('utf-8')
                    tmp_file.write(content)
                    file_path = tmp_file.name

        try:
            # Get the file extension
            extension = Path(file_path).suffix.lower()

            # Convert the tabular data to plain text, just like TextProcessor
            if extension in ['.xlsx', '.xls', '.ods']:
                text_content = self._excel_to_text(file_path)
            elif extension == '.csv':
                text_content = self._csv_to_text(file_path)
            elif extension == '.json':
                text_content = self._json_to_text(file_path)
            else:
                raise ValueError(f"Unsupported file extension: {extension}")

            # Create a document with the text content
            doc = Document(
                page_content=text_content,
                metadata={"source": file_path}
            )

            # Split into chunks
            chunks = self.text_splitter.split_documents([doc])

            return chunks

        finally:
            # Clean up temporary file only if we created it
            if created_tmp_file and os.path.exists(file_path):
                os.unlink(file_path)

    def _excel_to_text(self, file_path):
        """
        Convert Excel file to plain text

        Args:
        file_path: Path to the Excel file

        Returns:
        String representation of the Excel file
        """
        # Get sheet names
        xl = pd.ExcelFile(file_path)
        sheet_names = xl.sheet_names

        all_text = []

        # Process each sheet
        for sheet in sheet_names:
            df = pd.read_excel(file_path, sheet_name=sheet)

            # Add sheet name if there are multiple sheets
            if len(sheet_names) > 1:
                all_text.append(f"Sheet: {sheet}")

            # Convert to CSV string
            csv_string = df.to_csv(index=False)
            all_text.append(csv_string)

            # Add a separator between sheets
            if len(sheet_names) > 1 and sheet != sheet_names[-1]:
                all_text.append("\n---\n")

        return "\n".join(all_text)

    def _csv_to_text(self, file_path):
        """
        Convert CSV file to plain text (simply read the file)

        Args:
        file_path: Path to the CSV file

        Returns:
        String representation of the CSV file
        """
        try:
            # Just read the file content directly
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                return f.read()
        except:
            # If that fails, try with pandas (handles different encodings and delimiters)
            try:
                df = pd.read_csv(file_path)
                return df.to_csv(index=False)
            except:
                try:
                    df = pd.read_csv(file_path, encoding='latin1')
                    return df.to_csv(index=False)
                except:
                    df = pd.read_csv(file_path, sep=';')
                    return df.to_csv(index=False)

    def _json_to_text(self, file_path):
        """
        Convert JSON file to plain text

        Args:
        file_path: Path to the JSON file

        Returns:
        String representation of the JSON file
        """
        import json

        # Read the JSON file
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            data = json.load(f)

        # If it's a list of records, convert to CSV
        if isinstance(data, list) and all(isinstance(item, dict) for item in data):
            df = pd.DataFrame(data)
            return df.to_csv(index=False)
        else:
            # Otherwise, format as pretty JSON
            return json.dumps(data, indent=2)

    @classmethod
    def is_table_file(cls, file_path):
        """
        Check if a file is a supported tabular data file based on its extension

        Args:
        file_path: Path or file-like object with a name attribute

        Returns:
        bool: True if the file is a supported tabular data file
        """
        # Handle string paths
        if isinstance(file_path, str):
            extension = Path(file_path).suffix.lower()
        # Handle file-like objects with name attribute
        elif hasattr(file_path, 'name'):
            extension = Path(file_path.name).suffix.lower()
        # Handle Path objects
        else:
            extension = file_path.suffix.lower()

        return extension in cls.SUPPORTED_EXTENSIONS