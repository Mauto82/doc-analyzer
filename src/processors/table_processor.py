from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from .base.document_processor import DocumentProcessor
import tempfile
import os
import pandas as pd
import numpy as np
import io
from pathlib import Path
import json

def filter_complex_metadata(metadata):
    """
Filter complex metadata types that ChromaDB cannot handle.
Converts complex types like lists, numpy arrays, etc., to strings.

Args:
metadata (dict): The metadata dictionary to filter

Returns:
dict: The filtered metadata with complex types converted to strings
    """
    filtered = {}
    for key, value in metadata.items():
        # Handle None values
        if value is None:
            filtered[key] = ""
        # Convert lists and tuples to strings
        elif isinstance(value, (list, tuple)):
            filtered[key] = json.dumps(value)
        # Convert numpy types to native Python types
        elif hasattr(value, "dtype") and hasattr(value, "tolist"):
            # Convert numpy arrays, numpy.int64, numpy.float64, etc.
            filtered[key] = json.dumps(value.tolist() if hasattr(value, "tolist") else value.item())
        # Convert dictionaries to strings
        elif isinstance(value, dict):
            filtered[key] = json.dumps(value)
        # Keep strings, ints, floats, bools as they are
        elif isinstance(value, (str, int, float, bool)):
            filtered[key] = value
        # Convert everything else to strings
        else:
            filtered[key] = str(value)
    return filtered

class TableProcessor(DocumentProcessor):
    """
Processor for tabular data files like Excel, CSV, ODS and JSON.
Handles structured data preserving the relationship between rows and columns.
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
        self.max_rows_per_chunk = 50  # Default number of rows per chunk

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

            # Process the file based on its extension
            if extension in ['.xlsx', '.xls', '.ods']:
                return self._process_excel_file(file_path)
            elif extension == '.csv':
                return self._process_csv_file(file_path)
            elif extension == '.json':
                return self._process_json_file(file_path)
            else:
                raise ValueError(f"Unsupported file extension: {extension}")

        finally:
            # Clean up temporary file only if we created it
            if created_tmp_file and os.path.exists(file_path):
                os.unlink(file_path)

    def _process_excel_file(self, file_path):
        """
Process Excel or OpenOffice Calc files (.xlsx, .xls, .ods)

Args:
file_path: Path to the Excel file

Returns:
List of Document objects
        """
        # Read the Excel file with pandas
        try:
            # Get all sheet names
            xl = pd.ExcelFile(file_path)
            sheet_names = xl.sheet_names

            all_chunks = []

            # Process each sheet
            for sheet_name in sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                sheet_chunks = self._process_dataframe(df, sheet_name, file_path)
                all_chunks.extend(sheet_chunks)

            return all_chunks

        except Exception as e:
            raise ValueError(f"Error processing Excel file: {str(e)}")

    def _process_csv_file(self, file_path):
        """
Process CSV files

Args:
file_path: Path to the CSV file

Returns:
List of Document objects
        """
        try:
            # Try to read with different encodings and delimiters
            try:
                df = pd.read_csv(file_path)
            except UnicodeDecodeError:
                # Try with different encoding
                df = pd.read_csv(file_path, encoding='latin1')
            except pd.errors.ParserError:
                # Try with different delimiter
                df = pd.read_csv(file_path, sep=';')

            return self._process_dataframe(df, "CSV Data", file_path)

        except Exception as e:
            raise ValueError(f"Error processing CSV file: {str(e)}")

    def _process_json_file(self, file_path):
        """
Process JSON files

Args:
file_path: Path to the JSON file

Returns:
List of Document objects
        """
        try:
            # Load JSON data
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Check if JSON is a list of records (can be converted to DataFrame)
            if isinstance(data, list) and all(isinstance(item, dict) for item in data):
                df = pd.DataFrame(data)
                return self._process_dataframe(df, "JSON Data", file_path)
            else:
                # If complex JSON structure, format as text
                formatted_json = json.dumps(data, indent=2)
                text_chunks = self.text_splitter.split_text(formatted_json)

                # Convert to Document objects
                chunks = []
                for i, chunk in enumerate(text_chunks):
                    doc = Document(
                        page_content=chunk,
                        metadata={
                            "source": file_path,
                            "format": "json",
                            "chunk": i,
                            "total_chunks": len(text_chunks)
                        }
                    )
                    chunks.append(doc)

                return chunks

        except Exception as e:
            raise ValueError(f"Error processing JSON file: {str(e)}")

    def _process_dataframe(self, df, sheet_name, file_path):
        """
Process a pandas DataFrame into document chunks

Args:
df: pandas DataFrame
sheet_name: Name of the sheet or data source
file_path: Original file path

Returns:
List of Document objects
        """
        chunks = []

        # Check if DataFrame is empty
        if df.empty:
            return chunks

        # Calculate basic statistics
        stats = self._calculate_statistics(df)

        # Handle small tables (less than max_rows_per_chunk)
        if len(df) <= self.max_rows_per_chunk:
            content = self._format_dataframe(df, sheet_name, stats)

            # Create metadata and convert lists to strings to avoid ChromaDB issues
            metadata = {
                "source": file_path,
                "sheet_name": sheet_name,
                "format": "table",
                "rows": len(df),
                "columns": len(df.columns),
                "column_names": ",".join(df.columns.tolist()),  # Convert list to string
                "column_names_json": json.dumps(df.columns.tolist()),  # JSON string for recovery if needed
                "statistics": json.dumps(stats)  # Add statistics to metadata as JSON string
            }

            # Filter complex metadata that ChromaDB can't handle
            metadata = filter_complex_metadata(metadata)

            doc = Document(
                page_content=content,
                metadata=metadata
            )
            chunks.append(doc)
        else:
            # Split into chunks
            total_chunks = (len(df) + self.max_rows_per_chunk - 1) // self.max_rows_per_chunk

            for i in range(total_chunks):
                start_idx = i * self.max_rows_per_chunk
                end_idx = min((i + 1) * self.max_rows_per_chunk, len(df))

                df_chunk = df.iloc[start_idx:end_idx].copy()

                # Format this chunk
                chunk_content = self._format_dataframe(
                    df_chunk,
                    f"{sheet_name} (Rows {start_idx+1}-{end_idx} of {len(df)})",
                    stats
                )

                # Create metadata and convert lists to strings
                metadata = {
                    "source": file_path,
                    "sheet_name": sheet_name,
                    "format": "table",
                    "chunk": i,
                    "total_chunks": total_chunks,
                    "start_row": start_idx,
                    "end_row": end_idx,
                    "total_rows": len(df),
                    "columns": len(df.columns),
                    "column_names": ",".join(df.columns.tolist()),  # Convert list to string
                    "column_names_json": json.dumps(df.columns.tolist()),  # JSON string for recovery if needed
                    "statistics": json.dumps(stats)  # Add statistics to metadata as JSON string
                }

                # Filter complex metadata that ChromaDB can't handle
                metadata = filter_complex_metadata(metadata)

                doc = Document(
                    page_content=chunk_content,
                    metadata=metadata
                )
                chunks.append(doc)

        return chunks

    def _format_dataframe(self, df, title, stats=None):
        """
Format a pandas DataFrame as a text representation

Args:
df: pandas DataFrame
title: Title for the data
stats: Optional statistics dictionary

Returns:
String representation of the DataFrame
        """
        # Format column headers
        headers = "Columns: " + ", ".join(df.columns.tolist())

        # Format rows
        rows = []
        for idx, row in df.iterrows():
            # Format individual row values, handling various data types
            row_values = []
            for col, value in row.items():
                if pd.isna(value):
                    formatted_value = "N/A"
                elif isinstance(value, (int, float, bool)):
                    formatted_value = str(value)
                else:
                    formatted_value = str(value)
                row_values.append(f"{col}: {formatted_value}")

            rows.append(f"Row {idx + 1}: {', '.join(row_values)}")

        # Format statistics if available
        stats_text = ""
        if stats:
            stats_lines = [f"Statistics:"]
            for col, col_stats in stats.items():
                if col_stats:
                    # Make sure all values are strings
                    stat_items = []
                    for key, value in col_stats.items():
                        if isinstance(value, list):
                            value = ", ".join(map(str, value))
                        stat_items.append(f"{key}: {value}")
                    stats_lines.append(f"  Column '{col}': {', '.join(stat_items)}")
            stats_text = "\n".join(stats_lines)

        # Combine all parts
        result = f"Table: {title}\n{headers}\n\n" + "\n".join(rows)

        if stats_text:
            result += f"\n\n{stats_text}"

        return result

    def _calculate_statistics(self, df):
        """
Calculate basic statistics for the DataFrame

Args:
df: pandas DataFrame

Returns:
Dictionary with statistics per column as strings (safe for ChromaDB)
        """
        stats = {}

        for col in df.columns:
            col_stats = {}
            series = df[col]

            # Skip statistics for columns with all missing values
            if series.isna().all():
                continue

            # Handle numeric columns
            if pd.api.types.is_numeric_dtype(series):
                # Filter out NaN values for calculations
                valid_values = series.dropna()
                if not valid_values.empty:
                    col_stats['min'] = float(valid_values.min())
                    col_stats['max'] = float(valid_values.max())
                    col_stats['mean'] = float(valid_values.mean())
                    col_stats['median'] = float(valid_values.median())
                    # Convert numpy int64/float64 to Python int/float for JSON serialization
                    col_stats = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                                 for k, v in col_stats.items()}

            # Handle categorical/string columns
            elif pd.api.types.is_string_dtype(series) or pd.api.types.is_categorical_dtype(series):
                value_counts = series.value_counts(dropna=True)
                if not value_counts.empty:
                    # Get most common values (up to 3)
                    most_common = value_counts.nlargest(3)
                    common_values = [f"{val} ({count})" for val, count in zip(most_common.index, most_common.values)]
                    col_stats['most_common'] = ", ".join(common_values)  # Convert list to string

                    # Get unique count
                    col_stats['unique_values'] = series.nunique()

            # Handle date columns
            elif pd.api.types.is_datetime64_dtype(series):
                valid_dates = series.dropna()
                if not valid_dates.empty:
                    col_stats['min_date'] = valid_dates.min().strftime('%Y-%m-%d')
                    col_stats['max_date'] = valid_dates.max().strftime('%Y-%m-%d')
                    col_stats['date_range'] = (valid_dates.max() - valid_dates.min()).days

            # Add non-empty statistics to the result
            if col_stats:
                stats[col] = col_stats

        # Convert stats to a JSON string to ensure it can be stored in ChromaDB
        # We'll include it in the page_content rather than metadata to avoid ChromaDB issues
        return stats

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