import pytest
from src.processors.table_processor import TableProcessor
from langchain.schema import Document
import os
import tempfile
import pandas as pd
import json
from unittest.mock import patch

@pytest.fixture
def table_processor():
    return TableProcessor()

def test_init(table_processor):
    assert table_processor.text_splitter is not None
    assert table_processor.max_rows_per_chunk == 50

def create_temp_csv(data, headers=None):
    """Create a temporary CSV file for testing"""
    temp = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')

    # Create a pandas DataFrame
    if headers:
        df = pd.DataFrame(data, columns=headers)
    else:
        df = pd.DataFrame(data)

    # Write to CSV
    df.to_csv(temp.name, index=False)
    temp.close()
    return temp.name

def create_temp_excel(data, headers=None, sheet_name='Sheet1'):
    """Create a temporary Excel file for testing"""
    temp = tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx')

    # Create a pandas DataFrame
    if headers:
        df = pd.DataFrame(data, columns=headers)
    else:
        df = pd.DataFrame(data)

    # Write to Excel
    with pd.ExcelWriter(temp.name, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)

    temp.close()
    return temp.name

def create_temp_json(data):
    """Create a temporary JSON file for testing"""
    temp = tempfile.NamedTemporaryFile(delete=False, suffix='.json')

    with open(temp.name, 'w') as f:
        json.dump(data, f)

    temp.close()
    return temp.name

def test_process_csv_file(table_processor):
    # Create test data
    data = [
        [1, 'Product A', 100.0],
        [2, 'Product B', 200.0],
        [3, 'Product C', 150.0]
    ]
    headers = ['ID', 'Name', 'Price']

    # Create a temporary CSV file
    file_path = create_temp_csv(data, headers)

    try:
        # Process the file
        chunks = table_processor.process(file_path)

        # Verify results
        assert len(chunks) > 0
        assert all(isinstance(chunk, Document) for chunk in chunks)

        # Check content
        assert "Table: CSV Data" in chunks[0].page_content
        assert "ID: 1, Name: Product A, Price: 100.0" in chunks[0].page_content

        # Check metadata
        assert chunks[0].metadata['format'] == 'table'
        assert chunks[0].metadata['columns'] == 3
        assert 'statistics' in chunks[0].metadata
        # Corretto per confrontare la stringa con la stringa
        assert chunks[0].metadata['column_names'] == 'ID,Name,Price'
    finally:
        os.remove(file_path)

def test_process_excel_file(table_processor):
    # Create test data
    data = [
        ['2023-01-01', 'Resort A', 'Summer', 1200.0, 7],
        ['2023-02-15', 'Resort B', 'Winter', 1800.0, 5],
        ['2023-06-10', 'Resort C', 'Summer', 950.0, 10]
    ]
    headers = ['Date', 'Resort', 'Season', 'Price', 'Duration']

    # Create a temporary Excel file
    file_path = create_temp_excel(data, headers, 'Vacations')

    try:
        # Process the file
        chunks = table_processor.process(file_path)

        # Verify results
        assert len(chunks) > 0
        assert all(isinstance(chunk, Document) for chunk in chunks)

        # Check content
        assert "Table: Vacations" in chunks[0].page_content
        assert "Resort: Resort A" in chunks[0].page_content

        # Check metadata
        assert chunks[0].metadata['format'] == 'table'
        assert chunks[0].metadata['sheet_name'] == 'Vacations'
        assert chunks[0].metadata['columns'] == 5
        assert 'statistics' in chunks[0].metadata
    finally:
        os.unlink(file_path)

def test_process_excel_multiple_sheets(table_processor):
    # Create a temporary Excel file with multiple sheets
    temp = tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx')

    # Create DataFrames for two sheets
    df1 = pd.DataFrame({
        'ID': [1, 2, 3],
        'Product': ['A', 'B', 'C'],
        'Price': [100, 200, 300]
    })

    df2 = pd.DataFrame({
        'Region': ['North', 'South', 'East', 'West'],
        'Sales': [1000, 1500, 800, 1200]
    })

    # Write to Excel with two sheets
    with pd.ExcelWriter(temp.name, engine='openpyxl') as writer:
        df1.to_excel(writer, sheet_name='Products', index=False)
        df2.to_excel(writer, sheet_name='Regions', index=False)

    temp.close()
    file_path = temp.name

    try:
        # Process the file
        chunks = table_processor.process(file_path)

        # Verify results - should have chunks for both sheets
        assert len(chunks) >= 2  # At least one chunk per sheet

        # Check that both sheets are represented
        sheet_names = [chunk.metadata['sheet_name'] for chunk in chunks]
        assert 'Products' in sheet_names
        assert 'Regions' in sheet_names
    finally:
        os.unlink(file_path)

def test_process_json_file(table_processor):
    # Create test data for JSON
    data = [
        {"id": 1, "name": "Tour A", "price": 1200, "duration": 7},
        {"id": 2, "name": "Tour B", "price": 800, "duration": 5},
        {"id": 3, "name": "Tour C", "price": 1500, "duration": 10}
    ]

    # Create a temporary JSON file
    file_path = create_temp_json(data)

    try:
        # Process the file
        chunks = table_processor.process(file_path)

        # Verify results
        assert len(chunks) > 0
        assert all(isinstance(chunk, Document) for chunk in chunks)

        # Check content
        assert "Table: JSON Data" in chunks[0].page_content
        assert "name: Tour A" in chunks[0].page_content

        # Check metadata
        assert chunks[0].metadata['format'] == 'table'
    finally:
        os.unlink(file_path)

def test_process_nested_json_file(table_processor):
    # Create test data with nested structure
    nested_data = {
        "tours": [
            {"id": 1, "name": "Tour A", "price": 1200},
            {"id": 2, "name": "Tour B", "price": 800}
        ],
        "metadata": {
            "company": "Example Tours",
            "year": 2023
        }
    }

    # Create a temporary JSON file
    file_path = create_temp_json(nested_data)

    try:
        # Process the file
        chunks = table_processor.process(file_path)

        # Verify results
        assert len(chunks) > 0
        assert all(isinstance(chunk, Document) for chunk in chunks)

        # For nested JSON, it should be processed as text
        assert "Example Tours" in chunks[0].page_content
    finally:
        os.unlink(file_path)

def test_process_large_table(table_processor):
    # Create a large dataset that will be split into multiple chunks
    # Generate 100 rows (exceeds the default max_rows_per_chunk of 50)
    data = []
    for i in range(100):
        data.append([i, f"Product {i}", i * 10.0])

    headers = ['ID', 'Name', 'Price']

    # Create a temporary CSV file
    file_path = create_temp_csv(data, headers)

    try:
        # Process the file
        chunks = table_processor.process(file_path)

        # Verify the table was split into chunks
        assert len(chunks) > 1

        # Check that chunks have the correct metadata
        assert chunks[0].metadata['total_chunks'] == 2
        assert chunks[0].metadata['chunk'] == 0
        assert chunks[1].metadata['chunk'] == 1

        # Verify row ranges
        assert chunks[0].metadata['start_row'] == 0
        assert chunks[0].metadata['end_row'] == 50
        assert chunks[1].metadata['start_row'] == 50
        assert chunks[1].metadata['end_row'] == 100
    finally:
        os.unlink(file_path)

def test_calculate_statistics(table_processor):
    # Create a DataFrame with different data types
    df = pd.DataFrame({
        'numeric': [10, 20, 30, 40, 50],
        'text': ['A', 'B', 'A', 'C', 'B'],
        'date': pd.to_datetime(['2023-01-01', '2023-02-01', '2023-03-01', '2023-04-01', '2023-05-01']),
        'mixed': [10, 'text', 30, None, 50]  # Column with mixed types
    })

    # Calculate statistics
    stats = table_processor._calculate_statistics(df)

    # Check numeric column statistics
    assert 'numeric' in stats
    assert 'min' in stats['numeric']
    assert 'max' in stats['numeric']
    assert 'mean' in stats['numeric']
    assert stats['numeric']['min'] == 10.0
    assert stats['numeric']['max'] == 50.0
    assert stats['numeric']['mean'] == 30.0

    # Check text column statistics
    assert 'text' in stats
    assert 'most_common' in stats['text']
    assert 'unique_values' in stats['text']
    assert stats['text']['unique_values'] == 3

    # Check date column statistics
    assert 'date' in stats
    assert 'min_date' in stats['date']
    assert 'max_date' in stats['date']
    assert 'date_range' in stats['date']
    assert stats['date']['min_date'] == '2023-01-01'
    assert stats['date']['max_date'] == '2023-05-01'

def test_is_table_file():
    # Test the is_table_file static method
    assert TableProcessor.is_table_file("test.xlsx") == True
    assert TableProcessor.is_table_file("test.xls") == True
    assert TableProcessor.is_table_file("test.csv") == True
    assert TableProcessor.is_table_file("test.ods") == True
    assert TableProcessor.is_table_file("test.json") == True
    assert TableProcessor.is_table_file("test.txt") == False
    assert TableProcessor.is_table_file("test.pdf") == False

    # Test with file-like object
    class MockFile:
        name = "test.csv"

    assert TableProcessor.is_table_file(MockFile()) == True

    class MockFileInvalid:
        name = "test.doc"

    assert TableProcessor.is_table_file(MockFileInvalid()) == False

def test_format_dataframe(table_processor):
    # Create a simple DataFrame
    df = pd.DataFrame({
        'ID': [1, 2, 3],
        'Product': ['A', 'B', 'C'],
        'Price': [100.0, 200.0, 300.0]
    })

    # Sample statistics
    stats = {
        'Price': {'min': 100.0, 'max': 300.0, 'mean': 200.0}
    }

    # Format the DataFrame
    formatted = table_processor._format_dataframe(df, "Test Table", stats)

    # Check the output
    assert "Table: Test Table" in formatted
    assert "Columns: ID, Product, Price" in formatted
    assert "Row 1: ID: 1, Product: A, Price: 100.0" in formatted
    assert "Statistics:" in formatted
    assert "Column 'Price': min: 100.0, max: 300.0, mean: 200.0" in formatted