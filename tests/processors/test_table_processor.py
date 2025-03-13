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

        # Check content - should contain CSV text
        csv_content = "ID,Name,Price\n1,Product A,100.0\n2,Product B,200.0\n3,Product C,150.0"
        assert any(csv_content in chunk.page_content for chunk in chunks)

        # Check metadata
        assert chunks[0].metadata['source'] == file_path
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

        # Check content - should contain CSV-like text with the headers and data
        assert any('Date,Resort,Season,Price,Duration' in chunk.page_content for chunk in chunks)
        assert any('Resort A' in chunk.page_content for chunk in chunks)

        # Check metadata
        assert chunks[0].metadata['source'] == file_path
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

        # Verify results
        assert len(chunks) > 0
        assert all(isinstance(chunk, Document) for chunk in chunks)

        # Check that both sheets are represented
        combined_text = ' '.join([chunk.page_content for chunk in chunks])
        assert 'Sheet: Products' in combined_text
        assert 'Sheet: Regions' in combined_text
        assert 'ID,Product,Price' in combined_text
        assert 'Region,Sales' in combined_text
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

        # Check content - should contain CSV-like representation of the JSON
        assert any('id,name,price,duration' in chunk.page_content.lower() for chunk in chunks)
        assert any('tour a' in chunk.page_content.lower() for chunk in chunks)

        # Check metadata
        assert chunks[0].metadata['source'] == file_path
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

        # For nested JSON, it should be processed as formatted JSON
        assert any('example tours' in chunk.page_content.lower() for chunk in chunks)
    finally:
        os.unlink(file_path)

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