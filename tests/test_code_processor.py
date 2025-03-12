import pytest
from src.processors.code_processor import CodeProcessor
from langchain.schema import Document
import os
import tempfile

@pytest.fixture
def code_processor():
    return CodeProcessor()

def test_init(code_processor):
    assert code_processor.text_splitter is not None

def create_temp_file(content, suffix='.py'):
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp.write(content if isinstance(content, bytes) else content.encode())
    temp.close()
    return temp.name

def test_process_python_file(code_processor):
    # Create a temporary Python file
    test_content = "def hello_world():\n    print('Hello, World!')\n\nif __name__ == '__main__':\n    hello_world()"
    file_path = create_temp_file(test_content, '.py')

    try:
        chunks = code_processor.process(file_path)
        # Verify results
        assert len(chunks) > 0
        assert all(isinstance(chunk, Document) for chunk in chunks)
        # Verify content
        assert test_content in chunks[0].page_content
        # Verify metadata
        assert chunks[0].metadata['language'] == 'python'
        assert chunks[0].metadata['extension'] == '.py'
    finally:
        os.unlink(file_path)

def test_process_javascript_file(code_processor):
    # Create a temporary JavaScript file
    test_content = "function helloWorld() {\n  console.log('Hello, World!');\n}\n\nhelloWorld();"
    file_path = create_temp_file(test_content, '.js')

    try:
        chunks = code_processor.process(file_path)
        # Verify results
        assert len(chunks) > 0
        assert all(isinstance(chunk, Document) for chunk in chunks)
        # Verify content
        assert test_content in chunks[0].page_content
        # Verify metadata
        assert chunks[0].metadata['language'] == 'javascript'
        assert chunks[0].metadata['extension'] == '.js'
    finally:
        os.unlink(file_path)

def test_process_unknown_extension(code_processor):
    # Create a temporary file with a custom extension that isn't in our map
    test_content = "Custom code content"
    file_path = create_temp_file(test_content, '.custom')

    try:
        chunks = code_processor.process(file_path)
        # Verify results
        assert len(chunks) > 0
        assert all(isinstance(chunk, Document) for chunk in chunks)
        # Verify content
        assert test_content in chunks[0].page_content
        # Verify metadata for unknown extension
        assert chunks[0].metadata['language'] == 'unknown'
        assert chunks[0].metadata['extension'] == '.custom'
    finally:
        os.unlink(file_path)

def test_process_code_fileobj(code_processor):
    # Test with file-like object that has name attribute
    test_content = "package main\n\nfunc main() {\n    println(\"Hello, World!\")\n}"
    file_path = create_temp_file(test_content, '.go')

    class MockFile:
        def __init__(self, path):
            self.name = path

    mock_file = MockFile(file_path)

    try:
        chunks = code_processor.process(mock_file)
        assert len(chunks) > 0
        assert test_content in chunks[0].page_content
        assert chunks[0].metadata['language'] == 'go'
    finally:
        os.unlink(file_path)

def test_process_code_content(code_processor):
    # Test with content directly provided
    class ContentObject:
        def __init__(self, content, name="test.php"):
            self.content = content
            self.name = name

        def read(self):
            return self.content

    test_content = "<?php\necho 'Hello, World!';\n?>"
    content_obj = ContentObject(test_content.encode())

    chunks = code_processor.process(content_obj)
    assert len(chunks) > 0
    assert test_content in chunks[0].page_content
    assert chunks[0].metadata['language'] == 'php'

def test_is_code_file():
    # Test the is_code_file static method
    assert CodeProcessor.is_code_file("test.py") == True
    assert CodeProcessor.is_code_file("test.js") == True
    assert CodeProcessor.is_code_file("test.txt") == False
    assert CodeProcessor.is_code_file("test.pdf") == False

    # Test with file-like object
    class MockFile:
        name = "test.java"

    assert CodeProcessor.is_code_file(MockFile()) == True

    class MockFileInvalid:
        name = "test.doc"

    assert CodeProcessor.is_code_file(MockFileInvalid()) == False