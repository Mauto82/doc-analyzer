import pytest
from src.processors.code_processor import CodeProcessor
from langchain.schema import Document
import os
import tempfile
import shutil

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

def test_process_markdown_file(code_processor):
    # Create a temporary Markdown file
    test_content = "# Test Markdown\n\nThis is a *test* of **markdown** processing."
    file_path = create_temp_file(test_content, '.md')

    try:
        chunks = code_processor.process(file_path)
        # Verify results
        assert len(chunks) > 0
        assert all(isinstance(chunk, Document) for chunk in chunks)
        # Verify content
        assert test_content in chunks[0].page_content
        # Verify metadata
        assert chunks[0].metadata['language'] == 'markdown'
        assert chunks[0].metadata['extension'] == '.md'
    finally:
        os.unlink(file_path)

def test_process_dockerfile(code_processor):
    # Create a temporary Dockerfile
    test_content = "FROM python:3.9-slim\n\nWORKDIR /app\n\nCOPY . .\n\nRUN pip install -r requirements.txt"

    # Create a file named "Dockerfile" with no extension
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, "Dockerfile")

    with open(file_path, 'w') as f:
        f.write(test_content)

    try:
        chunks = code_processor.process(file_path)
        # Verify results
        assert len(chunks) > 0
        assert all(isinstance(chunk, Document) for chunk in chunks)
        # Verify content
        assert test_content in chunks[0].page_content
        # Verify metadata - extension should be empty string for Dockerfile
        assert chunks[0].metadata['language'] == 'dockerfile'
    finally:
        # Clean up
        shutil.rmtree(temp_dir)

def test_process_renamed_dockerfile(code_processor):
    # Create a temporary Dockerfile but save it with a .txt extension
    test_content = "FROM ubuntu:20.04\n\nRUN apt-get update\n\nCMD [\"bash\"]\n\nEXPOSE 80"
    file_path = create_temp_file(test_content, '.txt')

    try:
        chunks = code_processor.process(file_path)
        # Verify results
        assert len(chunks) > 0
        # Verify content
        assert test_content in chunks[0].page_content
        # Verify metadata - should detect it's a Dockerfile even with .txt extension
        assert chunks[0].metadata['language'] == 'dockerfile'
        assert chunks[0].metadata['is_dockerfile'] == True
        # The extension should still be .txt
        assert chunks[0].metadata['extension'] == '.txt'
    finally:
        os.unlink(file_path)

def test_process_not_dockerfile(code_processor):
    # Create a file that is not a Dockerfile
    test_content = "This is a regular text file\nIt has some content\nBut not Dockerfile instructions"
    file_path = create_temp_file(test_content, '.txt')

    try:
        chunks = code_processor.process(file_path)
        # Verify results
        assert len(chunks) > 0
        # Verify it's not detected as a Dockerfile
        assert chunks[0].metadata['language'] != 'dockerfile'
        assert not chunks[0].metadata.get('is_dockerfile', False)
    finally:
        os.unlink(file_path)

def test_is_code_file():
    # Test the is_code_file static method
    assert CodeProcessor.is_code_file("test.py") == True
    assert CodeProcessor.is_code_file("test.js") == True
    assert CodeProcessor.is_code_file("test.md") == True
    assert CodeProcessor.is_code_file("test.yml") == True
    assert CodeProcessor.is_code_file("test.yaml") == True
    assert CodeProcessor.is_code_file("Dockerfile") == True
    assert CodeProcessor.is_code_file("path/to/Dockerfile") == True
    assert CodeProcessor.is_code_file("dockerfile") == True  # Case insensitive
    assert CodeProcessor.is_code_file("test.txt") == False
    assert CodeProcessor.is_code_file("test.pdf") == False

    # Test with file-like object
    class MockFile:
        name = "test.java"

    assert CodeProcessor.is_code_file(MockFile()) == True

    class MockFileInvalid:
        name = "test.doc"

    assert CodeProcessor.is_code_file(MockFileInvalid()) == False

    class MockDockerfile:
        name = "Dockerfile"

    assert CodeProcessor.is_code_file(MockDockerfile()) == True