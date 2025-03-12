import pytest
from src.app import add_file_to_context, remove_file_from_context, query_document
from src.processors.rag_processor import RAGProcessor
import os

# Mock file object
class MockFile:
    def __init__(self, name):
        self.name = name

@pytest.fixture
def setup_context(monkeypatch, mocker):
    # Reset global state
    import src.app
    src.app.processed_files_map = {}

    # Mock RAGProcessor and its methods
    mock_rag = mocker.patch.object(src.app, 'rag_processor')
    mock_rag.add_document.return_value = ['id1', 'id2']
    mock_rag.remove_document.return_value = True
    mock_rag.query.return_value = "Test response"

    # Mock ProcessorFactory
    mock_factory = mocker.patch('src.app.ProcessorFactory')
    mock_processor = mocker.MagicMock()
    mock_processor.process.return_value = ['chunk1', 'chunk2']
    mock_factory.get_processor.return_value = mock_processor

    return mock_rag

def test_add_and_remove_file(setup_context):
    # Add a file
    mock_file = MockFile("test.txt")
    result, _ = add_file_to_context(mock_file)
    assert len(result) == 1
    assert result[0][0] == "test.txt"

    # Remove the file
    result = remove_file_from_context("test.txt")
    assert len(result) == 1
    assert result[0][0] == "No files in context"

    # Verify RAGProcessor methods were called
    import src.app
    assert 'test.txt' not in src.app.processed_files_map
    setup_context.remove_document.assert_called_once_with(['id1', 'id2'])

def test_remove_nonexistent_file(setup_context):
    # Try to remove a file that doesn't exist
    result = remove_file_from_context("nonexistent.txt")
    assert len(result) == 1
    assert result[0][0] == "No files in context"

    # Verify RAGProcessor.remove_document was not called
    setup_context.remove_document.assert_not_called()

def test_query_after_remove(setup_context):
    # Add a file
    mock_file = MockFile("test.txt")
    add_file_to_context(mock_file)

    # Add another file
    mock_file2 = MockFile("test2.txt")
    add_file_to_context(mock_file2)

    # Remove one file
    remove_file_from_context("test.txt")

    # Query should still work with remaining file
    response = query_document("test question", "default")
    assert response == "Test response"
    setup_context.query.assert_called_once()