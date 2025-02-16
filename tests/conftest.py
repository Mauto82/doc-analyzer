import pytest
from unittest.mock import Mock
import os
from pathlib import Path

@pytest.fixture
def test_data_dir():
    return Path(__file__).parent / "fixtures"

@pytest.fixture
def mock_ollama_client():
    client = Mock()
    client.chat.return_value = {
        'message': {'content': 'Test response'}
    }
    return client

@pytest.fixture
def mock_embeddings():
    return Mock()

@pytest.fixture
def mock_vectordb():
    db = Mock()
    db.as_retriever.return_value = Mock()
    db.persist = Mock()
    return db