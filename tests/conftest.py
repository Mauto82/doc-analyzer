import pytest
from unittest.mock import Mock, MagicMock
import os
from pathlib import Path


@pytest.fixture
def test_data_dir():
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def mock_ollama_client():
    client = MagicMock()
    # Modern ollama API: chat returns ChatResponse with .message.content
    response = MagicMock()
    response.message.content = "Test response"
    client.chat.return_value = response
    # embed returns EmbedResponse with .embeddings
    embed_response = MagicMock()
    embed_response.embeddings = [[0.1] * 1024]
    client.embed.return_value = embed_response
    # list returns ListResponse with .models
    model = MagicMock()
    model.model = "test-model"
    list_response = MagicMock()
    list_response.models = [model]
    client.list.return_value = list_response
    return client


@pytest.fixture
def mock_qdrant_client():
    client = MagicMock()
    collections_response = MagicMock()
    collections_response.collections = []
    client.get_collections.return_value = collections_response
    return client
