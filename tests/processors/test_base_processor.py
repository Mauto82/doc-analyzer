import os
import pytest
from src.processors.base.document_processor import DocumentProcessor, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP


def test_cannot_instantiate_abstract_base():
    with pytest.raises(TypeError):
        DocumentProcessor()


def test_default_chunk_constants_are_positive():
    assert DEFAULT_CHUNK_SIZE > 0
    assert DEFAULT_CHUNK_OVERLAP >= 0
    assert DEFAULT_CHUNK_OVERLAP < DEFAULT_CHUNK_SIZE


def test_temp_path_creates_and_cleans_up():
    """_temp_path yields an existing path that is deleted on exit."""
    recorded = []
    with DocumentProcessor._temp_path(suffix=".txt") as path:
        recorded.append(path)
        assert os.path.exists(path)
        assert path.endswith(".txt")
    assert not os.path.exists(recorded[0])


def test_temp_path_cleans_up_on_exception():
    """Temp file is removed even when the body raises."""
    recorded = []
    with pytest.raises(ValueError):
        with DocumentProcessor._temp_path(suffix=".pdf") as path:
            recorded.append(path)
            raise ValueError("boom")
    assert not os.path.exists(recorded[0])


def test_temp_path_no_suffix():
    with DocumentProcessor._temp_path() as path:
        assert os.path.exists(path)
    assert not os.path.exists(path)