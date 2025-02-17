import pytest
from processors.base.document_processor import DocumentProcessor

def test_cannot_instantiate_abstract_base():
    """Test that DocumentProcessor cannot be instantiated directly"""
    with pytest.raises(TypeError):
        DocumentProcessor()