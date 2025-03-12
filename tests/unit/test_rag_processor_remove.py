def test_remove_document(rag_processor, sample_chunks, mocker):
    # Mock embeddings
    mocker.patch.object(rag_processor, 'embeddings')

    # Mock vectordb
    mock_chroma = mocker.MagicMock()
    mocker.patch.object(rag_processor, '_ensure_db')
    rag_processor.vectordb = mock_chroma

    # Mock add_documents to return fake IDs
    mock_ids = ['id1', 'id2', 'id3']
    mock_chroma.add_documents.return_value = mock_ids

    # Add documents first
    ids = rag_processor.add_document(sample_chunks)
    assert ids == mock_ids

    # Now test remove_document
    result = rag_processor.remove_document(['id1', 'id2'])
    assert result is True
    mock_chroma.delete.assert_called_once_with(ids=['id1', 'id2'])

def test_remove_document_empty_ids(rag_processor, mocker):
    # Mock _ensure_db to ensure vectordb exists
    mocker.patch.object(rag_processor, '_ensure_db')

    # Test with empty list
    result = rag_processor.remove_document([])
    assert result is False

    # Test with None
    result = rag_processor.remove_document(None)
    assert result is False

def test_remove_document_no_vectordb(rag_processor, mocker):
    # Mock _ensure_db
    mocker.patch.object(rag_processor, '_ensure_db')
    rag_processor.vectordb = None

    # Test with vectordb None
    result = rag_processor.remove_document(['id1'])
    assert result is False