"""
DocumentService: single source of truth for document state and orchestration.

Responsibilities:
- Track which files are indexed and in which collection (files_map)
- Coordinate upload, removal, and clear operations with RAGProcessor
- Manage storage mode switches

This removes all state management and business logic from the FastAPI
route handlers, which now act as thin HTTP adapters.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, List

from src.models.document import Document

if TYPE_CHECKING:
    from src.processors.rag_processor import RAGProcessor

# {collection_name: {file_name: [chunk_ids]}}
FilesMap = dict[str, dict[str, list[str]]]


class DocumentService:
    def __init__(self, rag: "RAGProcessor") -> None:
        self._rag = rag
        self._files_map: FilesMap = {}

    # ── Read ──────────────────────────────────────────────────────────────────

    @property
    def files_map(self) -> FilesMap:
        return self._files_map

    def has_documents(self) -> bool:
        return any(bool(files) for files in self._files_map.values())

    def is_duplicate(self, file_name: str, collection: str) -> bool:
        return file_name in self._files_map.get(collection, {})

    def resolve_query_collections(self, requested: List[str]) -> List[str]:
        """Return the effective collections to query.
        If the caller specified collections explicitly, use those.
        Otherwise fall back to all collections that have at least one file.
        """
        return requested if requested else list(self._files_map.keys())

    # ── Write ─────────────────────────────────────────────────────────────────

    def add_file(
        self, file_name: str, chunks: List[Document], collection: str
    ) -> int:
        """Index chunks and track the file. Returns the number of chunks indexed."""
        ids = self._rag.add_document(chunks, collection_name=collection)
        self._files_map.setdefault(collection, {})[file_name] = ids
        return len(ids)

    def remove_file(self, file_name: str, collection: str) -> bool:
        """Remove a file from the vector store and the tracking map."""
        col_files = self._files_map.get(collection, {})
        if file_name not in col_files:
            return False
        ids = col_files[file_name]
        ok = self._rag.remove_document(ids, collection_name=collection)
        if ok:
            del self._files_map[collection][file_name]
        return ok

    def create_collection(self, name: str) -> None:
        self._rag.create_collection(name)
        self._files_map.setdefault(name, {})

    def delete_collection(self, name: str) -> None:
        self._rag.delete_collection(name)
        self._files_map.pop(name, None)

    def clear_all(self) -> None:
        """Drop all data. Delegates storage reset to RAGProcessor."""
        self._rag.reset()
        self._files_map = {}

    def switch_mode(self, mode: str) -> None:
        """Switch storage mode and reconcile the files map."""
        self._rag.set_mode(mode)
        self._files_map = self._rag.rebuild_files_map() if mode == "persist" else {}
