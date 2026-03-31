from typing import List
from src.models.document import Document


class RecursiveCharacterTextSplitter:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = ["\n\n", "\n", " ", ""]

    def split_documents(self, documents: List[Document]) -> List[Document]:
        chunks = []
        for doc in documents:
            text_chunks = self._split_text(doc.page_content)
            for chunk_text in text_chunks:
                chunks.append(Document(
                    page_content=chunk_text,
                    metadata=dict(doc.metadata),
                ))
        return chunks

    def _split_text(self, text: str) -> List[str]:
        return self._recursive_split(text, self.separators)

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        separator = separators[0] if separators else ""
        remaining_separators = separators[1:] if len(separators) > 1 else []

        if separator == "":
            return self._split_by_chars(text)

        parts = text.split(separator)

        chunks = []
        current = ""
        for part in parts:
            candidate = current + separator + part if current else part
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    if len(current) <= self.chunk_size:
                        chunks.append(current)
                    else:
                        chunks.extend(self._recursive_split(current, remaining_separators))
                current = part

        if current:
            if len(current) <= self.chunk_size:
                chunks.append(current)
            else:
                chunks.extend(self._recursive_split(current, remaining_separators))

        return self._merge_with_overlap(chunks)

    def _split_by_chars(self, text: str) -> List[str]:
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunks.append(text[start:end])
            start += self.chunk_size - self.chunk_overlap
        return chunks

    def _merge_with_overlap(self, chunks: List[str]) -> List[str]:
        if not chunks or self.chunk_overlap <= 0:
            return [c for c in chunks if c.strip()]

        merged = []
        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
            if i > 0 and merged:
                prev = merged[-1]
                overlap_text = prev[-self.chunk_overlap:] if len(prev) > self.chunk_overlap else prev
                candidate = overlap_text + chunk
                if len(candidate) <= self.chunk_size:
                    chunk = candidate
            merged.append(chunk)
        return merged
