import os
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Generator, List, Optional, Tuple

import ollama
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointIdsList, PointStruct, VectorParams

from src.config.prompts import (
    BASE_PROMPT,
    DEEP_ANSWER_PROMPT,
    DEEP_COMBINE_PROMPT,
    DEEP_MAP_PROMPT,
    registry as prompt_registry,
)
from src.models.document import Document

DEFAULT_COLLECTION = "default"


class RAGProcessor:
    """
    Orchestrates document embedding, vector storage and RAG query execution.

    Accepts ollama and qdrant clients as constructor parameters (Dependency
    Injection) so the class is fully testable without patching globals.
    Use RAGProcessor.from_env() to build an instance from environment variables.
    """

    def __init__(
        self,
        ollama_client: ollama.Client,
        qdrant_client: QdrantClient,
        model_name: str,
        embedding_model: str,
        qdrant_path: str,
        vector_size: int = 1024,
        embed_batch_size: int = 32,
        truncation_factor: float = 0.8,
        max_truncation_attempts: int = 12,
        deep_block_chars: int = 10000,
        deep_max_blocks: int = 60,
        deep_max_workers: int = 2,
        deep_max_reduce_depth: int = 4,
    ) -> None:
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.qdrant_path = qdrant_path
        self.ollama_client = ollama_client
        self._mode = "memory"
        self.qdrant = qdrant_client
        self._vector_size = vector_size
        self._embed_batch_size = embed_batch_size
        self._truncation_factor = truncation_factor
        self._max_truncation_attempts = max_truncation_attempts
        self._deep_block_chars = deep_block_chars
        self._deep_max_blocks = deep_max_blocks
        self._deep_max_workers = deep_max_workers
        self._deep_max_reduce_depth = deep_max_reduce_depth

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_env(cls) -> "RAGProcessor":
        """Build a RAGProcessor from environment variables."""
        model_name = os.getenv("LLM_MODEL")
        if not model_name:
            raise ValueError(
                "LLM_MODEL environment variable is not set. "
                "Configure this in your .env file."
            )
        host = os.getenv("OLLAMA_HOST", "localhost")
        port = os.getenv("OLLAMA_PORT", "11434")
        return cls(
            ollama_client=ollama.Client(host=f"http://{host}:{port}"),
            qdrant_client=QdrantClient(":memory:"),
            model_name=model_name,
            embedding_model=os.getenv("EMBEDDING_MODEL", "mxbai-embed-large:latest"),
            qdrant_path=os.getenv("QDRANT_DB_PATH", "./data/qdrant"),
            vector_size=int(os.getenv("EMBEDDING_VECTOR_SIZE", "1024")),
            embed_batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "32")),
            truncation_factor=float(os.getenv("EMBEDDING_TRUNCATION_FACTOR", "0.8")),
            max_truncation_attempts=int(os.getenv("EMBEDDING_MAX_TRUNCATION_ATTEMPTS", "12")),
            deep_block_chars=int(os.getenv("DEEP_BLOCK_CHARS", "10000")),
            deep_max_blocks=int(os.getenv("DEEP_MAX_BLOCKS", "60")),
            deep_max_workers=int(os.getenv("DEEP_MAX_WORKERS", "2")),
        )

    # ── Storage mode ──────────────────────────────────────────────────────────

    @property
    def mode(self) -> str:
        return self._mode

    def set_mode(self, mode: str) -> None:
        if mode not in ("memory", "persist"):
            raise ValueError("Mode must be 'memory' or 'persist'")
        if mode == self._mode:
            return
        self._mode = mode
        if mode == "persist":
            os.makedirs(self.qdrant_path, exist_ok=True)
            self.qdrant = QdrantClient(path=self.qdrant_path)
        else:
            self.qdrant = QdrantClient(":memory:")

    def reset(self) -> None:
        """Drop all data from the current storage without changing mode."""
        if self._mode == "memory":
            self.qdrant = QdrantClient(":memory:")
        else:
            for col in self.get_collections():
                self.delete_collection(col)

    # ── Collections ───────────────────────────────────────────────────────────

    def get_collections(self) -> List[str]:
        return [c.name for c in self.qdrant.get_collections().collections]

    def create_collection(self, name: str) -> None:
        if name not in self.get_collections():
            self.qdrant.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=self._vector_size, distance=Distance.COSINE),
            )

    def delete_collection(self, name: str) -> None:
        try:
            self.qdrant.delete_collection(collection_name=name)
        except Exception as exc:
            print(f"Error deleting collection '{name}': {exc}")

    # ── Embeddings ────────────────────────────────────────────────────────────

    def _embed_single(self, text: str) -> List[float]:
        """
        Embed a single text, retrying with progressive truncation if the model
        rejects the input due to context length limits.

        Raises:
            RuntimeError: if the text cannot be embedded even after truncation.
        """
        attempt = text
        for _ in range(self._max_truncation_attempts):
            try:
                response = self.ollama_client.embed(
                    model=self.embedding_model, input=[attempt]
                )
                return response.embeddings[0]
            except Exception as exc:
                error_msg = str(exc).lower()
                if "context length" in error_msg or "input length" in error_msg:
                    attempt = attempt[: int(len(attempt) * self._truncation_factor)]
                    if not attempt:
                        raise RuntimeError(
                            "Text reduced to empty string during embedding truncation"
                        ) from exc
                else:
                    raise
        raise RuntimeError(
            f"Failed to embed text after {self._max_truncation_attempts} truncation attempts"
        )

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Embed texts in batches of embed_batch_size (one Ollama call per batch).
        If a batch fails (e.g. one chunk exceeds the context length), fall back
        to per-text embedding, which retries with progressive truncation.
        """
        embeddings: List[List[float]] = []
        size = max(1, self._embed_batch_size)
        for start in range(0, len(texts), size):
            batch = texts[start:start + size]
            try:
                response = self.ollama_client.embed(
                    model=self.embedding_model, input=batch
                )
                if len(response.embeddings) != len(batch):
                    raise RuntimeError("Embedding count does not match batch size")
                embeddings.extend(response.embeddings)
            except Exception:
                embeddings.extend(self._embed_single(text) for text in batch)
        return embeddings

    # ── Documents ─────────────────────────────────────────────────────────────

    def add_document(
        self, chunks: List[Document], collection_name: str = DEFAULT_COLLECTION
    ) -> List[str]:
        if not chunks:
            raise ValueError("No document chunks provided")

        self.create_collection(collection_name)

        embeddings = self._get_embeddings([c.page_content for c in chunks])
        ids = [str(uuid.uuid4()) for _ in chunks]
        points = [
            PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "page_content": chunk.page_content,
                    "chunk_index": index,
                    **chunk.metadata,
                },
            )
            for index, (point_id, embedding, chunk) in enumerate(
                zip(ids, embeddings, chunks)
            )
        ]
        self.qdrant.upsert(collection_name=collection_name, points=points)
        return ids

    def remove_document(
        self, document_ids: List[str], collection_name: str = DEFAULT_COLLECTION
    ) -> bool:
        if not document_ids:
            return False
        try:
            self.qdrant.delete(
                collection_name=collection_name,
                points_selector=PointIdsList(points=document_ids),
            )
            return True
        except Exception as exc:
            print(f"Error removing documents: {exc}")
            return False

    def rebuild_files_map(self) -> dict:
        """
        Reconstruct {collection: {file_name: [ids]}} from Qdrant payloads.
        Used when switching to persist mode to recover previously indexed files.
        """
        result: dict = {}
        for col_name in self.get_collections():
            result[col_name] = {}
            offset = None
            while True:
                points, next_offset = self.qdrant.scroll(
                    collection_name=col_name,
                    limit=100,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
                for point in points:
                    fname = point.payload.get(
                        "file_name", point.payload.get("source", "unknown")
                    )
                    result[col_name].setdefault(fname, []).append(str(point.id))
                if next_offset is None:
                    break
                offset = next_offset
        return result

    # ── Query ─────────────────────────────────────────────────────────────────

    def query_stream(
        self,
        question: str,
        collections: List[str],
        role: str = "default",
        model: Optional[str] = None,
    ) -> Generator[dict, None, None]:
        """
        Yield structured events progressively:
          {"type": "sources", "sources": [...]}   — emitted once before text
          {"type": "text",    "text": "token"}    — one per LLM token
        """
        if role not in prompt_registry:
            raise ValueError(
                f"Invalid role. Must be one of: {', '.join(prompt_registry.keys())}"
            )

        query_model = model or self.model_name
        query_embedding = self._embed_single(question)

        # Gather candidates from all requested collections, then keep global top-4
        all_points = []
        for col in collections:
            try:
                results = self.qdrant.query_points(
                    collection_name=col, query=query_embedding, limit=4
                )
                all_points.extend(results.points)
            except Exception:
                # A selected collection that was never created has nothing to
                # return; any other failure (Qdrant down, bad vector) must surface.
                if col in self.get_collections():
                    raise

        if not all_points:
            yield {"type": "text", "text": (
                "I couldn't find relevant information in the selected collections "
                "to answer your question. Try rephrasing or uploading more documents."
            )}
            return

        all_points.sort(key=lambda p: p.score, reverse=True)
        top = all_points[:4]

        sources = [
            {
                "file": p.payload.get("file_name", p.payload.get("source", "unknown")),
                "page": p.payload.get("page"),
                "score": round(float(p.score), 3),
            }
            for p in top
        ]
        yield {"type": "sources", "sources": sources}

        context = "\n\n".join(p.payload["page_content"] for p in top)
        prompt = BASE_PROMPT.format(
            role_prompt=prompt_registry.get_prompt(role),
            context=context,
            question=question,
        )

        yield from self._stream_answer(query_model, prompt)

    def _stream_answer(self, model: str, prompt: str) -> Generator[dict, None, None]:
        for chunk in self.ollama_client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        ):
            content = chunk.message.content
            if content:
                yield {"type": "text", "text": content}

    # ── Deep mode (map → recursive reduce) ────────────────────────────────────

    def query_stream_deep(
        self,
        question: str,
        collections: List[str],
        role: str = "default",
        model: Optional[str] = None,
    ) -> Generator[dict, None, None]:
        """
        Read every block of the selected documents instead of the top-k chunks.

        Map:    each block (~deep_block_chars) is asked for question-relevant notes.
        Reduce: notes are merged recursively until they fit in one block.
        Answer: the final notes are passed to the LLM and the answer is streamed.

        Yields the same events as query_stream, plus:
          {"type": "progress", "stage": "map"|"reduce", "done": n, "total": m}
        """
        if role not in prompt_registry:
            raise ValueError(
                f"Invalid role. Must be one of: {', '.join(prompt_registry.keys())}"
            )

        query_model = model or self.model_name
        blocks = self._build_blocks(collections)
        if not blocks:
            yield {"type": "text", "text": (
                "I couldn't find any documents in the selected collections."
            )}
            return
        if len(blocks) > self._deep_max_blocks:
            raise ValueError(
                f"Selected documents are too large for deep mode "
                f"({len(blocks)} blocks, limit {self._deep_max_blocks}). "
                "Select fewer collections or raise DEEP_MAX_BLOCKS."
            )

        # Map: extract question-relevant notes from every block
        total = len(blocks)
        yield {"type": "progress", "stage": "map", "done": 0, "total": total}
        extracted: List[str] = [""] * total
        prompts = [
            DEEP_MAP_PROMPT.format(source=label, block=text, question=question)
            for label, text in blocks
        ]
        for done, (index, result) in enumerate(
            self._complete_parallel(query_model, prompts), start=1
        ):
            extracted[index] = result
            yield {"type": "progress", "stage": "map", "done": done, "total": total}

        relevant = [
            (label, note)
            for (label, _), note in zip(blocks, extracted)
            if not self._is_empty_note(note)
        ]
        if not relevant:
            yield {"type": "text", "text": (
                "I read the selected documents in full but found nothing relevant "
                "to your question."
            )}
            return

        # Reduce: merge notes until they fit in a single block
        notes = [f"[{label}]\n{note}" for label, note in relevant]
        for _ in range(self._deep_max_reduce_depth):
            if sum(len(n) for n in notes) <= self._deep_block_chars:
                break
            batches = self._batch_notes(notes)
            if len(batches) == len(notes):
                break  # every note is already block-sized: nothing left to merge
            merged: List[str] = [""] * len(batches)
            prompts = []
            for i, batch in enumerate(batches):
                if len(batch) == 1:
                    merged[i] = batch[0]
                else:
                    prompts.append((i, DEEP_COMBINE_PROMPT.format(
                        notes="\n\n".join(batch), question=question
                    )))
            yield {"type": "progress", "stage": "reduce", "done": 0, "total": len(prompts)}
            for done, (j, result) in enumerate(
                self._complete_parallel(query_model, [p for _, p in prompts]), start=1
            ):
                merged[prompts[j][0]] = result
                yield {"type": "progress", "stage": "reduce", "done": done, "total": len(prompts)}
            notes = merged

        yield {"type": "sources", "sources": self._deep_sources(blocks, relevant)}

        prompt = DEEP_ANSWER_PROMPT.format(
            role_prompt=prompt_registry.get_prompt(role),
            notes="\n\n".join(notes),
            question=question,
        )
        yield from self._stream_answer(query_model, prompt)

    def _load_file_chunks(self, collections: List[str]) -> Dict[Tuple[str, str], List[dict]]:
        """Return {(collection, file_name): [payload, ...]} in original document order."""
        files: Dict[Tuple[str, str], List[dict]] = {}
        for col in collections:
            offset = None
            while True:
                try:
                    points, next_offset = self.qdrant.scroll(
                        collection_name=col,
                        limit=256,
                        offset=offset,
                        with_payload=True,
                        with_vectors=False,
                    )
                except Exception:
                    if col in self.get_collections():
                        raise
                    break  # selected collection was never created
                for point in points:
                    payload = point.payload or {}
                    fname = payload.get("file_name", payload.get("source", "unknown"))
                    files.setdefault((col, fname), []).append(payload)
                if next_offset is None:
                    break
                offset = next_offset

        # chunk_index preserves the original order; chunks indexed before it
        # existed fall back to page order.
        for payloads in files.values():
            payloads.sort(key=lambda p: (
                p.get("chunk_index") is None,
                p.get("chunk_index") or 0,
                p.get("page") or 0,
            ))
        return files

    def _build_blocks(self, collections: List[str]) -> List[Tuple[str, str]]:
        """Group consecutive chunks of each file into (file_label, text) blocks."""
        blocks: List[Tuple[str, str]] = []
        for (_, fname), payloads in self._load_file_chunks(collections).items():
            current: List[str] = []
            size = 0
            for payload in payloads:
                text = payload.get("page_content", "")
                if current and size + len(text) > self._deep_block_chars:
                    blocks.append((fname, "\n".join(current)))
                    current, size = [], 0
                current.append(text)
                size += len(text)
            if current:
                blocks.append((fname, "\n".join(current)))
        return blocks

    def _batch_notes(self, notes: List[str]) -> List[List[str]]:
        """Greedily group notes so each batch fits in one block."""
        batches: List[List[str]] = []
        current: List[str] = []
        size = 0
        for note in notes:
            if current and size + len(note) > self._deep_block_chars:
                batches.append(current)
                current, size = [], 0
            current.append(note)
            size += len(note)
        if current:
            batches.append(current)
        return batches

    def _complete(self, model: str, prompt: str) -> str:
        response = self.ollama_client.chat(
            model=model, messages=[{"role": "user", "content": prompt}]
        )
        return (response.message.content or "").strip()

    def _complete_parallel(
        self, model: str, prompts: List[str]
    ) -> Generator[Tuple[int, str], None, None]:
        """Run prompts concurrently, yielding (index, result) as each one finishes."""
        if not prompts:
            return
        pool = ThreadPoolExecutor(max_workers=max(1, self._deep_max_workers))
        try:
            futures = {
                pool.submit(self._complete, model, prompt): index
                for index, prompt in enumerate(prompts)
            }
            for future in as_completed(futures):
                yield futures[future], future.result()
        finally:
            # Client disconnects close the generator: drop pending LLM calls
            pool.shutdown(wait=False, cancel_futures=True)

    @staticmethod
    def _is_empty_note(note: str) -> bool:
        cleaned = note.strip().strip(".*_`'\"").upper()
        return not cleaned or (cleaned.startswith("NONE") and len(cleaned) <= 12)

    @staticmethod
    def _deep_sources(
        blocks: List[Tuple[str, str]], relevant: List[Tuple[str, str]]
    ) -> List[dict]:
        """One source per file; score = share of the file's blocks that were relevant."""
        totals: Dict[str, int] = {}
        for label, _ in blocks:
            totals[label] = totals.get(label, 0) + 1
        hits: Dict[str, int] = {}
        for label, _ in relevant:
            hits[label] = hits.get(label, 0) + 1
        sources = [
            {
                "file": label,
                "page": None,
                "score": round(count / totals[label], 3),
                "blocks": count,
            }
            for label, count in hits.items()
        ]
        sources.sort(key=lambda s: s["blocks"], reverse=True)
        return sources

    def query(
        self,
        question: str,
        collections: Optional[List[str]] = None,
        role: str = "default",
        model: Optional[str] = None,
        deep: bool = False,
    ) -> str:
        cols = collections if collections is not None else [DEFAULT_COLLECTION]
        stream = self.query_stream_deep if deep else self.query_stream
        return "".join(
            item["text"]
            for item in stream(question, cols, role, model)
            if item.get("type") == "text"
        )

    # ── Models ────────────────────────────────────────────────────────────────

    def get_available_models(self) -> List[str]:
        try:
            return [m.model for m in self.ollama_client.list().models]
        except Exception:
            return [self.model_name] if self.model_name else []
