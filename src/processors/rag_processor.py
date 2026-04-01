import os
import uuid
from typing import Generator, List, Optional

import ollama
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointIdsList, PointStruct, VectorParams

from src.config.prompts import BASE_PROMPT, registry as prompt_registry
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
        truncation_factor: float = 0.8,
        max_truncation_attempts: int = 12,
    ) -> None:
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.qdrant_path = qdrant_path
        self.ollama_client = ollama_client
        self._mode = "memory"
        self.qdrant = qdrant_client
        self._vector_size = vector_size
        self._truncation_factor = truncation_factor
        self._max_truncation_attempts = max_truncation_attempts

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
            truncation_factor=float(os.getenv("EMBEDDING_TRUNCATION_FACTOR", "0.8")),
            max_truncation_attempts=int(os.getenv("EMBEDDING_MAX_TRUNCATION_ATTEMPTS", "12")),
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
        return [self._embed_single(text) for text in texts]

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
                payload={"page_content": chunk.page_content, **chunk.metadata},
            )
            for point_id, embedding, chunk in zip(ids, embeddings, chunks)
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
                pass

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

        for chunk in self.ollama_client.chat(
            model=query_model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        ):
            content = chunk.message.content
            if content:
                yield {"type": "text", "text": content}

    def query(
        self,
        question: str,
        collections: Optional[List[str]] = None,
        role: str = "default",
        model: Optional[str] = None,
    ) -> str:
        cols = collections if collections is not None else [DEFAULT_COLLECTION]
        return "".join(
            item["text"]
            for item in self.query_stream(question, cols, role, model)
            if item.get("type") == "text"
        )

    # ── Models ────────────────────────────────────────────────────────────────

    def get_available_models(self) -> List[str]:
        try:
            return [m.model for m in self.ollama_client.list().models]
        except Exception:
            return [self.model_name] if self.model_name else []
