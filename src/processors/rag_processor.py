import ollama
import os
import uuid
from typing import List, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, PointIdsList
from src.config.prompts import ROLE_PROMPTS, BASE_PROMPT
from src.models.document import Document

VECTOR_SIZE = 1024  # mxbai-embed-large dimension
DEFAULT_COLLECTION = "default"


class RAGProcessor:
    def __init__(self):
        self.model_name = os.getenv('LLM_MODEL')
        if not self.model_name:
            raise ValueError("LLM_MODEL environment variable is not set. Configure this in your .env file")

        self.embedding_model = os.getenv('EMBEDDING_MODEL', 'mxbai-embed-large:latest')
        self.qdrant_path = os.getenv('QDRANT_DB_PATH', './data/qdrant')

        ollama_host = os.getenv('OLLAMA_HOST', 'localhost')
        ollama_port = os.getenv('OLLAMA_PORT', '11434')
        self.ollama_client = ollama.Client(host=f"http://{ollama_host}:{ollama_port}")

        self._mode = "memory"
        self.qdrant = QdrantClient(":memory:")

    # ── Mode ──────────────────────────────────────────────────────────────────

    @property
    def mode(self) -> str:
        return self._mode

    def set_mode(self, mode: str):
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

    # ── Collections ───────────────────────────────────────────────────────────

    def get_collections(self) -> List[str]:
        return [c.name for c in self.qdrant.get_collections().collections]

    def create_collection(self, name: str):
        if name not in self.get_collections():
            self.qdrant.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            )

    def delete_collection(self, name: str):
        try:
            self.qdrant.delete_collection(collection_name=name)
        except Exception as e:
            print(f"Error deleting collection '{name}': {e}")

    # ── Embeddings ────────────────────────────────────────────────────────────

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            # Retry with progressive truncation if model context is exceeded
            attempt = text
            while attempt:
                try:
                    response = self.ollama_client.embed(model=self.embedding_model, input=[attempt])
                    embeddings.append(response.embeddings[0])
                    break
                except Exception as e:
                    if "context length" in str(e).lower() or "input length" in str(e).lower():
                        attempt = attempt[:int(len(attempt) * 0.8)]
                    else:
                        raise
        return embeddings

    # ── Documents ─────────────────────────────────────────────────────────────

    def add_document(self, chunks: List[Document], collection_name: str = DEFAULT_COLLECTION) -> List[str]:
        if not chunks:
            raise ValueError("No document chunks provided")

        self.create_collection(collection_name)

        texts = [chunk.page_content for chunk in chunks]
        embeddings = self._get_embeddings(texts)

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

    def remove_document(self, document_ids: List[str], collection_name: str = DEFAULT_COLLECTION) -> bool:
        if not document_ids:
            return False
        try:
            self.qdrant.delete(
                collection_name=collection_name,
                points_selector=PointIdsList(points=document_ids),
            )
            return True
        except Exception as e:
            print(f"Error removing documents: {e}")
            return False

    def rebuild_files_map(self) -> dict:
        """Reconstruct {collection: {file_name: [ids]}} from Qdrant payloads (persist mode)."""
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
                    fname = point.payload.get("file_name", point.payload.get("source", "unknown"))
                    if fname not in result[col_name]:
                        result[col_name][fname] = []
                    result[col_name][fname].append(str(point.id))
                if next_offset is None:
                    break
                offset = next_offset
        return result

    # ── Query ─────────────────────────────────────────────────────────────────

    def query_stream(self, question: str, collections: List[str], role: str = "default", model: Optional[str] = None):
        """Generator that yields text chunks as the LLM produces them."""
        if role not in ROLE_PROMPTS:
            raise ValueError(f"Invalid role. Must be one of: {', '.join(ROLE_PROMPTS.keys())}")

        query_model = model if model else self.model_name
        query_embedding = self._get_embeddings([question])[0]

        all_points = []
        for col in collections:
            try:
                results = self.qdrant.query_points(
                    collection_name=col,
                    query=query_embedding,
                    limit=4,
                )
                all_points.extend(results.points)
            except Exception:
                pass

        if not all_points:
            yield "I couldn't find relevant information in the selected collections to answer your question. Try rephrasing or uploading more documents."
            return

        # Sort by score descending, take top 4 overall
        all_points.sort(key=lambda p: p.score, reverse=True)
        top_points = all_points[:4]

        context = "\n\n".join([p.payload["page_content"] for p in top_points])

        prompt = BASE_PROMPT.format(
            role_prompt=ROLE_PROMPTS[role],
            context=context,
            question=question,
        )

        stream = self.ollama_client.chat(
            model=query_model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
        for chunk in stream:
            content = chunk.message.content
            if content:
                yield content

    def query(self, question: str, collections: List[str] = None, role: str = "default", model: Optional[str] = None) -> str:
        if collections is None:
            collections = [DEFAULT_COLLECTION]
        return "".join(self.query_stream(question, collections, role, model))

    def get_available_models(self) -> List[str]:
        try:
            response = self.ollama_client.list()
            return [m.model for m in response.models]
        except Exception:
            return [self.model_name] if self.model_name else []
