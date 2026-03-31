import ollama
import os
import shutil
import uuid
from typing import List, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, PointIdsList
from src.config.prompts import ROLE_PROMPTS, BASE_PROMPT
from src.models.document import Document

COLLECTION_NAME = "documents"
VECTOR_SIZE = 1024  # mxbai-embed-large dimension


class RAGProcessor:
    def __init__(self):
        self.model_name = os.getenv('LLM_MODEL')
        if not self.model_name:
            raise ValueError("LLM_MODEL environment variable is not set. Configure this in your .env file")

        self.embedding_model = os.getenv('EMBEDDING_MODEL', 'mxbai-embed-large:latest')
        self.qdrant_path = os.getenv('QDRANT_DB_PATH', './data/qdrant')
        self.persist_db = os.getenv('PERSIST_VECTORDB', 'false').lower() == 'true'

        ollama_host = os.getenv('OLLAMA_HOST', 'localhost')
        ollama_port = os.getenv('OLLAMA_PORT', '11434')
        self.ollama_client = ollama.Client(host=f"http://{ollama_host}:{ollama_port}")

        self._init_qdrant()

    def _init_qdrant(self):
        if self.persist_db:
            os.makedirs(self.qdrant_path, exist_ok=True)
            self.qdrant = QdrantClient(path=self.qdrant_path)
        else:
            self.qdrant = QdrantClient(":memory:")
        self._ensure_collection()

    def _ensure_collection(self):
        collections = [c.name for c in self.qdrant.get_collections().collections]
        if COLLECTION_NAME not in collections:
            self.qdrant.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            )

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

    def add_document(self, chunks: List[Document]) -> List[str]:
        if not chunks:
            raise ValueError("No document chunks provided")

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

        self.qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
        return ids

    def remove_document(self, document_ids: List[str]) -> bool:
        if not document_ids:
            return False
        try:
            self.qdrant.delete(
                collection_name=COLLECTION_NAME,
                points_selector=PointIdsList(points=document_ids),
            )
            return True
        except Exception as e:
            print(f"Error removing documents: {e}")
            return False

    def _clean_db(self):
        try:
            self.qdrant.delete_collection(collection_name=COLLECTION_NAME)
        except Exception:
            pass
        self._ensure_collection()

    def process_document(self, chunks: List[Document], clean_db: bool = True):
        if not chunks:
            raise ValueError("No document chunks provided")
        if clean_db:
            self._clean_db()
        self.add_document(chunks)

    def query_stream(self, question: str, role: str = "default", model: Optional[str] = None):
        """Generator that yields text chunks as the LLM produces them."""
        if role not in ROLE_PROMPTS:
            raise ValueError(f"Invalid role. Must be one of: {', '.join(ROLE_PROMPTS.keys())}")

        query_model = model if model else self.model_name

        query_embedding = self._get_embeddings([question])[0]
        results = self.qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=query_embedding,
            limit=4,
        )

        if not results.points:
            yield "I couldn't find relevant information in the uploaded documents to answer your question. Try rephrasing or uploading more documents."
            return

        context = "\n\n".join([point.payload["page_content"] for point in results.points])

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

    def query(self, question: str, role: str = "default", model: Optional[str] = None) -> str:
        if role not in ROLE_PROMPTS:
            raise ValueError(f"Invalid role. Must be one of: {', '.join(ROLE_PROMPTS.keys())}")

        query_model = model if model else self.model_name

        # Embed the question and search
        query_embedding = self._get_embeddings([question])[0]
        results = self.qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=query_embedding,
            limit=4,
        )

        if not results.points:
            return "I couldn't find relevant information in the uploaded documents to answer your question. Try rephrasing or uploading more documents."

        context = "\n\n".join([point.payload["page_content"] for point in results.points])

        prompt = BASE_PROMPT.format(
            role_prompt=ROLE_PROMPTS[role],
            context=context,
            question=question,
        )

        response = self.ollama_client.chat(
            model=query_model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.message.content

    def get_available_models(self) -> List[str]:
        try:
            response = self.ollama_client.list()
            return [m.model for m in response.models]
        except Exception:
            return [self.model_name] if self.model_name else []
