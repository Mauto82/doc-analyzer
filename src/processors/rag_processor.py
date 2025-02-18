from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
import ollama
import os
import shutil
from src.config.prompts import ROLE_PROMPTS, BASE_PROMPT

class RAGProcessor:
    def __init__(self):
        self.model_name = os.getenv('DEEPSEEK_MODEL', 'deepseek-r1:14b')
        self.chroma_path = os.getenv('CHROMA_DB_PATH', './data/chroma')
        self.persist_db = os.getenv('PERSIST_VECTORDB', 'false').lower() == 'true'

        self.embeddings = OllamaEmbeddings(
            model=self.model_name,
            base_url=f"http://{os.getenv('OLLAMA_HOST', 'localhost')}:{os.getenv('OLLAMA_PORT', '11434')}"
        )
        self.vectordb = None

    def _clean_db(self):
        """Completely clean the vector database"""
        if self.vectordb is not None:
            try:
                collections = self.vectordb._client.list_collections()
                for collection in collections:
                    self.vectordb._client.delete_collection(collection.name)
            except Exception:
                pass
            self.vectordb = None

        if os.path.exists(self.chroma_path):
            shutil.rmtree(self.chroma_path)
        os.makedirs(self.chroma_path, exist_ok=True)

    def process_document(self, chunks):
        """Initialize the vector database with new document chunks"""
        if not chunks:
            raise ValueError("No document chunks provided")

        self._clean_db()

        self.vectordb = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.chroma_path if self.persist_db else None
        )

        if self.persist_db:
            self.vectordb.persist()

    def query(self, question, role="default"):
        """
        Query the existing vector database with role-based context

        Args:
        question (str): The question to answer
        role (str): The role to assume when answering. Must be one of the roles defined in ROLE_PROMPTS

        Returns:
        str: The answer from the model

        Raises:
        ValueError: If no document is loaded or if role is invalid
        """
        if self.vectordb is None:
            raise ValueError("Please upload a document before asking questions")

        if role not in ROLE_PROMPTS:
            raise ValueError(f"Invalid role. Must be one of: {', '.join(ROLE_PROMPTS.keys())}")

        # Retrieve the most relevant chunks
        retriever = self.vectordb.as_retriever()
        relevant_chunks = retriever.get_relevant_documents(question)

        # Prepare the context
        context = "\n\n".join([chunk.page_content for chunk in relevant_chunks])

        # Prepare the role-specific prompt
        prompt = BASE_PROMPT.format(
            role_prompt=ROLE_PROMPTS[role],
            context=context,
            question=question
        )

        # Query DeepSeek
        client = ollama.Client(
            host=f"http://{os.getenv('OLLAMA_HOST', 'localhost')}:{os.getenv('OLLAMA_PORT', '11434')}"
        )

        response = client.chat(
            model=self.model_name,
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )

        return response['message']['content']