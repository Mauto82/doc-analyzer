from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
import ollama
import os
import shutil
from src.config.prompts import ROLE_PROMPTS, BASE_PROMPT

class RAGProcessor:
    def __init__(self):
        # Check if LLM_MODEL is set
        self.model_name = os.getenv('LLM_MODEL')
        if not self.model_name:
            raise ValueError("LLM_MODEL environment variable is not set. Configure this in your .env file")

        self.chroma_path = os.getenv('CHROMA_DB_PATH', './data/chroma')
        self.persist_db = os.getenv('PERSIST_VECTORDB', 'false').lower() == 'true'

        self.embeddings = OllamaEmbeddings(
            model=self.model_name,
            base_url=f"http://{os.getenv('OLLAMA_HOST', 'localhost')}:{os.getenv('OLLAMA_PORT', '11434')}"
        )
        self.vectordb = None
        self._ensure_db()

    def _ensure_db(self):
        """Ensure the vector database exists and is initialized"""
        if self.vectordb is None:
            if os.path.exists(self.chroma_path) and os.listdir(self.chroma_path):
                # Load existing database if it exists
                try:
                    self.vectordb = Chroma(
                        persist_directory=self.chroma_path,
                        embedding_function=self.embeddings
                    )
                except Exception:
                    # If loading fails, create a new one
                    self._create_new_db()
            else:
                # Create a new database if it doesn't exist
                self._create_new_db()

    def _create_new_db(self):
        """Create a new empty vector database"""
        if os.path.exists(self.chroma_path):
            shutil.rmtree(self.chroma_path)
        os.makedirs(self.chroma_path, exist_ok=True)

        # Initialize an empty database
        self.vectordb = Chroma(
            persist_directory=self.chroma_path if self.persist_db else None,
            embedding_function=self.embeddings
        )

        if self.persist_db:
            self.vectordb.persist()

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

        self._ensure_db()

    def process_document(self, chunks, clean_db=True):
        """
        Initialize the vector database with new document chunks

        Args:
        chunks: Document chunks to add to the database
        clean_db: If True, clean the database before adding new chunks
        """
        if not chunks:
            raise ValueError("No document chunks provided")

        if clean_db:
            self._clean_db()
            self.add_document(chunks)
        else:
            self.add_document(chunks)

    def add_document(self, chunks):
        """
        Add document chunks to the existing vector database

        Args:
        chunks: Document chunks to add to the database
        """
        if not chunks:
            raise ValueError("No document chunks provided")

        self._ensure_db()

        # Add documents to the existing database
        ids = self.vectordb.add_documents(chunks)

        if self.persist_db:
            self.vectordb.persist()

        return ids

    def remove_document(self, document_ids):
        """
        Remove specific documents from the vector database by their IDs

        Args:
        document_ids: List of document IDs to remove

        Returns:
        bool: True if successful, False otherwise
        """
        if not document_ids:
            return False

        self._ensure_db()

        if self.vectordb is None:
            return False

        try:
            # Remove the documents by their IDs
            self.vectordb.delete(ids=document_ids)

            if self.persist_db:
                self.vectordb.persist()

            return True
        except Exception as e:
            print(f"Error removing documents: {str(e)}")
            return False

    def query(self, question, role="default", model=None):
        """
        Query the existing vector database with role-based context

        Args:
        question (str): The question to answer
        role (str): The role to assume when answering. Must be one of the roles defined in ROLE_PROMPTS
        model (str, optional): The model to use for the query. Defaults to the model specified in LLM_MODEL

        Returns:
        str: The answer from the model

        Raises:
        ValueError: If no document is loaded or if role is invalid
        """
        self._ensure_db()

        if self.vectordb is None:
            raise ValueError("Please upload a document before asking questions")

        if role not in ROLE_PROMPTS:
            raise ValueError(f"Invalid role. Must be one of: {', '.join(ROLE_PROMPTS.keys())}")

        # Use specified model or fall back to default
        query_model = model if model else self.model_name

        # Retrieve the most relevant chunks
        retriever = self.vectordb.as_retriever()
        relevant_chunks = retriever.get_relevant_documents(question)

        if not relevant_chunks:
            return "I couldn't find relevant information in the uploaded documents to answer your question. Try rephrasing or uploading more documents."

        # Prepare the context
        context = "\n\n".join([chunk.page_content for chunk in relevant_chunks])

        # Prepare the role-specific prompt
        prompt = BASE_PROMPT.format(
            role_prompt=ROLE_PROMPTS[role],
            context=context,
            question=question
        )

        # Query LLM
        client = ollama.Client(
            host=f"http://{os.getenv('OLLAMA_HOST', 'localhost')}:{os.getenv('OLLAMA_PORT', '11434')}"
        )

        response = client.chat(
            model=query_model,
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )

        return response['message']['content']

    def get_available_models(self):
        """
        Get a list of available models from Ollama

        Returns:
        list: List of available model names
        """
        try:
            client = ollama.Client(
                host=f"http://{os.getenv('OLLAMA_HOST', 'localhost')}:{os.getenv('OLLAMA_PORT', '11434')}"
            )
            models = client.list()
            return [model['name'] for model in models['models']]
        except Exception:
            # Return just the current model if we can't get the list
            return [self.model_name] if self.model_name else []