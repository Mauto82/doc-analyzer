from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
import ollama
import os
import shutil

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
        # Delete any existing Chroma collections
        if self.vectordb is not None:
            try:
                collections = self.vectordb._client.list_collections()
                for collection in collections:
                    self.vectordb._client.delete_collection(collection.name)
            except Exception:
                pass  # If there's an error deleting collections, we'll clean the directory anyway
            self.vectordb = None

        # Remove the entire directory if it exists
        if os.path.exists(self.chroma_path):
            shutil.rmtree(self.chroma_path)

        # Recreate empty directory
        os.makedirs(self.chroma_path, exist_ok=True)

    def process_document(self, chunks):
        """Initialize the vector database with new document chunks"""
        if not chunks:
            raise ValueError("No document chunks provided")

        # Always clean the database before processing new document
        self._clean_db()

        # Create new vector database
        self.vectordb = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.chroma_path if self.persist_db else None
        )

        if self.persist_db:
            self.vectordb.persist()

    def query(self, question):
        """Query the existing vector database"""
        if self.vectordb is None:
            raise ValueError("Please upload a document before asking questions")

        # Retrieve the most relevant chunks
        retriever = self.vectordb.as_retriever()
        relevant_chunks = retriever.get_relevant_documents(question)

        # Prepare the context
        context = "\n\n".join([chunk.page_content for chunk in relevant_chunks])

        # Query DeepSeek
        client = ollama.Client(
            host=f"http://{os.getenv('OLLAMA_HOST', 'localhost')}:{os.getenv('OLLAMA_PORT', '11434')}"
        )

        prompt = f"""Based on the following document excerpts, answer the question.
Use ONLY the information provided in these excerpts to formulate your answer.
If the answer requires information from multiple sections, please specify which parts you're referencing.

            Document excerpts: {context}

            Question: {question}

            Please provide your answer in the same language as the question, using only information from the provided excerpts:"""

        response = client.chat(
            model=self.model_name,
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )

        return response['message']['content']