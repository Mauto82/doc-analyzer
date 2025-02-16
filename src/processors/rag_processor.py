from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
import ollama
import os

class RAGProcessor:
    def __init__(self):
        self.model_name = os.getenv('DEEPSEEK_MODEL', 'deepseek-r1:14b')
        self.chroma_path = os.getenv('CHROMA_DB_PATH', './data/chroma')
        self.persist_db = os.getenv('PERSIST_VECTORDB', 'false').lower() == 'true'

        self.embeddings = OllamaEmbeddings(
            model=self.model_name,
            base_url=f"http://{os.getenv('OLLAMA_HOST', 'localhost')}:{os.getenv('OLLAMA_PORT', '11434')}"
        )

        # Initialize the vector database
        self.vectordb = None

    def query(self, question, chunks):
        if not chunks:
            raise ValueError("No document chunks provided")
        # Create or clear the vector database
        if self.vectordb is None:
            self.vectordb = Chroma(
                persist_directory=self.chroma_path,
                embedding_function=self.embeddings
            )

        # Clean previous documents if not persisting
        if not self.persist_db:
            # Get all existing collection names
            collections = self.vectordb._client.list_collections()
            for collection in collections:
                # Delete each collection
                self.vectordb._client.delete_collection(collection.name)

        # Add new documents
        self.vectordb = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.chroma_path
        )

        # Retrieve the most relevant chunks
        retriever = self.vectordb.as_retriever()
        relevant_chunks = retriever.get_relevant_documents(question)

        # Prepare the context
        context = "\n\n".join([chunk.page_content for chunk in relevant_chunks])

        # Log the chunks being used (for debugging)
        print(f"Using the following chunks for context:\n{context}\n")

        # Persist changes
        self.vectordb.persist()

        # Query DeepSeek with a more structured prompt
        client = ollama.Client(
            host=f"http://{os.getenv('OLLAMA_HOST', 'localhost')}:{os.getenv('OLLAMA_PORT', '11434')}"
        )

        prompt = f"""This text contains multiple fairy tales. Based on the following excerpts, answer the question.
        First, identify which story or stories are relevant to the question.
        Then, provide an answer using ONLY the information from the relevant parts.
        If multiple stories mention the subject of the question, specify which story you're referring to.

        Text excerpts: {context}

        Question: {question}

        Detailed answer (specify which story/stories you're referring to):"""

        response = client.chat(
            model=self.model_name,
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )

        return response['message']['content']