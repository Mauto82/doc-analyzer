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

    def query(self, question, chunks):
        try:
            # Create the vector database
            vectordb = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=self.chroma_path
            )

            # Retrieve the most relevant chunks
            retriever = vectordb.as_retriever()
            relevant_chunks = retriever.get_relevant_documents(question)

            # Prepare the context
            context = "\n\n".join([chunk.page_content for chunk in relevant_chunks])

            # Persist changes
            vectordb.persist()

            # Query DeepSeek with a more structured prompt
            client = ollama.Client(
                host=f"http://{os.getenv('OLLAMA_HOST', 'localhost')}:{os.getenv('OLLAMA_PORT', '11434')}"
            )

            prompt = f"""Based on the following context, please answer the question.
            Stick to the information provided in the context and avoid making assumptions.

            Context: {context}

            Question: {question}

            Answer:"""

            response = client.chat(
                model=self.model_name,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )

            return response['message']['content']

        finally:
            # Clean up the database only if persistence is disabled
            if not self.persist_db and os.path.exists(self.chroma_path):
                shutil.rmtree(self.chroma_path)