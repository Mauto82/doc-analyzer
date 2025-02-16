from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
import ollama
import os
import shutil
import tempfile

class RAGProcessor:
    def __init__(self):
        self.model_name = os.getenv('DEEPSEEK_MODEL', 'deepseek-r1:1.5b')
        self.base_chroma_path = os.getenv('CHROMA_DB_PATH', './data/chroma')
        self.embeddings = OllamaEmbeddings(
            model=self.model_name,
            base_url=f"http://{os.getenv('OLLAMA_HOST', 'localhost')}:{os.getenv('OLLAMA_PORT', '11434')}"
        )

    def query(self, question, chunks):
        # Create a unique temporary directory for this query
        temp_dir = tempfile.mkdtemp(dir=os.path.dirname(self.base_chroma_path))

        try:
            # Create the vector database in the temporary directory
            vectordb = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=temp_dir
            )

            # Retrieve the most relevant chunks
            retriever = vectordb.as_retriever()
            relevant_chunks = retriever.get_relevant_documents(question)

            # Prepare the context
            context = "\n\n".join([chunk.page_content for chunk in relevant_chunks])

            # Close the vector database
            vectordb.persist()
            del vectordb

            # Query DeepSeek
            client = ollama.Client(
                host=f"http://{os.getenv('OLLAMA_HOST', 'localhost')}:{os.getenv('OLLAMA_PORT', '11434')}"
            )
            response = client.chat(
                model=self.model_name,
                messages=[{
                    "role": "user",
                    "content": f"Question: {question}\n\nContext: {context}"
                }]
            )

            return response['message']['content']

        finally:
            # Clean up the temporary directory
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
