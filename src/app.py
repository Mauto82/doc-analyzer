from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import gradio as gr
import os
from src.processors.factory import ProcessorFactory
from src.processors.rag_processor import RAGProcessor

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_processor = RAGProcessor()

def process_and_query(file_obj, question):
    if file_obj is None and question.strip() == "":
        return "Please upload a document or ask a question"

    try:
        if file_obj is not None:
            processor = ProcessorFactory.get_processor(file_obj)
            chunks = processor.process(file_obj)
            rag_processor.process_document(chunks)
            if not question.strip():
                return "Document processed successfully. You can now ask questions."

        if question.strip():
            return rag_processor.query(question)

    except ValueError as e:
        return str(e)
    except Exception as e:
        return f"An error occurred: {str(e)}"

interface = gr.Interface(
    fn=process_and_query,
    inputs=[
        gr.File(label="Upload a document", file_types=[".pdf", ".doc", ".docx"]),
        gr.Textbox(label="Ask a question about the document")
    ],
    outputs=gr.Textbox(label="Answer"),
    title="DocAnalyzer",
    description="Analyze documents with DeepSeek R1",
)

app = gr.mount_gradio_app(app, interface, path="/")