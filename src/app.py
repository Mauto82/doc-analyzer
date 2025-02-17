from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import gradio as gr
import os
from src.processors.factory import ProcessorFactory
from src.processors.rag_processor import RAGProcessor

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_processor = RAGProcessor()

def process_and_query(file_obj, question):
    if file_obj is None:
        return "Please upload a PDF file"

    try:
        processor = ProcessorFactory.get_processor(file_obj)
        chunks = processor.process(file_obj)
        return rag_processor.query(question, chunks)
    except ValueError as e:
        return str(e)
    except Exception as e:
        return f"An error occurred: {str(e)}"

interface = gr.Interface(
    fn=process_and_query,
    inputs=[
        gr.File(label="Upload a document", file_types=[".pdf"]),
        gr.Textbox(label="Ask a question about the document")
    ],
    outputs=gr.Textbox(label="Answer"),
    title="DocAnalyzer",
    description="Analyze documents with DeepSeek R1",
)

app = gr.mount_gradio_app(app, interface, path="/")