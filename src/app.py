from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import gradio as gr
import os
from processors.pdf_processor import PDFProcessor
from processors.rag_processor import RAGProcessor

app = FastAPI()
# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Processor Initialization
pdf_processor = PDFProcessor()
rag_processor = RAGProcessor()
def process_and_query(file_obj, question):
    if file_obj is None:
        return "Please upload a PDF file"

    if not file_obj.name.lower().endswith('.pdf'):
        return "Please upload a PDF file"

    try:
        chunks = pdf_processor.process(file_obj)
        return rag_processor.query(question, chunks)
    except Exception as e:
        return f"An error occurred: {str(e)}"

        # Execute the query
        result = rag_processor.query(question, chunks)
        return result
    except Exception as e:
        return f"An error occurred: {str(e)}"
# Gradio Interface
interface = gr.Interface(
    fn=process_and_query,
    inputs=[
        gr.File(label="Upload a PDF", file_types=[".pdf"]),
        gr.Textbox(label="Ask a question about the document")
    ],
    outputs=gr.Textbox(label="Answer"),
    title="DocAnalyzer",
    description="Analyze PDF documents with DeepSeek R1",
)
# Mount Gradio interface on FastAPI
app = gr.mount_gradio_app(app, interface, path="/")
