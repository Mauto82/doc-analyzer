from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import gradio as gr
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

def process_document(file_obj):
    """Handle document upload and processing"""
    if file_obj is None:
        return "Please upload a document"

    try:
        processor = ProcessorFactory.get_processor(file_obj)
        chunks = processor.process(file_obj)
        rag_processor.process_document(chunks)
        return "Document processed successfully. You can now ask questions."
    except ValueError as e:
        return str(e)
    except Exception as e:
        return f"An error occurred during document processing: {str(e)}"

def query_document(question):
    """Handle document querying"""
    if not question.strip():
        return "Please enter a question"

    try:
        return rag_processor.query(question)
    except ValueError as e:
        return str(e)
    except Exception as e:
        return f"An error occurred during querying: {str(e)}"

# Create Gradio interface
with gr.Blocks(title="DocAnalyzer") as interface:
    gr.Markdown("# DocAnalyzer\nAnalyze documents with DeepSeek R1")

    with gr.Row():
        # Document upload section
        with gr.Column():
            file_input = gr.File(label="Upload a document", file_types=[".pdf", ".doc", ".docx"])
            upload_button = gr.Button("Process Document")
            upload_output = gr.Textbox(label="Upload Status")

        # Query section
        with gr.Column():
            question_input = gr.Textbox(label="Ask a question about the document", lines=2)
            query_button = gr.Button("Ask")
            answer_output = gr.Textbox(label="Answer")

    # Set up event handlers
    upload_button.click(
        fn=process_document,
        inputs=[file_input],
        outputs=[upload_output]
    )

    query_button.click(
        fn=query_document,
        inputs=[question_input],
        outputs=[answer_output]
    )

app = gr.mount_gradio_app(app, interface, path="/")