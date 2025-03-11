from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import gradio as gr
import os
import tempfile
import shutil
from src.processors.factory import ProcessorFactory
from src.processors.rag_processor import RAGProcessor
from src.config.prompts import ROLE_PROMPTS

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create temporary directory for uploads
    os.makedirs("./temp_uploads", exist_ok=True)
    yield
    # Clean up on shutdown
    if os.path.exists("./temp_uploads"):
        shutil.rmtree("./temp_uploads")

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_processor = RAGProcessor()

# Global storage for tracking uploaded files
UPLOAD_DIR = "./temp_uploads"
processed_files = []

def add_file_to_context(file_obj):
    """
    Process a single file and add it to the existing context
    """
    global processed_files

    if file_obj is None:
        return "Please select a file to upload", None

    try:
        # Get file name for tracking
        file_name = os.path.basename(file_obj.name)

        # Process the file
        processor = ProcessorFactory.get_processor(file_obj)
        chunks = processor.process(file_obj)

        # Add to existing context without clearing
        rag_processor.add_document(chunks)

        # Add to tracked files
        processed_files.append(file_name)

        # Return status with current context
        return f"Added: {file_name} to context\nCurrent context: {', '.join(processed_files)}", None
    except ValueError as e:
        return f"Error: {str(e)}", None
    except Exception as e:
        return f"An error occurred while processing: {str(e)}", None

def clear_context():
    """
    Clear all files from context and reset the vector database
    """
    global processed_files

    # Reset the list of tracked files
    processed_files = []

    # Clear the vector database
    rag_processor._clean_db()

    return "Context cleared. All documents have been removed."

def query_document(question, role):
    """
    Handle document querying with role selection
    """
    if not question.strip():
        return "Please enter a question"

    if not processed_files:
        return "No documents in context. Please upload at least one document first."

    try:
        return rag_processor.query(question, role)
    except ValueError as e:
        return str(e)
    except Exception as e:
        return f"An error occurred during querying: {str(e)}"

# Create Gradio interface
with gr.Blocks(title="DocAnalyzer", theme=gr.themes.Soft()) as interface:
    gr.Markdown("# DocAnalyzer\nAnalyze documents with Large Language Models")

    with gr.Row():
        # Left column - Chat section
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(height=400)
            with gr.Row():
                with gr.Column(scale=3):
                    question_input = gr.Textbox(
                        show_label=False,
                        placeholder="Ask a question about the documents...",
                        container=False
                    )
                with gr.Column(scale=2):
                    role_input = gr.Dropdown(
                        choices=list(ROLE_PROMPTS.keys()),
                        value="default",
                        label="Analysis Role"
                    )
            query_button = gr.Button("Send")

        # Right column - Document upload section
        with gr.Column(scale=1):
            # Single file upload
            file_input = gr.File(
                label="Select Document",
                file_types=[".pdf", ".doc", ".docx"],
                file_count="single"
            )

            # Add to context button
            add_to_context_button = gr.Button("Add to Context", variant="primary")

            # Status display
            context_status = gr.Textbox(
                label="Context Status",
                value="No documents in context",
                lines=4
            )

            # Clear context button
            clear_context_button = gr.Button("Clear Context", variant="stop")

    def add_text(history, text, role):
        if not text:
            return history
        history = history + [(text, None)]
        return history

    def bot_response(history, role):
        if not history:
            return history
        user_message = history[-1][0]
        bot_message = query_document(user_message, role)
        history[-1] = (user_message, bot_message)
        return history

    # Document handling events - modificato per resettare il file_input
    add_to_context_button.click(
        fn=add_file_to_context,
        inputs=[file_input],
        outputs=[context_status, file_input]
    )

    clear_context_button.click(
        fn=clear_context,
        inputs=[],
        outputs=[context_status]
    )

    # Chat events
    question_input.submit(
            add_text,
            [chatbot, question_input, role_input],
            [chatbot]
        ).then(
            bot_response,
            [chatbot, role_input],
            [chatbot]
        ).then(lambda: "", None, [question_input])

    query_button.click(
            add_text,
            [chatbot, question_input, role_input],
            [chatbot]
        ).then(
            bot_response,
            [chatbot, role_input],
            [chatbot]
        ).then(lambda: "", None, [question_input])

app = gr.mount_gradio_app(app, interface, path="/")