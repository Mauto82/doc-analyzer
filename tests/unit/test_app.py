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

# Global storage for uploaded files
UPLOAD_DIR = "./temp_uploads"
uploaded_files = []

def add_file(file_obj):
    """Add a file to the list of uploaded files"""
    global uploaded_files

    if file_obj is None:
        return "No file selected"

    # Save file to temp directory
    file_name = os.path.basename(file_obj.name)
    target_path = os.path.join(UPLOAD_DIR, file_name)

    # Copy the file
    shutil.copy(file_obj.name, target_path)

    # Add to list of uploaded files
    uploaded_files.append(target_path)

    # Return status with all current files
    file_list = [os.path.basename(f) for f in uploaded_files]
    return f"File {file_name} added. Current files: {', '.join(file_list)}"

def process_all_documents(clean_db=True):
    """Process all uploaded documents"""
    global uploaded_files

    if not uploaded_files:
        return "No documents to process. Please upload files first."

    try:
        all_chunks = []
        processed_files = []
        failed_files = []

        for file_path in uploaded_files:
            try:
                processor = ProcessorFactory.get_processor(file_path)
                chunks = processor.process(file_path)
                all_chunks.extend(chunks)
                processed_files.append(os.path.basename(file_path))
            except Exception as e:
                failed_files.append((os.path.basename(file_path), str(e)))

        if not all_chunks:
            if failed_files:
                return f"Failed to process files: {', '.join(f[0] for f in failed_files)}"
            return "No content was extracted from the uploaded documents"

        # Process chunks
        if clean_db:
            rag_processor.process_document(all_chunks)
        else:
            rag_processor.add_document(all_chunks)

        success_msg = f"Successfully processed {len(processed_files)} document(s)"
        if failed_files:
            return f"{success_msg}. Failed to process: {', '.join(f[0] for f in failed_files)}"
        return f"{success_msg}. You can now ask questions."
    except Exception as e:
        return f"Error processing documents: {str(e)}"

def clear_uploads():
    """Clear the list of uploaded files"""
    global uploaded_files
    uploaded_files = []

    # Clean the upload directory
    for file in os.listdir(UPLOAD_DIR):
        os.remove(os.path.join(UPLOAD_DIR, file))

    return "All uploaded files have been cleared."

def clear_knowledge_base():
    """Clear the knowledge base"""
    rag_processor._clean_db()
    return "Knowledge base has been cleared."

def query_document(question, role):
    """Handle document querying with role selection"""
    if not question.strip():
        return "Please enter a question"

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
                        placeholder="Ask a question about the document...",
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
            # Document upload - single file at a time
            file_input = gr.File(
                label="Select Document",
                file_types=[".pdf", ".doc", ".docx"],
                file_count="single"
            )
            add_file_button = gr.Button("Add Document to List")

            # Status display
            status_output = gr.Textbox(label="Status", value="No files uploaded")

            # Processing controls
            gr.Markdown("### Process Documents")
            with gr.Row():
                process_button = gr.Button("Process All (Replace KB)", variant="primary")
                add_to_kb_button = gr.Button("Add All to Knowledge Base")

            # Clean up controls
            gr.Markdown("### Clean Up")
            with gr.Row():
                clear_uploads_button = gr.Button("Clear File List")
                clear_kb_button = gr.Button("Clear Knowledge Base")

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

    # Document handling events
    add_file_button.click(
        fn=add_file,
        inputs=[file_input],
        outputs=[status_output]
    )

    process_button.click(
        fn=lambda: process_all_documents(clean_db=True),
        inputs=[],
        outputs=[status_output]
    )

    add_to_kb_button.click(
        fn=lambda: process_all_documents(clean_db=False),
        inputs=[],
        outputs=[status_output]
    )

    clear_uploads_button.click(
        fn=clear_uploads,
        inputs=[],
        outputs=[status_output]
    )

    clear_kb_button.click(
        fn=clear_knowledge_base,
        inputs=[],
        outputs=[status_output]
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