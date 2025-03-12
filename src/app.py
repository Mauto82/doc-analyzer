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
from src.processors.code_processor import CodeProcessor

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

# Global storage for tracking uploaded files with their document IDs
UPLOAD_DIR = "./temp_uploads"
processed_files_map = {}  # {file_name: [document_ids]}

# Get available LLM models from Ollama
available_models = rag_processor.get_available_models()
default_model = os.getenv('LLM_MODEL')

# If the default model is not in the list, add it at the top
if default_model not in available_models:
    available_models.insert(0, default_model)
else:
    # Make sure the default model is at the top of the list
    available_models.remove(default_model)
    available_models.insert(0, default_model)

def add_file_to_context(file_obj):
    """
    Process a single file and add it to the existing context
    """
    global processed_files_map

    if file_obj is None:
        return [["No files in context"]], None

    try:
        # Get file name for tracking
        file_name = os.path.basename(file_obj.name)

        # Process the file
        processor = ProcessorFactory.get_processor(file_obj)
        chunks = processor.process(file_obj)

        # Add to existing context without clearing
        ids = rag_processor.add_document(chunks)

        # Add to tracked files with document IDs
        processed_files_map[file_name] = ids

        # Return status with current context as a table data
        files_table = [[file] for file in processed_files_map.keys()]
        return files_table, None
    except ValueError as e:
        return [[f"Error: {str(e)}"]], None
    except Exception as e:
        return [[f"An error occurred while processing: {str(e)}"]], None

def remove_file_from_context(file_name):
    """
    Remove a single file from the context
    """
    global processed_files_map

    if not file_name or file_name not in processed_files_map:
        return [["No files in context"]]

    # Get document IDs for this file
    doc_ids = processed_files_map[file_name]

    # Remove from vector database
    success = rag_processor.remove_document(doc_ids)

    if success:
        # Remove from tracked files
        del processed_files_map[file_name]

        if not processed_files_map:
            return [["No files in context"]]
        else:
            # Return updated context
            files_table = [[file] for file in processed_files_map.keys()]
            return files_table
    else:
        # Keep the file in the list if removal failed
        files_table = [[file] for file in processed_files_map.keys()]
        return files_table

def clear_context():
    """
    Clear all files from context and reset the vector database
    """
    global processed_files_map

    # Reset the tracked files
    processed_files_map = {}

    # Clear the vector database
    rag_processor._clean_db()

    return [["Context cleared. All documents have been removed."]]

def query_document(question, role, model=None):
    """
    Handle document querying with role selection and model selection
    """
    if not question.strip():
        return "Please enter a question"

    if not processed_files_map:
        return "No documents in context. Please upload at least one document first."

    try:
        return rag_processor.query(question, role, model)
    except ValueError as e:
        return str(e)
    except Exception as e:
        return f"An error occurred during querying: {str(e)}"

# Get the common code file extensions for the UI (a subset of supported extensions)
COMMON_CODE_EXTENSIONS = ['.py', '.js', '.java', '.c', '.cpp', '.html', '.css', '.php', '.go', '.ts', '.rb', '.json', '.xml', '.md', '.yaml', '.yml']

# Create Gradio interface
with gr.Blocks(title="DocAnalyzer", theme=gr.themes.Soft()) as interface:
    gr.Markdown("# DocAnalyzer\nAnalyze documents with local Large Language Models")

    with gr.Row():
        # Left column - Chat interface
        with gr.Column(scale=2):
            # Top row - Dropdown selectors
            with gr.Row():
                with gr.Column(scale=1):
                    role_input = gr.Dropdown(
                        choices=list(ROLE_PROMPTS.keys()),
                        value="default",
                        label="Analysis Role"
                    )
                with gr.Column(scale=1):
                    model_input = gr.Dropdown(
                        choices=available_models,
                        value=default_model,
                        label="LLM Model"
                    )

            # Middle row - Chat section
            with gr.Row():
                chatbot = gr.Chatbot(height=400)

            # Bottom row - Input and Send button
            with gr.Row():
                with gr.Column(scale=8):
                    question_input = gr.Textbox(
                        show_label=False,
                        placeholder="Ask a question about the documents...",
                        container=False
                    )
                with gr.Column(scale=1):
                    query_button = gr.Button("Send")

        # Right column - Document upload section
        with gr.Column(scale=1):
            # Single file upload
            file_input = gr.File(
                label="Select Document",
                file_types=[".pdf", ".doc", ".docx", ".txt", ".rtf"] + COMMON_CODE_EXTENSIONS,
                file_count="single"
            )

            # Add to context button
            add_to_context_button = gr.Button("Add to Context", variant="primary")

            # Status display with table component
            context_status = gr.Dataframe(
                headers=["Files in Context"],
                datatype=["str"],
                row_count=10,
                col_count=(1, "fixed"),
                value=[["No documents in context"]],
                height=200
            )

            # Dropdown for selecting a file to remove
            file_to_remove = gr.Dropdown(
                choices=[],
                label="Select file to remove",
                interactive=True
            )

            # Remove file and clear context buttons
            with gr.Row():
                remove_file_button = gr.Button("Remove Selected File")
                clear_context_button = gr.Button("Clear Context", variant="stop")

    def add_text(history, text, role):
        if not text:
            return history
        history = history + [(text, None)]
        return history

    def bot_response(history, role, model):
        if not history:
            return history
        user_message = history[-1][0]
        bot_message = query_document(user_message, role, model)
        history[-1] = (user_message, bot_message)
        return history

    def update_file_dropdown():
        return gr.Dropdown(choices=list(processed_files_map.keys()))

    # Document handling events
    add_to_context_button.click(
            fn=add_file_to_context,
            inputs=[file_input],
            outputs=[context_status, file_input]
        ).then(
            fn=update_file_dropdown,
            inputs=[],
            outputs=[file_to_remove]
        )

    remove_file_button.click(
            fn=remove_file_from_context,
            inputs=[file_to_remove],
            outputs=[context_status]
        ).then(
            fn=update_file_dropdown,
            inputs=[],
            outputs=[file_to_remove]
        )

    clear_context_button.click(
            fn=clear_context,
            inputs=[],
            outputs=[context_status]
        ).then(
            fn=update_file_dropdown,
            inputs=[],
            outputs=[file_to_remove]
        )

    # Chat events
    question_input.submit(
            add_text,
            [chatbot, question_input, role_input],
            [chatbot]
        ).then(
            bot_response,
            [chatbot, role_input, model_input],
            [chatbot]
        ).then(lambda: "", None, [question_input])

    query_button.click(
            add_text,
            [chatbot, question_input, role_input],
            [chatbot]
        ).then(
            bot_response,
            [chatbot, role_input, model_input],
            [chatbot]
        ).then(lambda: "", None, [question_input])

app = gr.mount_gradio_app(app, interface, path="/")