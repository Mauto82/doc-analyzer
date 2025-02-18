from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import gradio as gr
from src.processors.factory import ProcessorFactory
from src.processors.rag_processor import RAGProcessor
from src.config.prompts import ROLE_PROMPTS

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
    gr.Markdown("# DocAnalyzer\nAnalyze documents with DeepSeek R1")

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
            file_input = gr.File(
                label="Upload Document",
                file_types=[".pdf", ".doc", ".docx"]
            )
            upload_button = gr.Button("Process Document")
            upload_output = gr.Textbox(label="Status")

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

    # Event handlers
    upload_button.click(
        fn=process_document,
        inputs=[file_input],
        outputs=[upload_output]
    )

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