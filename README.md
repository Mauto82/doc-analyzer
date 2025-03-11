# Doc Analyzer

A web application that analyzes PDF, DOC, and DOCX documents using large language models through Ollama and RAG (Retrieval-Augmented Generation) architecture.

## Overview

Doc Analyzer enables users to:
- Upload PDF, DOC, and DOCX documents
- Add multiple documents to the context
- Ask questions about document content
- Receive AI-generated responses based on document content
- Process documents using state-of-the-art language models

The application leverages:
- Ollama for text generation and embeddings
- ChromaDB for vector storage
- LangChain for document processing
- FastAPI and Gradio for the web interface

## Project Structure

```
doc-analyzer/
├── src/                        # Source code
│   ├── app.py                  # Main FastAPI application
│   ├── config/                 # Configuration files
│   │   ├── __init__.py
│   │   └── prompts.py          # Role-based prompts configuration
│   └── processors/             # Document processors
│       ├── base/               # Base classes
│       │   └── document_processor.py
│       ├── factory.py          # Factory for processor creation
│       ├── pdf_processor.py    # PDF document handling
│       ├── word_processor.py   # Word document handling
│       └── rag_processor.py    # RAG implementation
├── tests/                      # Test files
│   ├── processors/             # Processor-specific tests
│   │   ├── test_base_processor.py
│   │   ├── test_factory.py
│   │   └── test_word_processor.py
│   └── unit/                   # Unit tests
│       ├── test_app.py
│       └── test_rag_processor.py
├── data/                       # Data directory
│   └── chroma/                 # ChromaDB storage
├── Dockerfile                  # Container definition
├── docker-compose.yml          # Container orchestration
├── docker-compose.test.yml     # Test container configuration
├── requirements.txt            # Production dependencies
├── requirements-dev.txt        # Development dependencies
└── setup.py                    # Package setup
```

## Requirements

- Docker and Docker Compose
- Ollama with your preferred language model
- 8GB RAM minimum
- 20GB disk space
- Additional system dependencies (managed by Docker):
  - poppler-utils (for PDF processing)
  - tesseract-ocr and libtesseract-dev (for text extraction)
  - antiword and unrtf (for DOC/DOCX processing)

### Installing Ollama

1. Install Ollama:
   - Linux: `curl https://ollama.ai/install.sh | sh`
   - macOS/Windows: Download from https://ollama.ai

2. Pull the language model (deepseek-r1:14b is the default, but you can use any Ollama model by updating the .env file):
```bash
ollama pull deepseek-r1:14b  # or your preferred model
```

## Quick Start with Docker

1. Clone and setup:
```bash
git clone https://github.com/Mauto82/doc-analyzer
cd doc-analyzer
cp .env.example .env
```

2. Configure `.env`:
```env
OLLAMA_HOST=host.docker.internal  # Use 'localhost' for local dev
OLLAMA_PORT=11434
CHROMA_DB_PATH=/app/data/chroma
LLM_MODEL=deepseek-r1:14b  # Use your preferred Ollama model
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
PERSIST_VECTORDB=false
```

3. Running the application:
```bash
# Start the web application
docker compose up -d

# Access the web interface at http://localhost:8000
```

4. Running tests:
```bash
# Run the test suite
docker compose -f docker-compose.test.yml up --abort-on-container-exit
```

5. Other Docker commands:
```bash
# Clean start: remove containers from old configurations
docker-compose up --remove-orphans

# View logs
docker-compose logs -f

# Rebuild without using cached images
docker-compose build --no-cache

# Restart services
docker-compose restart
```

### Deployment Without Docker
```bash
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows
pip install -r requirements.txt
pip install -r requirements-dev.txt
mkdir -p ./data/chroma
chmod -R 777 ./data
uvicorn src.app:app --reload
```

## Usage Guide

1. **Upload And Process Documents**
   - Click upload area or drag-and-drop your document
   - Supported formats: PDF, DOC, DOCX
   - Click "Add to Context" button to add the document to the current context
   - You can add multiple documents to the context one by one
   - Each document added will be shown in the Context Status area
   - The documents are processed automatically when added

2. **Managing Document Context**
   - All uploaded documents are accumulated in the context
   - Each document's content is indexed and made available for querying
   - You can view the current context in the "Context Status" area
   - To remove all documents, click the "Clear Context" button
   - Clearing the context resets the vector database completely

3. **Ask Questions**
   - Type your question in the text input
   - Select an analysis role from the dropdown:
     - Default: General document analysis
     - Legal: Legal implications and compliance analysis
     - Financial: Financial implications and economic analysis
     - Travel: Travel-related insights and recommendations
     - Technical: Technical details and implementation analysis
   - Click "Send" or press Enter
   - Receive role-specific AI-generated responses based on all documents in context

4. **Multi-Document Analysis**
   - The system automatically retrieves information from all added documents
   - You can ask questions that require information from multiple documents
   - The AI will combine relevant information from different documents to provide comprehensive answers
   - The more specific your question, the more targeted the response will be

5. **Best Practices**
   - Add related documents to the context for comprehensive analysis
   - Use clear, specific questions
   - Ask one question at a time
   - For complex multi-document scenarios, start with general questions
   - Wait for each response before asking the next question
   - If you're starting a new topic, consider clearing the context first

## Architecture

The application consists of several components:

### Web Interface (FastAPI + Gradio)
- Handles file uploads and user interactions
- Provides intuitive interface
- Manages user sessions

### Document Processors
- PDF Processor:
  - Extracts text from PDF documents using PyMuPDF
  - Splits content into manageable chunks
  - Uses LangChain for document handling
- Word Processor:
  - Processes DOC files using antiword
  - Handles DOCX files using python-docx
  - Extracts text content for analysis

### RAG Processor
- Creates embeddings using DeepSeek
- Stores vectors in ChromaDB
- Retrieves relevant content for queries
- Generates responses using DeepSeek

### Vector Store (ChromaDB)
- Stores document embeddings from multiple documents
- Enables semantic search across all documents
- Maintains document-query relevance

## Troubleshooting

### Database Issues
If you get "readonly database" error in local development:
```bash
chmod -R 777 ./data
rm -rf ./data/chroma/*
```

### Ollama Connection
1. Verify Ollama is running: `ollama list`
2. Check service: `curl http://localhost:11434/api/tags`
3. Check `.env` configuration matches your setup
4. For Docker Desktop users, ensure `host.docker.internal` is used

### Memory Issues
- Docker Desktop: Increase memory limit (Preferences → Resources → Memory)
- Recommended minimum: 8GB
- If needed, reduce chunk size in `pdf_processor.py`:
```python
chunk_size=500  # Decrease if experiencing memory issues
chunk_overlap=100
```

### Document Processing Issues
- PDF files: Ensure the PDF is not password-protected
- DOC files: File must be readable by antiword
- DOCX files: File must be a valid Office Open XML format
- If text extraction fails, try converting the document to PDF
- If adding a document doesn't update the context, try clearing the context and adding it again

## Contributing

1. Fork repository
2. Create feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push branch: `git push origin feature-name`
5. Submit Pull Request

### Development Guidelines
- Follow PEP 8 style guide
- Add tests for new features
- Update documentation
- Keep commits focused and clean

## License

MIT License - see LICENSE file for details.