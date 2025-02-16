# Doc Analyzer

A web application that analyzes PDF documents using DeepSeek R1 language model and RAG (Retrieval-Augmented Generation) architecture.

## Overview

Doc Analyzer is a tool that allows users to:
- Upload PDF documents
- Ask questions about the document content
- Receive AI-generated responses based on the document's content
- Process documents using state-of-the-art language models

The application uses:
- DeepSeek R1 for text generation and embeddings
- ChromaDB for vector storage
- LangChain for document processing
- FastAPI and Gradio for the web interface

## Requirements

- Docker and Docker Compose
- Ollama installed on the host machine
- At least 8GB of RAM
- 20GB of free disk space
- DeepSeek R1 1.5b (or later) model installed via Ollama

### Installing Ollama and DeepSeek

1. Install Ollama following the official instructions for your OS:
   - Linux: `curl https://ollama.ai/install.sh | sh`
   - macOS: Download from https://ollama.ai
   - Windows: Download from https://ollama.ai

2. Pull the DeepSeek model:
```bash
ollama pull deepseek-r1:1.5b
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Mauto82/doc-analyzer
cd doc-analyzer
```

2. Create the environment configuration:
```bash
cp .env.example .env
```

3. Build and start the application:
```bash
docker-compose build
docker-compose up -d
```

4. Access the web interface at:
```
http://localhost:8000
```

## Configuration

The application can be configured through the `.env` file:

```env
OLLAMA_HOST=host.docker.internal
OLLAMA_PORT=11434
CHROMA_DB_PATH=/app/data/chroma
DEEPSEEK_MODEL=deepseek-r1:1.5b
```

### Configuration Options

- `OLLAMA_HOST`: Hostname for the Ollama service
  - Use `host.docker.internal` for Docker Desktop
  - Use `localhost` for local development
  - Use the machine's IP for remote Ollama instances

- `OLLAMA_PORT`: Port for the Ollama service (default: 11434)

- `CHROMA_DB_PATH`: Path to store ChromaDB files
  - Keep the default `/app/data/chroma` for Docker setup
  - Change to `./data/chroma` for local development

- `DEEPSEEK_MODEL`: DeepSeek model to use
  - Available options: `deepseek-r1:1.5b`, `deepseek-r1:14b`, etc

## Usage

1. Open the web interface at `http://localhost:8000`

2. Upload a PDF document:
   - Click on the upload area or drag and drop a PDF file
   - Wait for the upload to complete

3. Ask questions:
   - Type your question in the text input
   - Click "Submit" or press Enter
   - Wait for the AI-generated response

4. Best practices:
   - Use clear, specific questions
   - Ask one question at a time
   - For complex documents, start with general questions before specific ones
   - Wait for each response before asking the next question

## Troubleshooting

### Database Write Permission Error

If you encounter a "readonly database" error:

```bash
# Stop the containers
docker-compose down

# Set correct permissions
mkdir -p ./data/chroma
chmod -R 777 ./data

# Clean existing database files
rm -rf ./data/chroma/*

# Rebuild and restart
docker-compose build --no-cache
docker-compose up -d
```

### Connection to Ollama Failed

If the application can't connect to Ollama:

1. Verify Ollama is running:
```bash
ollama list
```

2. Check the Ollama service:
```bash
curl http://localhost:11434/api/tags
```

3. Verify your `.env` configuration matches your Ollama setup

### Memory Issues

If the application crashes or becomes unresponsive:

1. Increase Docker memory limit:
   - Docker Desktop: Preferences → Resources → Memory
   - Recommended: At least 8GB

2. Reduce chunk size in `pdf_processor.py`:
```python
chunk_size=500  # Decrease this value
chunk_overlap=100  # Adjust accordingly
```

## Local Development (without Docker)

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment:
```bash
cp .env.example .env
# Edit .env to use localhost instead of host.docker.internal
```

4. Start the application:
```bash
uvicorn src.app:app --reload
```

### Development with Docker

For development with Docker:

```bash
# Start with logs
docker-compose up

# View logs
docker-compose logs -f

# Rebuild after changes
docker-compose build

# Restart services
docker-compose restart

# Stop all services
docker-compose down
```

## Architecture

The application consists of several components:

1. **Web Interface** (FastAPI + Gradio):
   - Handles file uploads and user interactions
   - Provides a simple, intuitive interface

2. **PDF Processor**:
   - Extracts text from PDF documents
   - Splits content into manageable chunks
   - Uses LangChain for document handling

3. **RAG Processor**:
   - Creates embeddings using DeepSeek
   - Stores vectors in ChromaDB
   - Retrieves relevant content for queries
   - Generates responses using DeepSeek

4. **Vector Store** (ChromaDB):
   - Stores document embeddings
   - Enables semantic search
   - Maintains document-query relevance

## Running Tests

Tests can be run using Docker:

```bash
# Run all tests
docker-compose run test

# Run specific test file
docker-compose run test pytest tests/unit/test_app.py -v

# Run tests with coverage
docker-compose run test pytest --cov=src tests/

# Interactive debug
docker-compose run test bash
pytest -v  # inside container
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit your changes: `git commit -am 'Add feature'`
4. Push to the branch: `git push origin feature-name`
5. Submit a Pull Request

### Development Guidelines

- Follow PEP 8 style guide
- Add tests for new features
- Update documentation as needed
- Keep commits focused and clean

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- DeepSeek team for the language model
- LangChain for the document processing framework
- ChromaDB for the vector storage solution
- FastAPI and Gradio for the web framework
