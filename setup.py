from setuptools import setup, find_packages

setup(
    name="doc-analyzer",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "fastapi",
        "uvicorn",
        "gradio",
        "langchain",
        "chromadb",
        "PyMuPDF",
        "python-multipart",
        "python-dotenv",
        "ollama",
        "langchain-community",
        "pandas",
        "numpy",
        "openpyxl",
        "odfpy"
    ],
)