from setuptools import setup, find_packages

setup(
    name="doc-analyzer",
    version="0.2.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "fastapi",
        "uvicorn",
        "gradio",
        "qdrant-client",
        "PyMuPDF",
        "python-multipart",
        "python-dotenv",
        "ollama",
        "python-docx",
        "textract",
        "pandas",
        "numpy",
        "openpyxl",
        "odfpy",
    ],
)
