import hashlib
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from logger import logger


def load_pdf(file_data: bytes) -> list[Document]:
    """
    Load a PDF from uploaded bytes and return LangChain Documents.
    """

    # Create temporary PDF file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_data)
        tmp_path = tmp.name

    try:
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
    finally:
        # Remove temp file after loading
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                logger.exception("Failed to remove temp file: %s", tmp_path)

    return documents


def get_file_hash(data: bytes) -> str:
    """
    Generate a unique hash for the uploaded file.
    Useful for caching or deduplication.
    """
    return hashlib.md5(data).hexdigest()


def get_filename(path: str) -> str:
    """
    Extract filename from file path.
    """
    return os.path.basename(path)


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 800,
    overlap: int = 100
) -> list[Document]:
    """
    Split documents into smaller chunks for embeddings.
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        add_start_index=True,
    )

    chunks = text_splitter.split_documents(documents)

    return chunks
