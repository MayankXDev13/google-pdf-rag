import hashlib
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def load_pdf(file_data: bytes) -> list[Document]:
    """
    Load a PDF from uploaded bytes and return docs.
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
                pass

    return documents


from langchain.text_splitter import RecursiveCharacterTextSplitter


async def chunk_documents(documents, filename):

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    chunks = []
    chunk_id = 0

    for doc in documents:

        splits = splitter.split_text(doc.page_content)

        for split in splits:

            chunks.append(
                {
                    "text": split,
                    "metadata": {
                        "filename": filename,
                        "page": doc.metadata.get("page", 1),
                        "chunk_id": f"{filename}_{chunk_id}",
                    },
                }
            )

            chunk_id += 1

    return chunks
