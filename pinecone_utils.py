from typing import List
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from config import PINECONE_API_KEY, PINECONE_INDEX_NAME
from embeddings import embeddings


def add_documents(chunks: List, batch_size: int = 50):
    """
    Store document chunks in Pinecone using batch insertion.
    """

    if not chunks:
        raise ValueError("No chunks provided for embedding")

    try:

        vector_store = PineconeVectorStore(
            index_name=PINECONE_INDEX_NAME,
            embedding=embeddings,
        )

        total_chunks = len(chunks)

        for i in range(0, total_chunks, batch_size):

            batch = chunks[i : i + batch_size]

            vector_store.add_documents(batch)

        return {"status": "success", "chunks_stored": total_chunks}

    except Exception as e:
        raise RuntimeError(f"Failed to store documents in Pinecone: {e}")


def delete_documents(ids: list[str]):
    """
    Delete vectors using IDs.
    """

    if not ids:
        raise ValueError("No IDs provided for deletion")

    try:
        index = pc.Index(INDEX_NAME)
        index.delete(ids=ids)

    except Exception as e:
        raise RuntimeError(f"Failed to delete documents: {e}")


def delete_by_source(file_name: str):
    """
    Delete all vectors belonging to a file.
    """

    if not file_name:
        raise ValueError("file_name cannot be empty")

    try:
        index = pc.Index(INDEX_NAME)

        index.delete(filter={"source": {"$eq": file_name}})

    except Exception as e:
        raise RuntimeError(f"Failed to delete documents for file {file_name}: {e}")
