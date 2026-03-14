from typing import List
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from config import PINECONE_API_KEY, PINECONE_INDEX_NAME
from embeddings import embeddings

pc = Pinecone(api_key=PINECONE_API_KEY)


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
        index = pc.Index(PINECONE_INDEX_NAME)
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
        index = pc.Index(PINECONE_INDEX_NAME)

        index.delete(filter={"source": {"$eq": file_name}})

    except Exception as e:
        raise RuntimeError(f"Failed to delete documents for file {file_name}: {e}")


def list_indexed_files() -> list[str]:
    """Get list of unique filenames from indexed documents in Pinecone"""
    try:
        index = pc.Index(PINECONE_INDEX_NAME)
        stats = index.describe_index_stats()

        filenames = set()
        if "namespaces" in stats:
            for ns_stats in stats["namespaces"].values():
                if "metadata" in ns_stats:
                    for key in ns_stats.get("metadata", {}).keys():
                        if key == "filename":
                            # Query to get unique filenames
                            pass

        query_response = index.query(
            vector=[0.0] * 768,
            top_k=10000,
            include_metadata=True,
            include_values=False,
        )

        filenames = set()
        for match in query_response.get("matches", []):
            if "metadata" in match and "filename" in match["metadata"]:
                filenames.add(match["metadata"]["filename"])

        return sorted(list(filenames))
    except Exception as e:
        print(f"Error listing indexed files: {e}")
        return []
