import hashlib
from typing import Optional
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from config import PINECONE_API_KEY, INDEX_NAME
from embeddings import embeddings
from logger import logger

pc = Pinecone(api_key=PINECONE_API_KEY)


def create_index():
    try:
        existing = pc.list_indexes()
        # list_indexes() may return an object or list depending on client version
        names = getattr(existing, "names", lambda: existing)()
    except Exception:
        # fallback: try treating as list
        try:
            names = list(existing)
        except Exception:
            names = []

    if INDEX_NAME not in names:
        try:
            pc.create_index(
                name=INDEX_NAME,
                dimension=768,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            logger.info("Created Pinecone index %s", INDEX_NAME)
        except Exception as e:
            logger.error("Failed to create index %s: %s", INDEX_NAME, e)
            raise


def get_vectorstore() -> PineconeVectorStore:
    create_index()
    return PineconeVectorStore(
        index=pc.Index(INDEX_NAME), embedding=embeddings, text_key="text"
    )


def compute_file_id(filename: str, file_hash: str) -> str:
    return f"{filename}_{file_hash}"


def file_exists_in_index(filename: str, file_hash: str) -> bool:
    index = pc.Index(INDEX_NAME)
    file_id = compute_file_id(filename, file_hash)

    try:
        fetch_response = index.fetch(ids=[file_id])
        return file_id in getattr(fetch_response, "vectors", fetch_response.get("vectors", {}))
    except Exception:
        logger.exception("Error checking if file exists in index: %s", file_id)
        return False


def add_documents(
    documents: list[Document],
    filename: str,
    file_hash: str,
    batch_size: int = 100,
) -> bool:
    vectorstore = get_vectorstore()

    for i, doc in enumerate(documents):
        doc.metadata["filename"] = filename
        doc.metadata["file_hash"] = file_hash
        doc.metadata["file_id"] = compute_file_id(filename, file_hash)
        doc.metadata["chunk_index"] = i

    try:
        vectorstore.add_documents(documents=documents, batch_size=batch_size)
        logger.info("Added %d documents for file %s", len(documents), filename)
        return True
    except Exception as e:
        logger.exception("Error adding documents for %s: %s", filename, e)
        return False


def delete_file(filename: str, file_hash: Optional[str] = None) -> bool:
    """Delete vectors by file_id when file_hash provided, otherwise by filename metadata."""
    index = pc.Index(INDEX_NAME)

    try:
        if file_hash:
            file_id = compute_file_id(filename, file_hash)
            index.delete(filter={"file_id": {"$eq": file_id}})
            logger.info("Deleted vectors for file_id %s", file_id)
        else:
            index.delete(filter={"filename": {"$eq": filename}})
            logger.info("Deleted vectors for filename %s", filename)
        return True
    except Exception as e:
        logger.exception("Error deleting file from vectorstore %s: %s", filename, e)
        return False


def list_indexed_files() -> list[str]:
    index = pc.Index(INDEX_NAME)
    try:
        stats = index.describe_index_stats()
    except Exception as e:
        logger.exception("Failed to describe index stats: %s", e)
        return []

    filenames = set()
    try:
        if "namespaces" in stats and "" in stats["namespaces"]:
            namespace_stats = stats["namespaces"][""]
            if "vectors" in namespace_stats and namespace_stats["vectors"] > 0:
                query_response = index.query(
                    vector=[0.0] * 768, top_k=10000, include_metadata=True
                )
                for match in query_response.get("matches", []):
                    metadata = match.get("metadata", {})
                    if "filename" in metadata:
                        filenames.add(metadata["filename"])
    except Exception as e:
        logger.exception("Error querying index for filenames: %s", e)

    return list(filenames)
