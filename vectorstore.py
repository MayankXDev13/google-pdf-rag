import hashlib
from typing import Optional
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from config import PINECONE_API_KEY, INDEX_NAME
from embeddings import embeddings

pc = Pinecone(api_key=PINECONE_API_KEY)


def create_index():
    if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=INDEX_NAME,
            dimension=768,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )


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
        return file_id in fetch_response.vectors
    except Exception:
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
        return True
    except Exception as e:
        print(f"Error adding documents: {e}")
        return False


def delete_file(filename: str, file_hash: str) -> bool:
    index = pc.Index(INDEX_NAME)
    file_id = compute_file_id(filename, file_hash)

    try:
        index.delete(filter={"file_id": {"$eq": file_id}})
        return True
    except Exception as e:
        print(f"Error deleting file: {e}")
        return False


def list_indexed_files() -> list[str]:
    index = pc.Index(INDEX_NAME)
    stats = index.describe_index_stats()

    filenames = set()
    if "namespaces" in stats and "" in stats["namespaces"]:
        namespace_stats = stats["namespaces"][""]
        if "vectors" in namespace_stats and namespace_stats["vectors"] > 0:
            query_response = index.query(
                vector=[0.0] * 768, top_k=10000, include_metadata=True
            )
            for match in query_response["matches"]:
                if "filename" in match["metadata"]:
                    filenames.add(match["metadata"]["filename"])

    return list(filenames)
