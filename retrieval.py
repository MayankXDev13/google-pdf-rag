from typing import Optional
from langchain_pinecone import PineconeVectorStore
from embeddings import embeddings
from config import PINECONE_INDEX_NAME


def get_retriever(k: int = 3, filename: Optional[str] = None):

    # Ensure k is valid
    if k <= 0:
        k = 3

    vectorstore = PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings,
        text_key="text",
    )

    search_kwargs = {"k": k}

    # Apply metadata filter if filename provided
    if filename:
        search_kwargs["filter"] = {"filename": {"$eq": filename}}

    return vectorstore.as_retriever(
        search_type="similarity", search_kwargs=search_kwargs
    )
