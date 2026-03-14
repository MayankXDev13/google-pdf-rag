from typing import Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from retrieval import get_retriever
from config import LLM_MODEL, GOOGLE_API_KEY

llm = ChatGoogleGenerativeAI(
    model=LLM_MODEL,
    google_api_key=GOOGLE_API_KEY,
    temperature=0.2,  # lower temperature reduces hallucination
)


system_prompt = """
You are a helpful AI assistant.

Answer the user's question strictly using the provided context.

Context:
{context}

Rules:
- If the answer is not in the context, say "I could not find the answer in the provided documents."
- Keep answers concise and factual.
- Do not invent information.
"""


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)


def chat(question: str, k: int = 3, filename: Optional[str] = None) -> dict:

    # Create retriever
    retriever = get_retriever(k=k, filename=filename)

    # Create document chain
    combine_docs_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt,
    )

    # Create retrieval chain
    retrieval_chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=combine_docs_chain,
    )

    # Run chain
    result = retrieval_chain.invoke({"input": question})

    # Extract sources
    sources = []
    for doc in result.get("context", []):

        preview = doc.page_content[:200]
        if len(doc.page_content) > 200:
            preview += "..."

        sources.append(
            {
                "filename": doc.metadata.get("filename", "unknown"),
                "page": doc.metadata.get("page", 1),
                "text": preview,
            }
        )

    return {
        "question": question,
        "answer": result.get("answer", "No answer generated."),
        "sources": sources,
    }
