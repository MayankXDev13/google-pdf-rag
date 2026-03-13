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
    temperature=0.7,
)

system_prompt = """You are a helpful assistant. Use the context provided to answer the user's question.
If you cannot find the answer in the context, say so. Be concise and accurate."""

prompt = ChatPromptTemplate.from_messages(
    [("system", system_prompt), ("human", "{input}")]
)


def ask(question: str, k: int = 3, filename: Optional[str] = None) -> dict:
    retriever = get_retriever(k=k, filename=filename)

    combine_docs_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    retrieval_chain = create_retrieval_chain(
        retriever=retriever, combine_docs_chain=combine_docs_chain
    )

    result = retrieval_chain.invoke({"input": question})

    sources = []
    for doc in result.get("context", []):
        sources.append(
            {
                "filename": doc.metadata.get("filename"),
                "page": doc.metadata.get("page", 1),
                "text": (
                    doc.page_content[:200] + "..."
                    if len(doc.page_content) > 200
                    else doc.page_content
                ),
            }
        )

    return {
        "answer": result["answer"],
        "sources": sources,
    }
