from google import genai
from config import GOOGLE_API_KEY, LLM_MODEL
from retrieval import retrieve, format_context

client = genai.Client(api_key=GOOGLE_API_KEY)


def ask(question: str, k: int = 3, filename: str | None = None, show_sources: bool = True) -> dict:
    matches = retrieve(question, k=k, filename=filename)

    if not matches:
        return {
            "answer": "No relevant context found.",
            "sources": []
        }

    context_text = format_context(matches)

    prompt = f"""Answer the question using the context below.

Context:
{context_text}

Question:
{question}

Answer:
"""

    response = client.models.generate_content(
        model=LLM_MODEL,
        contents=prompt
    )

    sources = []
    for match in matches:
        sources.append({
            "filename": match["filename"],
            "page": match["page"],
            "score": match["score"],
            "text": match["text"][:200] + "..." if len(match["text"]) > 200 else match["text"]
        })

    return {
        "answer": response.text,
        "sources": sources
    }