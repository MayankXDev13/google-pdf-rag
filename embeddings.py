from langchain_google_genai import GoogleGenerativeAIEmbeddings
from config import GOOGLE_API_KEY, EMBED_MODEL

embeddings = GoogleGenerativeAIEmbeddings(
    model=EMBED_MODEL,
    google_api_key=GOOGLE_API_KEY,
    task_type="retrieval_document",

)

query_embeddings = GoogleGenerativeAIEmbeddings(
    model=EMBED_MODEL,
    google_api_key=GOOGLE_API_KEY,
    task_type="retrieval_query",
)