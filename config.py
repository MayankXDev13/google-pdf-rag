import os
from dotenv import load_dotenv
from typing import List

load_dotenv()

INDEX_NAME = "pdf-rag"
INDEX_PREFIX = "pdf-rag"

EMBED_MODEL = os.getenv("EMBED_MODEL", "models/gemini-embedding-001")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-1.5-flash-002")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
S3_REGION = os.getenv("S3_REGION", "us-east-1")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")


def validate_env(required: List[str] | None = None) -> None:
    """Validate required environment variables and raise RuntimeError with helpful messages."""
    if required is None:
        required = [
            "GOOGLE_API_KEY",
            "PINECONE_API_KEY",
            "S3_BUCKET_NAME",
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
        ]

    missing = [name for name in required if not os.getenv(name)]
    if missing:
        raise RuntimeError(
            "Missing required environment variables: " + ", ".join(missing)
        )


# Validate on import in production usage
try:
    if os.getenv("SKIP_ENV_VALIDATION", "false").lower() not in ["1", "true", "yes"]:
        validate_env()
except RuntimeError:
    # Allow the app to import in tooling contexts but re-raise for actual runs
    raise
