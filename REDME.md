# PDF RAG

Lightweight PDF Retrieval-Augmented Generation (RAG) service using LangChain, Pinecone, and Google Generative AI. This repo provides a FastAPI-based HTTP API to upload PDFs, index their text into a vector store, and ask questions using an LLM with retrieved context.

Key features
- Upload PDF files and store the original file in S3
- Split PDFs into chunks and index embeddings in Pinecone (vectorstore)
- Query indexed documents to get LLM answers with source snippets
- Simple FastAPI endpoints for ingest, query, file listing and deletion

Quick start

1. Create a Python 3.11+ virtual environment and install deps:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Set environment variables (example .env file):

```
# Google generative model key
GOOGLE_API_KEY=your-google-api-key

# Pinecone
PINECONE_API_KEY=your-pinecone-api-key

# AWS S3
S3_BUCKET_NAME=your-bucket
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
S3_REGION=us-east-1

# Optional model overrides
EMBED_MODEL=models/gemini-embedding-001
LLM_MODEL=gemini-1.5-flash-002

# For local import without env validation set:
SKIP_ENV_VALIDATION=true
```

Note: `config.validate_env()` runs on import and will raise if required env vars are missing. For quick development you can set `SKIP_ENV_VALIDATION=true`.

Run the API

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Endpoints
- POST /ingest — upload a PDF. Form fields:
  - `file` (multipart file, PDF)
  - `chunk_size` (int, default 800)
  - `overlap` (int, default 100)
  - `rebuild` (bool, default false) — if true, re-index even when file exists

Example:

```bash
curl -X POST "http://localhost:8000/ingest" \
  -F "file=@/path/to/doc.pdf;type=application/pdf" \
  -F "chunk_size=800" \
  -F "overlap=100"
```

- POST /query — JSON body `{ "question": "...", "k": 3, "filename": "optional.pdf" }`

Example:

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question":"What is the return policy?","k":3}'
```

- GET /files — list files known in S3 and the vectorstore
- GET /health — simple health check
- DELETE /files/{filename} — delete file from S3 and vectorstore

Project layout (important files)
- `main.py` — FastAPI app and endpoints
- `pdf_utils.py` — PDF loading, chunking and hashing
- `vectorstore.py` — wrapper to interact with Pinecone (embeddings + metadata)
- `s3_utils.py` — simple S3 upload/download/delete helpers
- `rag.py` — retrieval + LLM chain wiring
- `retrieval.py` — builds the retriever used for queries
- `config.py` — environment configuration and validation

Notes & next steps
- This repository expects Pinecone for vector storage and AWS S3 for file storage; those can be swapped by changing the helper modules.
- Consider adding authentication, rate-limiting and request size limits before exposing publicly.
- Add tests for ingestion, retrieval, and error conditions.

License

This project currently has no license file. Add a `LICENSE` if you plan to open-source the code.

If you want this file saved as `README.md` instead, tell me and I will add/rename it.
