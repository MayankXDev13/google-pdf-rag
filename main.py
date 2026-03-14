import logging
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from fastapi import FastAPI, UploadFile, File, HTTPException
from utils.s3_utils import (
    upload_file,
    delete_from_s3,
    file_exists,
    download_file,
    list_files,
)
from utils.pdf_utils import load_pdf, chunk_documents
from utils.pinecone_utils import add_documents, delete_by_source, list_indexed_files
from pydantic import BaseModel
from chat import chat

app = FastAPI()


@app.post("/ingest")
async def ingest(
    file: UploadFile = File(...),
    chunk_size: int = 800,
    overlap: int = 100,
    rebuild: bool = False,
):
    try:

        if file.content_type != "application/pdf":
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")

        file_name = file.filename
        if not file_name:
            raise HTTPException(status_code=400, detail="Filename cannot be empty")

        logger.info(f"Processing ingest request for file: {file_name}")

        if file_exists(file_name) and not rebuild:
            raise HTTPException(
                status_code=409,
                detail="File already exists. Use rebuild=true to re-index.",
            )

        file_data = await file.read()

        upload_file(file_data, file_name)
        logger.info(f"Uploaded file to S3: {file_name}")

        downloaded_file_data = download_file(file_name)

        documents = load_pdf(downloaded_file_data)
        logger.info(f"Loaded PDF, got {len(documents)} pages")

        chunks = chunk_documents(documents, file_name, chunk_size, overlap)
        logger.info(f"Created {len(chunks)} chunks")

        add_documents(chunks)
        logger.info(f"Indexed {len(chunks)} chunks to Pinecone")

        return {
            "message": "File successfully ingested",
            "file_name": file_name,
            "chunks_created": len(chunks),
        }

    except HTTPException as e:
        raise e

    except Exception as e:
        logger.error(f"Error during ingest: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


class QueryRequest(BaseModel):
    question: str  # User query
    k: Optional[int] = 3  # Number of results to retrieve from vector DB
    filename: Optional[str] = (
        None  # Optional filter to search inside a specific document
    )


@app.post("/query")
def query(request: QueryRequest):

    if not request.question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        logger.info(f"Processing query: {request.question[:50]}...")

        result = chat(
            question=request.question, k=request.k or 3, filename=request.filename
        )

        return {
            "question": request.question,
            "answer": result["answer"],
            "sources": result["sources"],
        }

    except Exception as e:
        logger.error(f"Error during query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.get("/files")
def list_all_files():
    """List all files in S3 and Pinecone"""
    try:
        s3_files = list_files()
        indexed_files = list_indexed_files()

        return {
            "s3_files": s3_files,
            "indexed_files": indexed_files,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list files: {str(e)}")


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


@app.delete("/files/{filename}")
def delete_file(filename: str):
    """Delete a file from both S3 and Pinecone"""
    try:
        s3_deleted = delete_from_s3(filename)

        try:
            delete_by_source(filename)
            pinecone_deleted = True
        except Exception as e:
            pinecone_deleted = False
            print(f"Warning: Failed to delete from Pinecone: {e}")

        if not s3_deleted and not pinecone_deleted:
            raise HTTPException(status_code=404, detail="File not found")

        return {
            "message": "File deleted successfully",
            "s3_deleted": s3_deleted,
            "pinecone_deleted": pinecone_deleted,
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete file: {str(e)}")
