from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from typing import Optional
from pydantic import BaseModel
from pdf_utils import load_pdf, chunk_documents, get_file_hash
from vectorstore import (
    add_documents,
    delete_file as delete_from_vectorstore,
    list_indexed_files,
    file_exists_in_index,
)
from s3_utils import (
    upload_file,
    download_file,
    delete_file as delete_from_s3,
    file_exists,
    list_files as list_s3_files,
)
from logger import logger
from rag import ask as rag_ask

app = FastAPI(
    title="PDF RAG API", description="LangChain-powered PDF RAG system with S3 storage"
)


class QueryRequest(BaseModel):
    question: str
    k: Optional[int] = 3
    filename: Optional[str] = None


@app.post("/ingest")
async def ingest(
    file: UploadFile = File(...),
    chunk_size: int = Form(800),
    overlap: int = Form(100),
    rebuild: bool = Form(False),
):

    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    filename = file.filename or "uploaded.pdf"

    if not rebuild and file_exists(filename):
        # file already present in S3; still check vectorstore by hash after reading
        logger.info("File %s already exists in S3", filename)

    file_data = await file.read()
    file_hash = get_file_hash(file_data)

    if not rebuild and file_exists_in_index(filename, file_hash):
        raise HTTPException(
            status_code=409,
            detail=f"File '{filename}' already indexed. Use rebuild=true to re-index.",
        )

    if rebuild:
        # delete by specific file_id
        delete_from_vectorstore(filename, file_hash)

    upload_file(file_data, filename)

    documents = load_pdf(file_data)
    chunks = chunk_documents(documents, chunk_size=chunk_size, overlap=overlap)

    success = add_documents(chunks, filename, file_hash)

    if not success:
        raise HTTPException(
            status_code=500, detail="Failed to add documents to vector store"
        )

    return {
        "message": "File indexed successfully",
        "filename": filename,
        "pages": len(documents),
        "chunks": len(chunks),
    }


@app.post("/query")
async def query(request: QueryRequest):
    try:
        result = rag_ask(
            question=request.question,
            k=request.k or 3,
            filename=request.filename,
        )
        return result
    except Exception as e:
        logger.exception("Error during query: %s", e)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/files")
async def list_files():
    s3_files = list_s3_files()
    pinecone_files = list_indexed_files()

    all_files = list(set(s3_files + pinecone_files))

    return {"files": all_files}


@app.get("/health")
async def health():
    return JSONResponse({"status": "ok"})


@app.delete("/files/{filename}")
async def delete_file(filename: str):
    from config import INDEX_NAME

    # Attempt to delete from S3 and vectorstore. If we don't know the file hash,
    # delete by filename metadata in the vectorstore.
    try:
        s3_deleted = delete_from_s3(filename)
    except Exception:
        logger.exception("Error deleting file from S3: %s", filename)
        s3_deleted = False

    try:
        vector_deleted = delete_from_vectorstore(filename)
    except Exception:
        logger.exception("Error deleting file from vectorstore: %s", filename)
        vector_deleted = False

    if not s3_deleted and not vector_deleted:
        raise HTTPException(status_code=404, detail="File not found")

    return {"message": f"File '{filename}' deleted from S3 and Pinecone"}
