from fastapi import FastAPI, UploadFile, File, HTTPException
from s3_utils import upload_file, delete_from_s3, file_exists, download_file
from pdf_utils import load_pdf, chunk_documents
from pinecone_utils import add_documents
from pydantic import BaseModel
from fastapi import HTTPException
from chat import chat

app = FastAPI()


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):

    try:

        # Validate file type
        if file.content_type != "application/pdf":
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")

        file_name = file.filename

        # Check if file already exists
        if file_exists(file_name):
            raise HTTPException(status_code=409, detail="File already exists")

        # Read file
        file_data = await file.read()

        # Upload to S3
        await upload_file(file_data, file_name)

        # Download from S3
        downloaded_file_data = await download_file(file_name)

        # Load PDF
        documents = await load_pdf(downloaded_file_data)

        # Chunk documents
        chunks = await chunk_documents(documents)

        # Store embeddings in Pinecone
        await add_documents(chunks)

        return {
            "message": "File successfully ingested",
            "file_name": file_name,
            "chunks_created": len(chunks),
        }

    except HTTPException as e:
        raise e

    except Exception as e:
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

        result = chat(question=request.question, k=request.k, filename=request.filename)

        return {
            "question": request.question,
            "answer": result["answer"],
            "sources": result["sources"],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")
