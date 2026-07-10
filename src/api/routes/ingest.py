import logging
from pathlib import Path
from shutil import copyfileobj
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from src.ingestion.loader import load_and_chunk_pdf
from src.ingestion.vectorstore import store_chunks
from src.retrieval.keyword import reset_bm25_index


logger = logging.getLogger(__name__)

DOCUMENTS_DIR = Path("data/documents")

router = APIRouter(
    prefix="/api/v1",
    tags=["Ingestion"],
)


class IngestionResponse(BaseModel):
    """
    Response model for document ingestion
    """

    status: str
    filename: str
    chunks_created: int
    message: str


@router.post("/ingest", response_model=IngestionResponse)
def ingest_document(file: UploadFile = File(...)) -> IngestionResponse:
    """
    Upload and ingest a PDF document
    """

    filename = file.filename or ""

    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported.",
        )

    logger.info("Received PDF ingestion request.")

    safe_filename = Path(filename).name
    stored_filename = f"{uuid4().hex}_{safe_filename}"
    stored_path = DOCUMENTS_DIR / stored_filename

    try:
        DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

        with stored_path.open("wb") as buffer:
            copyfileobj(file.file, buffer)

        logger.info("Uploaded PDF saved successfully: %s", stored_filename)

        chunks = load_and_chunk_pdf(str(stored_path))
        store_chunks(chunks)
        reset_bm25_index()

        return IngestionResponse(
            status="success",
            filename=stored_filename,
            chunks_created=len(chunks),
            message="PDF uploaded and ingested successfully.",
        )

    except Exception as exc:
        logger.exception("PDF ingestion failed.")

        if stored_path.exists():
            stored_path.unlink()

        raise HTTPException(status_code=500, detail="Failed to ingest PDF") from exc
