import logging

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from src.api.services.ingest import ingest_pdf_file


logger = logging.getLogger(__name__)

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

    try:
        result = ingest_pdf_file(file)

        return IngestionResponse(**result)

    except Exception as exc:
        logger.exception("PDF ingestion failed.")

        raise HTTPException(status_code=500, detail="Failed to ingest PDF") from exc
