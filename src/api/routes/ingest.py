import logging
from pathlib import Path
from pydantic import BaseModel
from fastapi import APIRouter, File, UploadFile, HTTPException, status
import anyio

from src.ingestion.loader import load_and_chunk_pdf
from src.ingestion.vectorstore import store_chunks
from src.retrieval.keyword import reset_bm25_index

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1",
    tags=["Ingestion"],
)

class IngestionResponse(BaseModel):
    """
    Response model for document ingestion status.
    """
    status: str
    chunks_created: int


@router.post("/ingest", response_model=IngestionResponse, status_code=status.HTTP_200_OK)
async def ingest_pdf(file: UploadFile = File(...)) -> IngestionResponse:
    """
    Upload a PDF document, chunk it, and store it in ChromaDB.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        logger.error("Rejected upload: File is not a PDF")
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported.",
        )

    documents_dir = Path("data/documents")
    documents_dir.mkdir(parents=True, exist_ok=True)

    safe_filename = Path(file.filename).name
    dest_path = documents_dir / safe_filename

    try:
        logger.info("Saving uploaded file to %s", dest_path)
        # Async file saving using anyio
        async with await anyio.open_file(dest_path, "wb") as f:
            while chunk := await file.read(1024 * 1024):
                await f.write(chunk)

        logger.info("Processing ingestion for: %s", dest_path)
        # Load and chunk PDF
        chunks = load_and_chunk_pdf(str(dest_path))
        
        # Store chunks in ChromaDB
        store_chunks(chunks)
        
        # Reset BM25 index cache since new documents have been added
        reset_bm25_index()

        logger.info("Ingestion completed successfully for %s. Created %d chunks.", safe_filename, len(chunks))
        return IngestionResponse(
            status="success",
            chunks_created=len(chunks),
        )

    except HTTPException:
        # Re-raise HTTPExceptions (e.g. 400 from validation)
        # Make sure to clean up the saved file if it was created
        if dest_path.exists():
            dest_path.unlink()
        raise

    except Exception as exc:
        # Clean up file on failure
        if dest_path.exists():
            try:
                dest_path.unlink()
                logger.info("Cleaned up file after failure: %s", dest_path)
            except Exception as cleanup_exc:
                logger.error("Failed to clean up file %s: %s", dest_path, cleanup_exc)
        
        logger.exception("PDF Ingestion failed for file %s", safe_filename)
        raise HTTPException(
            status_code=500,
            detail=f"PDF Ingestion failed: {str(exc)}",
        ) from exc

    finally:
        await file.close()
