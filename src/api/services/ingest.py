import logging
from pathlib import Path
from shutil import copyfileobj
from uuid import uuid4

from fastapi import UploadFile

from src.ingestion.loader import load_and_chunk_pdf
from src.ingestion.vectorstore import store_chunks
from src.retrieval.keyword import reset_bm25_index

logger = logging.getLogger(__name__)

DOCUMENTS_DIR = Path("data/documents")


def ingest_pdf_file(file: UploadFile) -> dict:
    """
    Save, process, and store an uploaded PDF file.
    """

    filename = file.filename or ""
    safe_filename = Path(filename).name

    stored_filename = f"{uuid4().hex}_{safe_filename}"
    stored_path = DOCUMENTS_DIR / stored_filename

    try:
        DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

        with stored_path.open("wb") as f:
            copyfileobj(file.file, f)

        logger.info("Uploaded PDF saved successfully: %s", stored_filename)

        chunks = load_and_chunk_pdf(str(stored_path))
        store_chunks(chunks)
        reset_bm25_index()

        return {
            "status": "success",
            "filename": stored_filename,
            "chunks_created": len(chunks),
            "message": "PDF uploaded and ingested successfully.",
        }

    except Exception:
        logger.exception("PDF ingestion failed")

        if stored_path.exists():
            try:
                stored_path.unlink()
            except Exception:
                logger.exception("Failed to delete")

        raise
