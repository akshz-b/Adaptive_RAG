import logging
from typing import List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.api.services.document import list_ingested_documents, delete_ingested_document


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/documents",
    tags=["Documents"],
)


class DocumentInfo(BaseModel):
    """
    Metadata for one ingested document.
    """

    document_id: str
    filename: str
    chunk_count: int


class DocumentListResponse(BaseModel):
    """
    Response model for listing ingested documents.
    """

    documents: List[DocumentInfo] = Field(default_factory=list)


class DeleteResponse(BaseModel):
    """
    Response model for successful deletion.
    """

    status: str
    document_id: str
    chunks_deleted: int
    message: str


@router.get("/documents", response_model=DocumentListResponse)
def list_documents() -> DocumentListResponse:
    """
    List all documents ingested into the system.
    """

    try:
        documents = list_ingested_documents()

        return DocumentListResponse(
            documents=[DocumentInfo(**document) for document in documents]
        )
    except Exception as exc:
        logger.exception("Failed to list documents")

        raise HTTPException(status_code=500, detail="Failed to list documents") from exc


@router.delete("/documents/{document_id}", response_model=DeleteResponse)
def delete_document(document_id: str) -> DeleteResponse:
    """
    Delete an ingested document and all its chunks.
    """
    try:
        result = delete_ingested_document(document_id)
        return DeleteResponse(**result)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Document not found") from exc
    except Exception as exc:
        logger.exception("Failed to delete document")
        raise HTTPException(
            status_code=500, detail="Failed to delete document"
        ) from exc
