from src.retrieval.keyword import reset_bm25_index
from src.api.services.ingest import DOCUMENTS_DIR
import logging
import re
from collections import Counter
from pathlib import Path

from src.ingestion.vectorstore import get_vectorstore


logger = logging.getLogger(__name__)
_SAFE_DOCUMENT_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def _validate_document_id(document_id: str) -> str:
    if not document_id or document_id in {".", ".."}:
        raise ValueError("Invalid document ID.")
    if not _SAFE_DOCUMENT_ID_RE.fullmatch(document_id):
        raise ValueError("Invalid document ID.")
    return document_id


def list_ingested_documents() -> list:
    """
    List ingested documents grouped by source metadata
    """

    vectorstore = get_vectorstore()
    stored_data = vectorstore.get(include=["metadatas"])

    metadatas = stored_data.get("metadatas") or []

    source_counts: Counter[str] = Counter()

    for metadata in metadatas:
        if not metadata:
            continue

        source = metadata.get("source")

        if source:
            source_counts[source] += 1

    documents = []

    for source, chunk_count in source_counts.items():
        documents.append(
            {
                "document_id": source,
                "filename": source,
                "chunk_count": chunk_count,
            }
        )

    return documents


def delete_ingested_document(document_id: str) -> dict:
    """
    Delete an ingested document and its chunks.
    """
    safe_document_id = _validate_document_id(document_id)

    # Validate and resolve path to prevent path traversal vulnerability (CWE-22)
    resolved_dir = DOCUMENTS_DIR.resolve()
    document_path = (resolved_dir / Path(safe_document_id)).resolve()

    if not document_path.is_relative_to(resolved_dir) or document_path == resolved_dir:
        raise ValueError("Invalid document ID.")

    # 1. Find Chroma chunk IDs where the metadata["source"] == document_id

    vectorstore = get_vectorstore()

    stored_data = vectorstore.get(
        where={"source": safe_document_id},
        include=["metadatas"],
    )

    chunk_ids = stored_data.get("ids", [])

    if not chunk_ids:
        raise FileNotFoundError(f"Document not found: {safe_document_id}")

    vectorstore.delete(ids=chunk_ids)

    if document_path.exists():
        document_path.unlink()
    else:
        logger.warning("Document file not found on disk: %s", document_path)

    reset_bm25_index()

    return {
        "status": "success",
        "document_id": safe_document_id,
        "chunks_deleted": len(chunk_ids),
        "message": "Document deleted successfully.",
    }
