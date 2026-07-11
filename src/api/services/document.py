from src.retrieval.keyword import reset_bm25_index
from src.api.routes.ingest import DOCUMENTS_DIR
import logging
from collections import Counter

from src.ingestion.vectorstore import get_vectorstore


logger = logging.getLogger(__name__)


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
    # 1. Find Chroma chunk IDs where the metadata["source"] == document_id

    vectorstore = get_vectorstore()

    stored_data = vectorstore.get(
        where={"source": document_id},
        include=["metadatas"],
    )

    chunk_ids = stored_data.get("ids", [])

    if not chunk_ids:
        raise FileNotFoundError(f"Document not found: {document_id}")

    vectorstore.delete(ids=chunk_ids)

    document_path = DOCUMENTS_DIR / document_id

    if document_path.exists():
        document_path.unlink()
    else:
        logger.warning("Document file not found on disk: %s", document_path)

    reset_bm25_index()

    return {
        "status": "success",
        "document_id": document_id,
        "chunks_deleted": len(chunk_ids),
        "message": "Document deleted successfully.",
    }
