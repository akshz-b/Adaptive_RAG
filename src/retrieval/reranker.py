import logging
from typing import List, Optional

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from src.config import settings

logger = logging.getLogger(__name__)

_cross_encoder: Optional[CrossEncoder] = None


def _get_cross_encoder() -> CrossEncoder:
    """
    Lazily load and return the cross-encoder reranker model.
    """
    global _cross_encoder

    if _cross_encoder is None:
        logger.info("Loading reranker model: %s", settings.reranker_model)
        _cross_encoder = CrossEncoder(settings.reranker_model)
        logger.info("Reranker model loaded successfully.")

    return _cross_encoder


def rerank(
    query: str,
    chunks: List[Document],
    top_n: Optional[int] = None,
) -> List[Document]:
    """
    Rerank candidate chunks using a cross-encoder model.

    Falls back to original retrieved chunks if reranking fails.

    Args:
        query: User query string.
        chunks: Candidate chunks from hybrid search.
        top_n: Number of reranked chunks to return.

    Returns:
        List[Document]: Top reranked chunks, or original top chunks on failure.
    """
    if not chunks:
        logger.info("Reranking skipped because no chunks were provided.")
        return []

    final_top_n = top_n if top_n is not None else settings.top_k_rerank

    try:
        logger.info(
            "Starting reranking for %s chunks. Returning top %s.",
            len(chunks),
            final_top_n,
        )

        model = _get_cross_encoder()

        pairs = [(query, doc.page_content) for doc in chunks]
        scores = model.predict(pairs)

        scored = list(zip(chunks, scores))
        scored.sort(key=lambda item: item[1], reverse=True)

        results: List[Document] = []

        for doc, score in scored[:final_top_n]:
            metadata = dict(doc.metadata)
            metadata["reranker_score"] = float(score)

            results.append(
                Document(
                    page_content=doc.page_content,
                    metadata=metadata,
                )
            )

        logger.info("Reranking completed. Returned %s chunks.", len(results))
        return results

    except Exception:
        logger.exception("Reranking failed. Falling back to original retrieved chunks.")

        fallback_chunks = chunks[:final_top_n]
        fallback_results: List[Document] = []

        for doc in fallback_chunks:
            metadata = dict(doc.metadata)
            metadata["reranker_fallback"] = True

            fallback_results.append(
                Document(
                    page_content=doc.page_content,
                    metadata=metadata,
                )
            )

        return fallback_results
