import logging
from typing import Dict, List, Optional

from langchain_core.documents import Document

from src.config import settings
from src.retrieval.keyword import keyword_search
from src.retrieval.semantic import semantic_retrieval

logger = logging.getLogger(__name__)

RRF_K = 60


def _add_rrf_scores(
    results: List[Document],
    scores: Dict[str, float],
    documents_by_key: Dict[str, Document],
) -> None:
    """
    Add Reciprocal Rank Fusion scores for one ranked result list.

    Args:
        results: Ranked list of retrieved documents.
        scores: Mapping from unique chunk key to accumulated RRF score.
        documents_by_key: Mapping from unique chunk key to the Document object.
    """
    for rank, document in enumerate(results):
        key = document.metadata.get("chunk_id", document.page_content)

        if key not in documents_by_key:
            documents_by_key[key] = document

        scores[key] = scores.get(key, 0.0) + (1 / (rank + RRF_K))


def hybrid_search(query: str, k: Optional[int] = None) -> List[Document]:
    """
    Retrieve chunks using semantic search and BM25 keyword search,
    then merge the results using Reciprocal Rank Fusion.

    Args:
        query: User query string.
        k: Number of final chunks to return. If not provided,
           settings.top_k_retrieval is used.

    Returns:
        List[Document]: Top-k fused and ranked documents.
    """
    if not query.strip():
        logger.warning("Hybrid search skipped because query is empty.")
        return []

    top_k = k if k is not None else settings.top_k_retrieval

    semantic_results = semantic_retrieval(query, k=top_k)
    keyword_results = keyword_search(query, k=top_k)

    logger.info(
        "Hybrid search retrieved %s semantic chunks and %s keyword chunks.",
        len(semantic_results),
        len(keyword_results),
    )

    if not semantic_results and not keyword_results:
        logger.warning("Hybrid search found no semantic or keyword results.")
        return []

    try:
        scores: Dict[str, float] = {}
        documents_by_key: Dict[str, Document] = {}

        _add_rrf_scores(
            results=semantic_results,
            scores=scores,
            documents_by_key=documents_by_key,
        )

        _add_rrf_scores(
            results=keyword_results,
            scores=scores,
            documents_by_key=documents_by_key,
        )

        ranked_keys = sorted(scores.keys(), key=lambda key: scores[key], reverse=True)

        fused_results: List[Document] = []

        for key in ranked_keys[:top_k]:
            document = documents_by_key[key]
            metadata = dict(document.metadata)
            metadata["rrf_score"] = scores[key]

            fused_results.append(
                Document(
                    page_content=document.page_content,
                    metadata=metadata,
                )
            )

        return fused_results

    except Exception:
        logger.exception(
            "Hybrid fusion failed. Falling back to available retrieval results."
        )

        fallback_results = semantic_results or keyword_results
        return fallback_results[:top_k]
