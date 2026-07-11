import logging
from typing import Any


logger = logging.getLogger(__name__)


def _extract_sources(state: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Extract source objects from reranked chunks in graph state.
    """
    sources: list[dict[str, Any]] = []

    reranked_chunks = state.get("reranked_chunks", [])

    for chunk in reranked_chunks:
        metadata = chunk.get("metadata", [])

        sources.append(
            {
                "source": metadata.get("source"),
                "page_number": metadata.get("page_number"),
                "chunk_id": metadata.get("chunk_id"),
            }
        )

    return sources


def run_query(query: str, include_sources: bool = True) -> dict[str, Any]:
    """
    Run the adaptive RAG query.

    Args:
        query: The query to run.
        include_sources: Whether to include sources in the response.

    """

    from src.graph import rag_graph

    logger.info("Running AdaptiveRAG query from API service.")

    state = rag_graph.invoke(
        {
            "query": query,
            "retry_count": 0,
        },
        config={
            "tags": ["api", "adaptive-rag"],
            "metadata": {
                "app_stage": "FastAPI-layer",
            },
        },
    )

    answer = state.get("answer") or "I could not find answer for this query."
    route = state.get("route", "")

    sources: list[dict[str, Any]] = []

    if include_sources:
        sources = _extract_sources(state)

    logger.info("AdaptiveRAG API query completed. Route: %s", route)

    return {
        "answer": answer,
        "route": route,
        "sources": sources,
    }
