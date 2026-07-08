import logging
from pydantic import BaseModel, Field
from typing import List, Optional

from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1",
    tags=["Query"],
)


class Source(BaseModel):
    """
    Source metadata returned with an answer
    """

    source: Optional[str] = None
    page_number: Optional[int] = None
    chunk_id: Optional[str] = None


class QueryRequest(BaseModel):
    """
    Request model for querying AdaptiveRAG
    """

    query: str = Field(..., min_length=1)
    include_sources: bool = True


class QueryResponse(BaseModel):
    """
    Response model for AdaptiveRAG query results.
    """

    answer: str
    sources: List[Source] = Field(default_factory=list)
    route: Optional[str] = None


def _extract_sources(state: dict) -> List[Source]:
    """
    Extract source objects from reranked_chunks in graph state.
    """
    sources: List[Source] = []

    reranked_chunks = state.get("reranked_chunks", [])

    # for each chunk get metadata and build source object from metadata
    for chunk in reranked_chunks:
        metadata = chunk.get("metadata", {})

        source = Source(
            source=metadata.get("source"),
            page_number=metadata.get("page_number"),
            chunk_id=metadata.get("chunk_id"),
        )
        sources.append(source)

    return sources


@router.post("/query", response_model=QueryResponse)
def query_endpoint(request: QueryRequest) -> QueryResponse:
    """
    Handle query requests and return answers with sources
    POST  /api/v1/query
    """

    query = request.query.strip()

    if not query:
        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty.",
        )

    try:
        from src.graph import rag_graph

        logger.info("Received API query request")

        state = rag_graph.invoke(
            {
                "query": query,
                "retry_count": 0,
            },
            config={
                "tags": ["api", "adaptive-rag"],
                "metadata": {
                    "app-stage": "FastAPI-layer",
                },
            },
        )

        answer = state.get("answer", "")
        route = state.get("route", "")

        sources: List[Source] = None
        if request.include_sources:
            sources = _extract_sources(state)

        logger.info("API query completed. RouteL %s", route)

        return QueryResponse(
            answer=answer,
            sources=sources,
            route=route,
        )
    except HTTPException:
        raise

    except Exception as exc:
        logger.exception("API query request failed.")

        raise HTTPException(
            status_code=500,
            detail="Failed to process query.",
        ) from exc
