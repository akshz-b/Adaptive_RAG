import logging
from pydantic import BaseModel, Field
from typing import List, Optional

from fastapi import APIRouter, HTTPException

from src.api.services.query import run_query

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
        result = run_query(
            query=query,
            include_sources=request.include_sources,
        )

        return QueryResponse(**result)

    except Exception as exc:
        logger.exception("API query request failed.")

        raise HTTPException(
            status_code=500,
            detail="Failed to process query.",
        ) from exc
