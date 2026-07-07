from typing import List, Optional

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Response model for the health endpoint."""

    status: str
    service: str
    version: str


class QueryRequest(BaseModel):
    """Request model for the query endpoint."""

    query: str
    debug: bool = False


class QueryResponse(BaseModel):
    """Response model for the query endpoint."""

    status: str
    answer: str
    route: Optional[str] = None
    citations: List[str] = []
    final_context: Optional[str] = None
    retry_count: int = 0
    error: Optional[str] = None
