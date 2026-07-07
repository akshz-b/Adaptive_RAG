from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import status as http_status

from src.api.schemas import HealthResponse, QueryRequest, QueryResponse
from src.graph import rag_graph

app = FastAPI(
    title="AdaptiveRAG API",
    version="0.1.0",
    description="FastAPI layer for the AdaptiveRAG agent.",
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return a simple health payload for the API service."""
    return HealthResponse(
        status="ok",
        service="adaptive-rag",
        version="0.1.0",
    )


@app.get("/")
def root() -> dict[str, str]:
    """Root endpoint for a quick service check."""
    return {"message": "AdaptiveRAG API is running"}


@app.post("/query", response_model=QueryResponse)
def run_query(request: QueryRequest) -> QueryResponse:
    """Accept a user query and return the agent response."""
    if not request.query.strip():
        raise HTTPException(
            status_code=http_status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Query must not be empty.",
        )

    try:
        state = rag_graph.invoke(
            {
                "query": request.query,
                "retry_count": 0,
            },
            config={
                "tags": ["api", "adaptive-rag"],
                "metadata": {
                    "endpoint": "query",
                    "debug": request.debug,
                },
            },
        )

        response = QueryResponse(
            status="success",
            answer=state.get("answer", ""),
            route=state.get("route", "retrieval"),
            citations=state.get("citations", []),
            final_context=state.get("final_context") if request.debug else None,
            retry_count=state.get("retry_count", 0),
        )

        return response

    except Exception as exc:
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )
