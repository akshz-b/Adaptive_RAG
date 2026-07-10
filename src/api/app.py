import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.routes.query import router as query_router
from src.api.routes.ingest import router as ingest_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Manage FastAPI application startup and shutdown.
    """

    logger.info("Starting up AdaptiveRAG API application...")
    yield
    logger.info("Shutting down AdaptiveRAG API application...")


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    """

    app = FastAPI(
        title="AdaptiveRAG API",
        description="HTTP API layer for the AdaptiveRAG Agent",
        version="0.1.0",
        lifespan=lifespan,
    )

    # add CORS middleware to allow requests from any origin
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",
            "http://localhost:5173",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health", tags=["Health"])
    async def health_check():
        """
        Health check endpoint to verify that the API is running.
        """
        return {
            "status": "ok",
            "service": "AdaptiveRAG API",
        }

    app.include_router(query_router)
    app.include_router(ingest_router)

    return app
