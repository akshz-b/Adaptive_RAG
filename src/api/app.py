import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """"
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

    #add CORS middleware to allow requests from any origin
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
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
    
    return app