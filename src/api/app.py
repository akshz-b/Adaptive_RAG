import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes.query import router as query_router
from src.api.routes.ingest import router as ingest_router
from src.api.routes.documents import router as document_router
from src.api.routes.health import router as health_router

from src.utils.logging_config import setup_logging
from src.api.middleware.request_logging import LoggingMiddleware
from src.api.middleware.error_handler import ErrorHandlingMiddleware
from src.config import settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Manage FastAPI application startup and shutdown.
    """

    logger.info("Starting up AdaptiveRAG API application...")

    if settings.api_enable_startup_warmup:
        logger.info("API startup warmup is enabled.")
        # warup logic can be added later
    else:
        logger.info("API startup warmup is disabled.")

    yield
    logger.info("Shutting down AdaptiveRAG API application...")


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    """
    setup_logging()

    app = FastAPI(
        title="AdaptiveRAG API",
        description="HTTP API layer for the AdaptiveRAG Agent",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(ErrorHandlingMiddleware)
    app.add_middleware(LoggingMiddleware)

    # add CORS middleware to allow requests from any origin
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api_cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health_router)
    app.include_router(query_router)
    app.include_router(ingest_router)
    app.include_router(document_router)

    return app
