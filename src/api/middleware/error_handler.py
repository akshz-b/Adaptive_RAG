import logging
from typing import Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from src.api.schemas.errors import ErrorResponse

logger = logging.getLogger(__name__)


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """
    Middleware that converts unexpected API errors into standardized responses.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Handle unexpected errors raised while processing API requests.
        """

        try:
            return await call_next(request)

        except Exception:
            logger.exception("Unhandled API error")

            error_response = ErrorResponse(
                detail="Internal server error.", error_code="internal_server_error"
            )

            return JSONResponse(
                status_code=500,
                content=error_response.model_dump(),
            )
