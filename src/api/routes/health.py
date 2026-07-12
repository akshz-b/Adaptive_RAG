from typing import Any

from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

from src.api.services.readiness import check_readiness

router = APIRouter(
    tags=["Health"],
)


@router.get("/health")
def health_check() -> dict[str, str]:
    """
    Return lightweight API health status.
    """

    return {
        "status": "ok",
        "service": "Adaptive RAG",
    }


@router.get("/ready")
async def readiness_check():
    """
    Return API readiness status.
    """
    result: dict[str, Any] = check_readiness()

    status_code = (
        status.HTTP_200_OK
        if result["status"] == "ready"
        else status.HTTP_503_SERVICE_UNAVAILABLE
    )

    return JSONResponse(
        status_code=status_code,
        content=result,
    )
