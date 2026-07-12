import logging
from typing import Any
from pathlib import Path
from src.config import settings


logger = logging.getLogger(__name__)


def check_readiness() -> dict[str, Any]:
    """
    Check lightweight API dependency readiness.
    """

    chroma_path = Path(settings.chroma_persist_dir)

    checks: dict[str, Any] = {
        "config_loaded": True,
        "chroma_path_configured": bool(settings.chroma_persist_dir),
        "chroma_path_accessible": False,
    }

    try:
        chroma_path.mkdir(parents=True, exist_ok=True)
        checks["chroma_path_accessible"] = chroma_path.exists()

    except Exception:
        logger.exception("Readiness check failed for ChromaDB.")
        checks["chroma_path_accessible"] = False

    is_ready = all(checks.values())

    return {
        "status": "ready" if is_ready else "not_ready",
        "checks": checks,
    }
