import logging
from pathlib import Path


def setup_logging(log_filename: str = "adaptive_rag.log") -> None:
    """
    Configure central logging for both CLI and API to output to the
    terminal and write to logs/adaptive_rag.log using a consistent format.
    """
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_dir / log_filename, encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,  # Override uvicorn/other default logging configs
    )
