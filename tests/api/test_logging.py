import logging
from pathlib import Path
import pytest
from src.utils.logging_config import setup_logging


@pytest.fixture
def temp_log_file():
    """Fixture to ensure the temporary test log file is cleaned up."""
    log_file = Path("logs/test_log.log")
    if log_file.exists():
        log_file.unlink()
    yield log_file
    if log_file.exists():
        log_file.unlink()


def test_setup_logging(temp_log_file):
    # Call the central setup_logging utility using the test log file name
    setup_logging(log_filename="test_log.log")

    test_logger = logging.getLogger("test_logging_module")
    test_message = "Validating structured logs for the API entry point"

    # Emit a log message
    test_logger.info(test_message)

    # Assert that the log file was created
    assert temp_log_file.exists()

    # Read and inspect the file content
    with open(temp_log_file, "r", encoding="utf-8") as f:
        log_content = f.read().strip()

    # The entry should contain our message, log level, and logger name
    assert test_message in log_content
    assert "test_logging_module" in log_content
    assert "INFO" in log_content

    # The format is: timestamp - logger_name - log_level - message
    # Split from right to handle timestamp containing hyphens
    parts = log_content.split(" - ")
    assert len(parts) >= 4
    assert parts[-3] == "test_logging_module"
    assert parts[-2] == "INFO"
    assert parts[-1] == test_message
