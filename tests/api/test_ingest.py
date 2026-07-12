from io import BytesIO
from unittest.mock import patch

from fastapi.testclient import TestClient

from src.api.app import create_app


def test_ingest_endpoint_uploads_pdf_successfully() -> None:
    """
    Test that ingestion endpoint accepts a PDF upload.
    """

    app = create_app()
    client = TestClient(app)

    mock_result = {
        "status": "success",
        "filename": "test.pdf",
        "chunks_created": 3,
        "message": "PDF uploaded and ingested successfully.",
    }

    with patch("src.api.routes.ingest.ingest_pdf_file", return_value=mock_result):
        response = client.post(
            "/api/v1/ingest",
            files={
                "file": (
                    "test.pdf",
                    BytesIO(b"%PDF-1.4 mock pdf content"),
                    "application/pdf",
                )
            },
        )

    assert response.status_code == 200
    assert response.json() == mock_result


def test_ingest_endpoint_rejects_non_pdf() -> None:
    """
    Test that ingestion endpoint rejects non-PDF files.
    """

    app = create_app()
    client = TestClient(app)

    response = client.post(
        "/api/v1/ingest",
        files={
            "file": (
                "test.txt",
                BytesIO(b"This is not a pdf"),
                "text/plain",
            )
        },
    )

    assert response.status_code == 400
    assert response.json() == {
        "detail": "Only PDF files are supported.",
    }


def test_ingest_endpoint_service_failure() -> None:
    """
    Test that ingestion endpoint handles service failures gracefully.
    """

    app = create_app()
    client = TestClient(app)

    with patch(
        "src.api.routes.ingest.ingest_pdf_file",
        side_effect=RuntimeError("Ingestion failed"),
    ):
        response = client.post(
            "/api/v1/ingest",
            files={
                "file": (
                    "test.pdf",
                    BytesIO(b"%PDF-1.4 mock pdf content"),
                    "application/pdf",
                )
            },
        )

    assert response.status_code == 500
    assert response.json() == {
        "detail": "Failed to ingest PDF",
    }
