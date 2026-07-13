from unittest.mock import patch

from fastapi.testclient import TestClient

from src.api.app import create_app


def test_list_document_returns_documents() -> None:
    """
    Test that list documents endpoint returns all ingested documents.
    """
    app = create_app()
    client = TestClient(app)

    mock_documents = [
        {
            "document_id": "doc1.pdf",
            "filename": "doc1.pdf",
            "chunk_count": 3,
        }
    ]

    with patch(
        "src.api.routes.documents.list_ingested_documents",
        return_value=mock_documents,
    ):
        response = client.get("/api/v1/documents")

    assert response.status_code == 200
    assert response.json() == {
        "documents": mock_documents,
    }


def test_list_document_empty() -> None:
    """
    Test that list documents endpoint returns an empty list when there are no ingested documents.
    """
    app = create_app()
    client = TestClient(app)

    with patch(
        "src.api.routes.documents.list_ingested_documents",
        return_value=[],
    ):
        response = client.get("/api/v1/documents")

    assert response.status_code == 200
    assert response.json() == {
        "documents": [],
    }


def test_list_document_service_failure() -> None:
    """
    Test that list documents endpoint handles service failures gracefully.
    """
    app = create_app()
    client = TestClient(app)

    with patch(
        "src.api.routes.documents.list_ingested_documents",
        side_effect=RuntimeError("Failed to list documents"),
    ):
        response = client.get("/api/v1/documents")

    assert response.status_code == 500
    assert response.json() == {
        "detail": "Failed to list documents",
    }


def test_delete_document_success() -> None:
    """
    Test that delete document endpoint successfully deletes an ingested document.
    """
    app = create_app()
    client = TestClient(app)

    mock_result = {
        "status": "success",
        "document_id": "doc1.pdf",
        "chunks_deleted": 3,
        "message": "Document deleted successfully.",
    }

    with patch(
        "src.api.routes.documents.delete_ingested_document",
        return_value=mock_result,
    ):
        response = client.delete("/api/v1/documents/doc1.pdf")

    assert response.status_code == 200
    assert response.json() == mock_result


def test_delete_document_not_found() -> None:
    """
    Test that delete document endpoint returns a 404 when the document is not found.
    """
    app = create_app()
    client = TestClient(app)

    with patch(
        "src.api.routes.documents.delete_ingested_document",
        side_effect=FileNotFoundError("Document not found"),
    ):
        response = client.delete("/api/v1/documents/missing.pdf")

    assert response.status_code == 404
    assert response.json() == {
        "detail": "Document not found.",
    }


def test_delete_document_service_failure() -> None:
    """
    Test that delete document endpoint handles service failures gracefully.
    """
    app = create_app()
    client = TestClient(app)

    with patch(
        "src.api.routes.documents.delete_ingested_document",
        side_effect=RuntimeError("Delete failed"),
    ):
        response = client.delete("/api/v1/documents/doc1.pdf")

    assert response.status_code == 500
    assert response.json() == {
        "detail": "Failed to delete document.",
    }


def test_delete_document_invalid_id() -> None:
    """
    Test that delete document endpoint returns a 400 when document_id is invalid (e.g. traversal attempt).
    """
    app = create_app()
    client = TestClient(app)

    with patch(
        "src.api.routes.documents.delete_ingested_document",
        side_effect=ValueError("Invalid document ID."),
    ):
        response = client.delete("/api/v1/documents/invalid_traversal_id")

    assert response.status_code == 400
    assert response.json() == {
        "detail": "Invalid document ID.",
    }
