from unittest.mock import patch

from fastapi.testclient import TestClient

from src.api.app import create_app


def test_query_endpoint_returns_answer_with_sources() -> None:
    """
    Tets that query endpoint return answer, route, and sources.
    """

    app = create_app()
    client = TestClient(app)

    mock_result = {
        "answer": "Mock answer",
        "route": "retrieval",
        "sources": [{"source": "doc_1.pdf", "page_number": 1, "chunk_id": "chunk_1"}],
    }

    with patch("src.api.routes.query.run_query", return_value=mock_result):
        response = client.post(
            "/api/v1/query",
            json={
                "query": "What is the capital of France?",
                "include_sources": True,
            },
        )
    assert response.status_code == 200

    data = response.json()

    assert data == {
        "answer": "Mock answer",
        "route": "retrieval",
        "sources": [{"source": "doc_1.pdf", "page_number": 1, "chunk_id": "chunk_1"}],
    }


def test_query_endpoint_rejects_empty_query_validation() -> None:
    """
    Tets that query endpoint rejects empty query.
    """

    app = create_app()
    client = TestClient(app)

    response = client.post(
        "/api/v1/query",
        json={
            "query": "",
            "include_sources": True,
        },
    )

    assert response.status_code == 422


def test_query_endpoint_rejects_whitespace_query() -> None:
    app = create_app()
    client = TestClient(app)

    response = client.post(
        "/api/v1/query",
        json={
            "query": "      ",
            "include_sources": True,
        },
    )

    assert response.status_code == 400
    assert response.json() == {
        "detail": "Query cannot be empty.",
    }


def test_query_endpoint_returns_500_when_service_falls() -> None:
    """
    Test that query endpoint returns 500 when query service fails.
    """

    app = create_app()
    client = TestClient(app)

    with patch(
        "src.api.routes.query.run_query",
        side_effect=RuntimeError("Service failed"),
    ):
        response = client.post(
            "/api/v1/query",
            json={
                "query": "What is this document about?",
                "include_sources": True,
            },
        )

    assert response.status_code == 500
    assert response.json() == {"detail": "Failed to process query."}
