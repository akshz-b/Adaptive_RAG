from fastapi.testclient import TestClient

from src.api.app import create_app


def test_health_endpoint_returns_ok() -> None:
    """
    Test that the health endpoint return API status.
    """

    app = create_app()
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "Adaptive RAG",
    }


def test_readiness_endpoint_returns_ok() -> None:
    """
    Test that the readiness endpoint return API status.
    """

    app = create_app()
    client = TestClient(app)
    response = client.get("/ready")
    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ready"
    assert data["checks"]["config_loaded"] is True
    assert data["checks"]["chroma_path_configured"] is True
    assert data["checks"]["chroma_path_accessible"] is True
