import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from src.api.app import create_app

app = create_app()
client = TestClient(app)


@pytest.fixture(autouse=True)
def cleanup_test_files():
    """Fixture to clean up test files before and after each test."""
    test_files = [
        Path("data/documents/test.pdf"),
        Path("data/documents/test_fail.pdf"),
    ]
    for p in test_files:
        if p.exists():
            p.unlink()
    yield
    for p in test_files:
        if p.exists():
            p.unlink()


@patch("src.api.routes.ingest.load_and_chunk_pdf")
@patch("src.api.routes.ingest.store_chunks")
@patch("src.api.routes.ingest.reset_bm25_index")
def test_ingest_pdf_success(mock_reset, mock_store, mock_load, cleanup_test_files):
    # Configure load_and_chunk_pdf mock to return 3 mock chunks
    mock_load.return_value = [MagicMock(), MagicMock(), MagicMock()]
    
    file_content = b"%PDF-1.4 mock pdf content"
    files = {"file": ("test.pdf", file_content, "application/pdf")}
    
    response = client.post("/api/v1/ingest", files=files)
    
    assert response.status_code == 200
    response_data = response.json()
    assert response_data["status"] == "success"
    assert response_data["chunks_created"] == 3
    
    # Assert destination file was created during the process
    # Note: Since the endpoint completes successfully, the file remains in data/documents/
    dest_path = Path("data/documents/test.pdf")
    assert dest_path.exists()
    
    # Verify mock interactions
    mock_load.assert_called_once_with(str(dest_path))
    mock_store.assert_called_once_with(mock_load.return_value)
    mock_reset.assert_called_once()


def test_ingest_pdf_invalid_extension(cleanup_test_files):
    file_content = b"some plain text content"
    files = {"file": ("test.txt", file_content, "text/plain")}
    
    response = client.post("/api/v1/ingest", files=files)
    
    assert response.status_code == 400
    assert "Only PDF files are supported" in response.json()["detail"]
    
    # Assert no file was written to data/documents/
    dest_path = Path("data/documents/test.txt")
    assert not dest_path.exists()


@patch("src.api.routes.ingest.load_and_chunk_pdf")
@patch("src.api.routes.ingest.store_chunks")
@patch("src.api.routes.ingest.reset_bm25_index")
def test_ingest_pdf_failure_cleanup(mock_reset, mock_store, mock_load, cleanup_test_files):
    # Configure mock to raise an error during chunking
    mock_load.side_effect = Exception("Parsing error")
    
    file_content = b"%PDF-1.4 mock pdf content"
    files = {"file": ("test_fail.pdf", file_content, "application/pdf")}
    
    dest_path = Path("data/documents/test_fail.pdf")
    
    response = client.post("/api/v1/ingest", files=files)
    
    assert response.status_code == 500
    assert "PDF Ingestion failed" in response.json()["detail"]
    
    # Assert that the uploaded file was cleaned up on failure
    assert not dest_path.exists()
    
    # Verify store_chunks and reset_bm25_index were not called
    mock_store.assert_not_called()
    mock_reset.assert_not_called()
