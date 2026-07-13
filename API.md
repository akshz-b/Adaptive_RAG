# AdaptiveRAG API Documentation

This document describes the HTTP API for the AdaptiveRAG service. The API exposes endpoints for querying the RAG system, ingesting documents, and managing the document repository.

## Running the Server

Start the FastAPI application locally using `uvicorn`:

```bash
uvicorn src.api.app:create_app --factory --reload --port 8000
```

Once running, the interactive Swagger documentation is available at:
- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## Production API Notes

### Request Logging

The API includes request logging middleware that records the HTTP method, request path, response status code, and request duration.

### Error Handling

Unexpected API errors are returned using a standardized error response format.

Example:

```json
{
  "detail": "Internal server error.",
  "error_code": "internal_server_error"
}
```

### CORS Configuration

CORS origins are configurable via the `API_CORS_ORIGINS` environment variable.

Example:

```env
API_CORS_ORIGINS=http://localhost:3000,http://localhost:5173
```

### Security and Rate Limiting

For production deployments, add security headers and rate limiting at the API gateway, reverse proxy, or middleware layer to protect expensive endpoints such as `/api/v1/query` and `/api/v1/ingest`.

---

## Health, Readiness, and Startup Warmup

### Health Check

`GET /health` is a lightweight liveness endpoint.

It only confirms that the API process is running and does not load heavy dependencies such as the RAG graph, LLM, embeddings, reranker, BM25 index, Tavily, or Chroma vectorstore.

### Readiness Check

`GET /ready` checks lightweight runtime readiness.

It verifies that required configuration is loaded and that the Chroma persistence path is configured and accessible.

Example response:

```json
{
  "status": "ready",
  "checks": {
    "config_loaded": true,
    "chroma_path_configured": true,
    "chroma_path_accessible": true
  }
}
```

If readiness checks fail, the endpoint returns `503 Service Unavailable`.

### Startup Warmup

Startup warmup is controlled by:

```env
API_ENABLE_STARTUP_WARMUP=false
```
By default, warmup is disabled so the API can start quickly and /health remains lightweight.

Heavy components such as LLMs, embedding models, BM25 index, reranker, and the full RAG graph are not loaded during startup by default. They are initialized lazily when needed.

---

## Endpoints

### 1. Health Check
Checks the status of the API server.

* **URL**: `/health`
* **Method**: `GET`
* **Response**: `200 OK`
  ```json
  {
    "status": "ok",
    "service": "Adaptive RAG"
  }
  ```

---

### 2. Readiness Check

Checks whether lightweight runtime dependencies are ready.

* **URL**: `/ready`
* **Method**: `GET`
* **Response**: `200 OK`
  ```json
  {
    "status": "ready",
    "checks": {
      "config_loaded": true,
      "chroma_path_configured": true,
      "chroma_path_accessible": true
    }
  }
  ```

* **Errors**:
  * `503 Service Unavailable`: If one or more readiness checks fail.

---

### 3. Query AdaptiveRAG
Submits a query to the AdaptiveRAG agent, running it through the LangGraph decision workflow.

* **URL**: `/api/v1/query`
* **Method**: `POST`
* **Headers**: `Content-Type: application/json`
* **Request Body**:
  ```json
  {
    "query": "What is an embedding?",
    "include_sources": true
  }
  ```
* **Response**: `200 OK`
  ```json
  {
    "answer": "An embedding is a low-dimensional space into which high-dimensional vectors can be translated...",
    "sources": [
      {
        "source": "embeddings_guide.pdf",
        "page_number": 3,
        "chunk_id": "doc_chunk_1"
      }
    ],
    "route": "retrieval"
  }
  ```
* **Errors**:
  * `400 Bad Request`: If the query is empty.
    ```json
    {
      "detail": "Query cannot be empty."
    }
    ```
  * `500 Internal Server Error`: If processing the query fails.
    ```json
    {
      "detail": "Failed to process query."
    }
    ```

---

### 4. Ingest PDF
Uploads and processes a PDF document, chunking it, storing the embeddings in the Chroma vector store, and updating the BM25 search index.

* **URL**: `/api/v1/ingest`
* **Method**: `POST`
* **Headers**: `Content-Type: multipart/form-data`
* **Request Body**:
  * `file` (binary, required): The PDF file to ingest.
* **Response**: `200 OK`
  ```json
  {
    "status": "success",
    "filename": "embeddings_guide.pdf",
    "chunks_created": 15,
    "message": "PDF uploaded and ingested successfully."
  }
  ```
* **Errors**:
  * `400 Bad Request`: If the file uploaded is not a PDF.
    ```json
    {
      "detail": "Only PDF files are supported."
    }
    ```
  * `500 Internal Server Error`: If ingestion fails.
    ```json
    {
      "detail": "Failed to ingest PDF"
    }
    ```

---

### 5. List Ingested Documents
Retrieves a list of all documents currently ingested and available in the vector database.

* **URL**: `/api/v1/documents`
* **Method**: `GET`
* **Response**: `200 OK`
  ```json
  {
    "documents": [
      {
        "document_id": "embeddings_guide.pdf",
        "filename": "embeddings_guide.pdf",
        "chunk_count": 15
      }
    ]
  }
  ```
* **Errors**:
  * `500 Internal Server Error`: If retrieval fails.
    ```json
    {
      "detail": "Failed to list documents"
    }
    ```

---

### 6. Delete Ingested Document
Deletes an ingested document, removing all its chunks from the Chroma vector store, removing the PDF file from disk, and resetting/updating the BM25 search index.

* **URL**: `/api/v1/documents/{document_id}`
* **Method**: `DELETE`
* **Request Parameters**:
  * `document_id` (Path, string, required): The ID/filename of the document to delete.
* **Response**: `200 OK`
  ```json
  {
    "status": "success",
    "document_id": "embeddings_guide.pdf",
    "chunks_deleted": 15,
    "message": "Document deleted successfully."
  }
  ```
* **Errors**:
  * `404 Not Found`: If the document does not exist.
    ```json
    {
      "detail": "Document not found."
    }
    ```
  * `500 Internal Server Error`: If deletion fails.
    ```json
    {
      "detail": "Failed to delete document."
    }
    ```
