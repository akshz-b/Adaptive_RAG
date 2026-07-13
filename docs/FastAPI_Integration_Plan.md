# AdaptiveRAG FastAPI Integration Plan

> Goal: Add a production-ready FastAPI layer to the existing AdaptiveRAG agent so it can be used as a web service for querying documents, ingesting PDFs, and exposing traces/evaluation-friendly endpoints.
>
> This document is a step-by-step implementation plan for the next phase of the project.

---

## 1. Why We Are Adding FastAPI

The current project is a strong Python-based research-grade agent with:

- LangGraph orchestration
- Hybrid retrieval + reranking
- CRAG grading
- Adaptive routing
- LangSmith tracing
- CLI-based interaction

However, it is still not usable as a real backend service. To make it practical for demos, internal tools, or deployment, we need a web API layer.

FastAPI is the right next step because it allows us to:

- expose the RAG agent over HTTP
- accept user queries from apps or UI
- upload and ingest PDFs through an API
- make the project production-ready for local deployment
- prepare the system for future frontend integration

---

## 2. Final Goal

By the end of this phase, the project should have:

- a FastAPI application running locally
- a POST endpoint for answering questions
- a POST endpoint for uploading and ingesting PDFs
- health endpoint for service monitoring
- clean request/response models using Pydantic
- LangSmith tracing tied to each incoming request
- a structure that is easy to extend later for authentication, async support, streaming, and deployment

---

## 3. High-Level Architecture

```text
Client / Browser / Postman / Frontend
            |
            v
      FastAPI Application
            |
            v
      AdaptiveRAG Graph
   (Router -> Retriever -> Grader -> Generator -> Critic)
            |
            v
   ChromaDB / LLM / Tavily / LangSmith
```

The FastAPI layer will sit in front of the existing LangGraph agent and will not replace its logic. Instead, it will expose the existing graph as an API service.

---

## 4. Project Goals

### Primary Goals

1. Expose the existing AdaptiveRAG workflow through an HTTP API.
2. Keep the current LangGraph logic intact and reusable.
3. Make the project easier to demo and integrate with frontend apps.
4. Make the service suitable for local deployment and future cloud deployment.

### Secondary Goals

1. Add request-level metadata for LangSmith tracing.
2. Support document ingestion through API upload.
3. Improve code organization by separating API concerns from core RAG logic.
4. Prepare the codebase for authentication and production hardening later.

---

## 5. Non-Goals for This Phase

This phase is not about building a full SaaS platform. We are not trying to implement yet:

- user authentication
- multi-tenant architecture
- database-backed session management
- streaming token responses
- advanced UI frontend
- Kubernetes deployment
- fully asynchronous model serving

These can be added later once the core API is stable.

---

## 6. Proposed Project Structure Changes

We will add a new API package while keeping the existing project structure intact.

```text
src/
  api/
    __init__.py
    app.py
    schemas.py
    deps.py
    utils.py
```

Optional additions later:

```text
src/api/
  middleware.py
  auth.py
  logging.py
```

---

## 7. Core API Endpoints

### 7.1 POST /query

Purpose:
- Accept a user question
- Run the AdaptiveRAG graph
- Return the answer, route, citations, and optional debug data

Request body example:

```json
{
  "query": "What is an embedding?",
  "debug": false
}
```

Response example:

```json
{
  "answer": "An embedding is a numerical vector representation of data.",
  "route": "direct",
  "citations": [],
  "final_context": "",
  "status": "success"
}
```

### 7.2 POST /ingest

Purpose:
- Accept a PDF upload
- Save the file into the documents folder
- Ingest it into ChromaDB
- Rebuild or refresh the BM25 index

Request type:
- multipart/form-data

Fields:
- file: PDF file
- source_name: optional custom name

Response example:

```json
{
  "status": "success",
  "message": "PDF ingested successfully",
  "file_name": "paper.pdf"
}
```

### 7.3 GET /health

Purpose:
- verify the API is up
- optionally check whether Chroma and model environment are ready

Response example:

```json
{
  "status": "ok"
}
```

### 7.4 GET /docs

Purpose:
- FastAPI auto-generated Swagger docs
- useful for testing and demoing

---

## 8. Step-by-Step Implementation Plan

## Phase 0 — API Foundation Setup

### Goal
Create the base FastAPI app and minimal package structure.

### Tasks

1. Create the package structure:
   - src/api/__init__.py
   - src/api/app.py
   - src/api/schemas.py

2. Add FastAPI and Uvicorn to dependencies.

3. Create a minimal app instance that returns a simple health response.

4. Ensure the app can be started locally with:

```bash
uvicorn src.api.app:app --reload --port 8000
```

### Acceptance Criteria

- Running the app shows the FastAPI docs at `/docs`
- `/health` responds successfully

---

## Phase 1 — Request and Response Schemas

### Goal
Create clean, typed request/response models.

### Tasks

1. Create Pydantic models for:
   - QueryRequest
   - QueryResponse
   - IngestRequest or multipart upload handling
   - HealthResponse

2. Keep response schema simple for the first version.

3. Add optional flags such as:
   - debug
   - include_context
   - include_citations

### Acceptance Criteria

- Requests are validated automatically
- Invalid input returns clean 422 validation errors
- API responses are structured and predictable

---

## Phase 2 — Query Endpoint Integration

### Goal
Wire the existing AdaptiveRAG graph into a POST endpoint.

### Tasks

1. Import the existing `rag_graph` from the current project.

2. Create a route that accepts a query and calls:

```python
rag_graph.invoke(...)
```

3. Pass metadata for LangSmith tracing:
   - request_id
   - endpoint
   - stage
   - query length

4. Return:
   - answer
   - route
   - citations
   - final_context
   - optional debug details

### Acceptance Criteria

- A POST request to `/query` returns a valid answer
- The response contains the same final answer logic as the CLI version
- LangSmith receives a trace for each request

---

## Phase 3 — Ingestion Endpoint

### Goal
Allow users to upload PDFs through the API and ingest them into the knowledge base.

### Tasks

1. Accept uploaded PDF files using `UploadFile`.

2. Save the file into a safe location such as:

```text
data/documents/
```

3. Reuse the current ingestion pipeline:
   - load PDF
   - chunk it
   - store it in Chroma

4. After ingestion:
   - reset BM25 index cache if needed
   - return success response

### Acceptance Criteria

- Uploading a PDF creates a stored file in the documents directory
- The document becomes available in the vector store for future queries
- A clear success or failure response is returned

---

## Phase 4 — LangSmith Request Tracing

### Goal
Make each API request traceable inside LangSmith.

### Tasks

1. Add metadata to each invocation:
   - endpoint: `query`
   - request_id
   - query_type
   - debug

2. Add tags such as:
   - `api`
   - `adaptive-rag`
   - `stage-fastapi`

3. If needed, add request-scoped logging with correlation IDs.

### Acceptance Criteria

- A request to `/query` appears in LangSmith with request metadata
- The trace can be filtered by API usage

---

## Phase 5 — Error Handling and Robustness

### Goal
Make the API safe and predictable.

### Tasks

1. Catch model/LLM failures and return a graceful error response.

2. Handle ingestion failures with clear messages.

3. Add timeout handling for long-running requests.

4. Avoid crashing the whole service when one part fails.

### Acceptance Criteria

- The API returns 200/400/500 appropriately depending on the problem
- Users receive a clear response instead of a raw stack trace

---

## Phase 6 — Testing and Verification

### Goal
Verify the new API works end to end.

### Tasks

1. Add tests for:
   - health endpoint
   - query endpoint with a simple question
   - invalid request data
   - ingestion endpoint with a sample PDF

2. Run local smoke tests using:

```bash
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" -d '{"query":"What is an embedding?"}'
```

3. Check that the service works both with CLI and API entry points.

### Acceptance Criteria

- See successful responses from each endpoint
- Tests pass without breaking the core graph logic

---

## Phase 7 — Deployment Preparation

### Goal
Prepare the service for simple deployment.

### Tasks

1. Add a `Dockerfile`.

2. Optionally add a `docker-compose.yml` file.

3. Add startup instructions for local and container-based execution.

4. Prepare environment variables required for running in production.

### Acceptance Criteria

- The service can be started via Docker or local Uvicorn
- The environment is documented clearly

---

## 9. Suggested Implementation Order

We should implement in this exact order:

1. FastAPI app skeleton
2. health endpoint
3. query endpoint
4. Pydantic schemas
5. ingestion endpoint
6. LangSmith metadata
7. error handling
8. tests
9. deployment prep

This order keeps the scope manageable and avoids overcomplicating things early.

---

## 10. Recommended Coding Principles

During implementation, we should follow these principles:

- Keep the API thin
- Do not duplicate core RAG logic inside the API layer
- Reuse existing modules instead of rewriting them
- Keep request/response models simple
- Make failures graceful and observable
- Keep the code modular so future authentication and streaming can be added easily

---

## 11. Suggested API Design Style

### Keep the endpoints simple

- `/query` for answering questions
- `/ingest` for uploading documents
- `/health` for liveness

### Keep the response format simple

Return something like:

```json
{
  "status": "success",
  "answer": "...",
  "route": "retrieval",
  "citations": []
}
```

Avoid over-encoding the response in the first version.

---

## 12. Milestones Checklist

### Milestone 1 — API skeleton
- [ ] FastAPI app created
- [ ] `/health` works
- [ ] `/docs` works

### Milestone 2 — Query API
- [ ] `/query` works
- [ ] It returns an answer from the graph
- [ ] LangSmith trace is generated

### Milestone 3 — Ingestion API
- [ ] PDF upload works
- [ ] File is stored
- [ ] ChromaDB is updated

### Milestone 4 — Hardening
- [ ] Error responses are clean
- [ ] Logging is structured
- [ ] Basic tests pass

### Milestone 5 — Deployment prep
- [ ] Dockerfile added
- [ ] Local run instructions documented

---

## 13. Definition of Done

This phase will be considered complete when:

- the FastAPI app runs locally
- the query endpoint works end to end
- the ingestion endpoint works end to end
- traces are visible in LangSmith
- the project remains modular and easy to extend

---

## 14. Recommended Next Step

The first implementation task should be:

1. create the FastAPI app skeleton
2. add the `/health` endpoint
3. add the `/query` endpoint
4. test it locally

Once that is working, we will add `/ingest` and then improve robustness.

---

## 15. Suggested Working Style for the Next Session

We will proceed in small, verified steps:

- implement one endpoint at a time
- run the app locally after each milestone
- test it with real requests
- only then move to the next step

This keeps the work organized and reduces errors.
