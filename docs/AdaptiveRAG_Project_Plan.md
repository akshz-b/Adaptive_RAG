# AdaptiveRAG Agent — Complete Project Plan

> A production-grade Adaptive RAG system built step by step.
> Goal: Resume-worthy project covering LangChain + LangGraph + LangSmith.
> Approach: Start simple, add one enhancement at a time, always production structure.

---

## How to use this document

- Follow stages in order. Never skip a stage.
- Each stage has a clear goal. Finish that goal before moving to the next.
- You write the code. Use AI to guide, correct, and explain — not to write everything for you.
- At the end of every stage, your code should be working and committed to GitHub.

---

## Project Overview

**Name:** AdaptiveRAG Agent

**What it does:** An intelligent document Q&A system that retrieves from a knowledge base, grades its own retrieval quality, searches the web when documents aren't enough, critiques its own answers, and adapts its retrieval strategy based on query complexity.

**Why it stands out:** Covers every major RAG pattern used in production in 2026 — hybrid search, reranking, CRAG, Self-RAG, Adaptive routing — built with LangGraph for orchestration and LangSmith for observability.

---

## Final Architecture (what you're building toward)

```text
User Query
    ↓
Adaptive Router (LangGraph node)
    ├── simple factual → answer directly, no retrieval
    ├── medium → Hybrid Search → Rerank → Generate → Answer
    └── complex ↓
        Query Rewriter → HyDE
                ↓
        Hybrid Search (Semantic + BM25) → Reranker
                ↓
        CRAG Grader — grades each retrieved chunk
            ├── relevant   → Generate Answer
            ├── ambiguous  → Web Search + Docs → Generate
            └── irrelevant → Web Search only → Generate
                    ↓
            Self-RAG Critic
                ├── answer grounded → return with citations
                └── not grounded    → re-retrieve → generate again
```

---

## Tech Stack

| Component | Tool | Why |
| --- | --- | --- |
| Language | Python 3.11+ | Industry standard |
| Package manager | `uv` | Modern, fast, replaces pip |
| Vector store | ChromaDB | Free, local, no setup needed |
| Keyword search | BM25 via `rank_bm25` | Free, pairs with ChromaDB for hybrid search |
| Embeddings | `BAAI/bge-base-en-v1.5` via HuggingFace | Free, excellent quality |
| Reranker | `BAAI/bge-reranker-base` via HuggingFace | Free, strong reranking |
| LLM | Qwen/Qwen2.5-72B-Instruct via HuggingFace | What you already use, supports tools |
| Orchestration | LangGraph | Agent decision making |
| Observability | LangSmith | Trace every step |
| Web search fallback | TavilySearch | Already familiar |
| PDF loading | `pypdf` or `pymupdf` | Standard document loaders |
| Config management | `pydantic-settings` | Production standard |
| Environment | `.env` via `python-dotenv` | Already familiar |
| Version control | Git + GitHub | Required |

---

## Production Project Structure

This is the structure you will maintain from Stage 1 onwards. Do not start coding without this in place.

```text
adaptive-rag/
│
├── src/
│   ├── __init__.py
│   ├── config.py          # All settings via pydantic-settings
│   ├── client.py          # LLM + embeddings clients
│   ├── prompts.py         # All prompt templates
│   ├── tools.py           # Tavily and any other tools
│   ├── state.py           # LangGraph State definition
│   │
│   ├── ingestion/         # Everything about loading + storing docs
│   │   ├── __init__.py
│   │   ├── loader.py      # PDF loading and chunking
│   │   ├── embedder.py    # Embedding logic
│   │   └── vectorstore.py # ChromaDB setup and operations
│   │
│   ├── retrieval/         # Everything about finding relevant chunks
│   │   ├── __init__.py
│   │   ├── semantic.py    # Vector/semantic search
│   │   ├── keyword.py     # BM25 keyword search
│   │   ├── hybrid.py      # Combines both with RRF
│   │   └── reranker.py    # Reranking retrieved chunks
│   │
│   ├── nodes/             # LangGraph node functions
│   │   ├── __init__.py
│   │   ├── router.py      # Adaptive router node
│   │   ├── retriever.py   # Retrieval node
│   │   ├── grader.py      # CRAG grader node
│   │   ├── generator.py   # Answer generation node
│   │   ├── critic.py      # Self-RAG critic node
│   │   └── rewriter.py    # Query rewriter node
│   │
│   └── graph.py           # LangGraph graph assembly
│
├── data/
│   └── documents/         # Put your PDFs here
│
├── tests/                 # Unit tests (added in Stage 5)
│   └── __init__.py
│
├── notebooks/             # Jupyter notebooks for experimentation
│   └── exploration.ipynb
│
├── main.py                # Entry point
├── ingest.py              # Script to ingest documents into vectorstore
├── pyproject.toml         # Project metadata and dependencies
├── .env                   # API keys (never commit this)
├── .env.example           # Template for teammates
├── .gitignore             # Must include .env, __pycache__, .chroma
└── README.md              # Updated at every stage
```

---

## Stage 0 — Project Setup (Day 1)

**Goal:** Clean project skeleton running with no errors before writing any RAG logic.

### What to do:

1. Create GitHub repo named `adaptive-rag`
2. Clone it locally
3. Create the full folder structure above (empty files with just `pass` or `# TODO`)
4. Set up `pyproject.toml` with all dependencies listed
5. Create `.env` with your API keys
6. Create `.env.example` with placeholder values
7. Create `.gitignore`
8. Set up `src/config.py` using `pydantic-settings` — all keys and constants live here
9. Set up `src/client.py` — LLM client (your existing Qwen setup)
10. Commit:
   ```bash
   git commit -m "stage-0: project skeleton"
   ```

### `config.py` should cover:

```python
HUGGINGFACE_API_KEY
TAVILY_API_KEY
LANGSMITH_API_KEY
LANGSMITH_PROJECT = "adaptive-rag"

LLM_MODEL = "Qwen/Qwen2.5-72B-Instruct"
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
RERANKER_MODEL = "BAAI/bge-reranker-base"

CHROMA_PERSIST_DIR = "./data/chroma"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

TOP_K_RETRIEVAL = 10
TOP_K_RERANK = 4
```

### Acceptance criteria:

- `python main.py` runs without errors (even if it does nothing yet)
- All API keys load correctly from `.env`
- Repo is on GitHub with first commit

---

## Stage 1 — Naive RAG (Days 2–3)

**Goal:** A working end-to-end RAG pipeline.

Load PDF → chunk → embed → store → retrieve → answer.

### What to build:

#### `src/ingestion/loader.py`

- Load PDF using `pypdf`
- Split into chunks using `RecursiveCharacterTextSplitter`
- Add metadata to each chunk:
  - `source`
  - `page_number`
  - `chunk_index`

#### `src/ingestion/embedder.py`

- Load `BAAI/bge-base-en-v1.5` from HuggingFace
- Function to embed a list of texts
- Function to embed a single query

#### `src/ingestion/vectorstore.py`

- Initialize ChromaDB with persist directory from config
- Function to store chunks with embeddings
- Function to retrieve top-k similar chunks for a query

#### `ingest.py`

- Script that takes a PDF path, loads it, chunks it, embeds it, stores in ChromaDB
- Run this once to populate the knowledge base

#### `src/retrieval/semantic.py`

- Function takes a query string, returns top-k chunks from ChromaDB

#### `main.py`

- Simple loop:
  user types question → semantic retrieval → LLM answers with context → print answer

### Key concepts to understand at this stage:

- What is a chunk and why does size matter
- What is an embedding and what does it represent
- What does ChromaDB actually store
- What does `similarity_search` return

### Acceptance criteria:

- `python ingest.py --file data/documents/your.pdf` populates ChromaDB
- `python main.py` answers questions from the document
- Every step is visible in LangSmith traces
- Commit:
  ```bash
  git commit -m "stage-1: naive rag working"
  ```

---

## Stage 2 — Hybrid Search + Reranking (Days 4–5)

**Goal:** Retrieval that doesn't miss exact keyword matches and ranks results by true relevance.

### Why you need this:

Run Stage 1. Ask a question containing a specific name, number, or acronym. Notice it misses obvious matches. That's the problem hybrid search fixes.

### What to build:

#### `src/retrieval/keyword.py`

- Build a BM25 index from your stored chunks using `rank_bm25`
- Function: takes query string, returns top-k chunks by keyword match
- **Important:** BM25 index must be rebuilt when new documents are ingested

#### `src/retrieval/hybrid.py`

- Takes results from `semantic.py` and `keyword.py`
- Combines them using Reciprocal Rank Fusion (RRF)
- RRF formula:
  ```text
  score = 1 / (rank + 60)
  ```
  Sum scores across both lists
- Returns unified ranked list

#### `src/retrieval/reranker.py`

- Load `BAAI/bge-reranker-base` from HuggingFace
- Takes query + list of chunks
- Scores each `(query, chunk)` pair for true relevance
- Returns chunks sorted by reranker score, keep top `TOP_K_RERANK`

### Update `main.py`

Replace:

```python
semantic_retrieval
```

with:

```python
hybrid_retrieval → reranker
```

### Key concepts to understand:

- Why semantic search alone misses exact matches
- What BM25 is (keyword frequency scoring, no embeddings)
- What RRF does (merges two ranked lists fairly)
- Why reranking after retrieval is different from retrieval itself

### Acceptance criteria:

- Same question from Stage 1 that missed exact match now returns correct chunk
- Reranker visibly reorders the retrieved chunks
- Commit:

```bash
git commit -m "stage-2: hybrid search and reranking"
```

---

## Stage 3 — Query Intelligence (Days 6–7)

**Goal:** Handle vague, short, or poorly phrased user questions better.

### What to build:

#### `src/nodes/rewriter.py`

- Query rewriter: takes original query, asks LLM to rewrite it as a clearer search query
- Multi-query: asks LLM to generate 3 different versions of the question
- Retrieve for all 3 versions, deduplicate results, then rerank

#### HyDE (Hypothetical Document Embeddings) — inside `src/retrieval/semantic.py`

- Instead of embedding the question directly, ask LLM to generate a hypothetical answer paragraph
- Embed that paragraph and search with it
- Why: a short question and a long document answer are semantically far apart; a hypothetical answer bridges that gap

### Update `main.py`

Add query rewriting before retrieval.

### Key concepts to understand:

- Why short queries perform poorly in semantic search
- What HyDE does and when it helps
- Trade-off: HyDE adds one LLM call before retrieval (latency cost vs quality gain)

### Acceptance criteria:

- Vague question like `"tell me about the main findings"` now retrieves better chunks than before
- Commit:

```bash
git commit -m "stage-3: query intelligence"
```

---

## Stage 4 — LangGraph Agent + CRAG (Days 8–10)

**Goal:** Turn the RAG pipeline into a LangGraph agent with Corrective RAG.

### This is where LangGraph comes in

Up to Stage 3, your RAG was a linear pipeline:

```text
query → rewrite → retrieve → rerank → generate
```

Now you add decision making. The agent grades retrieved chunks and decides what to do next.

### What to build:

#### `src/state.py`

```python
from langgraph.graph import MessagesState
from typing import List

class RAGState(MessagesState):
    query: str
    rewritten_query: str
    retrieved_chunks: List[dict]
    chunk_grades: List[str]      # "relevant" / "irrelevant" / "ambiguous"
    web_results: List[dict]
    final_context: str
    answer: str
    citations: List[str]
```

#### `src/nodes/retriever.py`

- LangGraph node that runs hybrid retrieval + reranking
- Reads `query` from state, writes `retrieved_chunks` to state

#### `src/nodes/grader.py` *(this is CRAG)*

- For each retrieved chunk, ask LLM:
  > "Is this chunk relevant to the query? Answer: relevant / irrelevant / ambiguous."
- Write `chunk_grades` to state

Conditional logic:

- All relevant → proceed to generate
- Any ambiguous → keep relevant chunks + trigger web search
- All irrelevant → discard all → trigger web search only

#### `src/nodes/generator.py`

- Takes `final_context` (chunks + optional web results) from state
- Generates answer with citations
- Each claim in answer references a source chunk or web result

#### `src/graph.py`

```text
START
  ↓
rewriter_node
  ↓
retriever_node
  ↓
grader_node
  ↓ (conditional)
├── all relevant   → generator_node → END
├── ambiguous      → web_search_node → generator_node → END
└── all irrelevant → web_search_node → generator_node → END
```

### Key concepts to understand:

- Why CRAG matters: without it, your agent answers confidently even when retrieved chunks are wrong
- The three CRAG paths: correct / ambiguous / incorrect
- How LangGraph state flows between nodes

### Acceptance criteria:

- Ask a question your documents don't contain → agent uses web search instead
- Ask a question your documents contain → agent uses docs
- LangSmith trace shows grading decision at each run
- Commit:

```bash
git commit -m "stage-4: langgraph agent with CRAG"
```

---

## Stage 5 — Adaptive Router + Self-RAG (Days 11–13)

**Goal:** Agent adapts retrieval depth to query complexity. Agent critiques its own answers.

### What to build:

#### `src/nodes/router.py` — Adaptive Router

- Classify incoming query into one of three types:
  - `direct` → simple factual question LLM can answer from training (e.g., *"What is an embedding?"*)
  - `retrieval` → needs document search
  - `web` → needs live web search (current events, latest news)
- Write classification to state
- Graph routes to different paths based on classification

#### `src/nodes/critic.py` — Self-RAG

After `generator_node` produces an answer, critic evaluates it:

- "Is every claim in this answer supported by the provided context?"
- If **yes** → return answer
- If **no** → identify unsupported claims → re-retrieve specifically for those claims → regenerate
- Add iteration counter to state to prevent infinite loops (max 2 retries)

### Update `src/graph.py`

```text
START
  ↓
router_node
├── direct
│      → generator_node (no retrieval) → END
├── web
│      → web_search_node → generator_node → critic_node → END
└── retrieval
       ↓
    rewriter_node
       ↓
    retriever_node
       ↓
    grader_node (CRAG)
       ↓
    generator_node
       ↓
    critic_node
       ├── grounded → END
       └── not grounded → retriever_node (loop, max 2x)
```

### Key concepts to understand:

- Adaptive routing saves compute — don't run full RAG pipeline for *"What is Python?"*
- Self-RAG prevents hallucination by making agent verify its own output
- Iteration limits are critical — without them agents loop forever

### Acceptance criteria:

- Simple question answered directly without retrieval (LangSmith confirms no retrieval node ran)
- Answer that references unsupported claims gets re-retrieved and corrected
- Commit:

```bash
git commit -m "stage-5: adaptive router and self-rag critic"
```

---

## Stage 6 — LangSmith Observability + Evaluation (Days 14–15)

**Goal:** Make the system measurable and debuggable. This is what separates a toy project from a production system.

### What to build:

#### Tracing *(already partially working from LangSmith setup)*

- Add custom metadata to every trace:
  - `query_type`
  - `chunks_retrieved`
  - `grade_decision`
  - `retry_count`
- Tag traces by stage for filtering in LangSmith dashboard

#### Evaluation dataset

- Create 20 question-answer pairs from your documents manually
- These are your ground truth examples
- Store as a LangSmith dataset

#### Evaluators *(write these using LangSmith SDK)*

- `answer_relevance`: does the answer actually address the question?
- `faithfulness`: is every claim in the answer supported by retrieved context?
- `retrieval_precision`: what fraction of retrieved chunks were actually used?
- `citation_accuracy`: do citations point to real chunks?

#### Run evaluations

```python
from langsmith import evaluate

results = evaluate(
    your_rag_pipeline,
    data="your-dataset-name",
    evaluators=[
        answer_relevance,
        faithfulness,
        retrieval_precision,
    ],
)
```

### Key concepts to understand:

- Why evaluation matters: you can't improve what you can't measure
- Difference between tracing (what happened) and evaluation (how good was it)
- Faithfulness vs relevance — two different failure modes

### Acceptance criteria:

- LangSmith dashboard shows all traces with custom metadata
- Evaluation run completes with scores for all 3 evaluators
- You can identify at least one failure case from the eval results
- Commit:

```bash
git commit -m "stage-6: langsmith observability and evaluation"
```

---

## Stage 7 — Production Polish (Days 16–18)

**Goal:** Make this look and feel like a real project, not a script.

### What to do:

#### Code quality

- Add type hints to every function
- Add docstrings to every function (what it does, args, returns)
- Remove all `print()` statements — replace with `logging`
- Run `ruff` for linting

#### Error handling

- Every external call (LLM, ChromaDB, Tavily, HuggingFace) wrapped in `try/except`
- Graceful degradation: if reranker fails, fall back to retriever results
- If web search fails, answer from docs only with a note

#### `README.md` — this is your resume artifact

Write a README that covers:

1. What problem this solves
2. Architecture diagram (draw with ASCII or Mermaid)
3. Every RAG technique implemented and why
4. Tech stack with justification
5. How to run it locally (step by step)
6. Example queries and outputs with LangSmith trace screenshots
7. What you'd add next (shows engineering thinking)

#### Demo preparation

Prepare 5 demo questions that show off different capabilities:

- One that triggers web search fallback (CRAG irrelevant path)
- One that triggers Self-RAG retry
- One that gets answered directly without retrieval
- One that shows hybrid search beating semantic-only
- One that shows citations working

Commit:

```bash
git commit -m "stage-7: production polish"
```

---

## What This Covers for Your Resume

| Skill | Where it appears |
| --- | --- |
| LangChain | Embeddings, document loaders, prompt templates, retrievers |
| LangGraph | Agent orchestration, conditional edges, state management, human-in-the-loop |
| LangSmith | Tracing, evaluation, datasets, custom metrics |
| RAG | Naive → Hybrid → Reranking → CRAG → Self-RAG → Adaptive |
| Vector databases | ChromaDB — ingestion, retrieval, persistence |
| Production code structure | Proper Python project layout, config management, error handling |
| Evaluation mindset | Custom evaluators, ground truth datasets, measuring quality |

---

## How to Resume This Plan with Any AI

If you start a new conversation with any AI assistant, paste this at the top:

> "I am building a production-grade Adaptive RAG system called AdaptiveRAG Agent. I have a complete project plan. I am currently on Stage X. Here is my current code: [paste relevant files]. Help me implement the next step following the plan. Do not give me complete code — guide me step by step, explain concepts, and correct my mistakes."

---

## Rules to Follow Throughout

1. **One stage at a time.** Never start Stage N+1 until Stage N is working and committed.
2. **You write the code.** Use AI to guide, explain, and correct — not to write everything.
3. **Commit at every stage.** Your Git history tells the story of how you built it.
4. **Always production structure.** No spaghetti scripts. Every file has one job.
5. **Understand every line.** If you can't explain a line, you haven't learned it.
6. **Test with real questions.** At every stage, run 3–5 real questions and see what breaks.
7. **LangSmith open always.** Every run should be traceable. If it's not in LangSmith, it didn't happen.

---

## Documents to use for demo

Use any publicly available PDFs. Good options:

- Download 2–3 Wikipedia articles as PDF (any topic you like)
- Any publicly available research paper from arxiv.org
- Any company annual report (publicly listed companies)

Keep it to 3–5 documents for now. Quality of the system matters more than quantity of documents.

---

*Last updated: Start of project. Update this document as you complete each stage.*
