import argparse
from typing import List
import logging
from src.graph import rag_graph
from src.utils.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def _print_chunks(title: str, chunks: List[dict]) -> None:
    """Print graph-state chunk dictionaries for debugging."""
    print(f"\n{title}")
    print("=" * 100)

    for i, chunk in enumerate(chunks, start=1):
        metadata = chunk.get("metadata", {})
        page_content = chunk.get("page_content", "")

        print(f"\nChunk {i}")
        print("-" * 80)
        print("Chunk ID:", metadata.get("chunk_id"))
        print("Source:", metadata.get("source"))
        print("Page:", metadata.get("page_number"))
        print("Chunk index:", metadata.get("chunk_index"))
        print("BM25 score:", metadata.get("bm25_score"))
        print("RRF score:", metadata.get("rrf_score"))
        print("Reranker score:", metadata.get("reranker_score"))
        print("\nPreview:")
        print(page_content[:400])


def _print_web_results(web_results: List[dict]) -> None:
    """Print web search results for debugging."""
    print("\nWEB RESULTS")
    print("=" * 100)

    for i, result in enumerate(web_results, start=1):
        print(f"\nWeb Result {i}")
        print("-" * 80)
        print("Title:", result.get("title"))
        print("URL:", result.get("url"))
        print("\nContent:")
        print(result.get("content", "")[:400])


def _print_debug_state(state: dict) -> None:
    """Print useful LangGraph state fields for debugging."""
    print("\nGRAPH DEBUG STATE")
    print("=" * 100)

    print("\nOriginal query:")
    print(state.get("query"))

    print("\nRewritten query:")
    print(state.get("rewritten_query"))

    print("\nMultiple queries:")
    for query in state.get("multiple_queries", []):
        print("-", query)

    print("\nChunk grades:")
    print(state.get("chunk_grades", []))

    retrieved_chunks = state.get("retrieved_chunks", [])
    reranked_chunks = state.get("reranked_chunks", [])
    web_results = state.get("web_results", [])

    if retrieved_chunks:
        _print_chunks("RETRIEVED CHUNKS", retrieved_chunks)

    if reranked_chunks:
        _print_chunks("RERANKED CHUNKS", reranked_chunks)

    if web_results:
        _print_web_results(web_results)

    print("\nCitations:")
    for citation in state.get("citations", []):
        print("-", citation)


def main() -> None:
    """
    Run AdaptiveRAG Stage 5 LangGraph question-answering loop.

    Flow:
    user question
        -> router node
            -> Direct answer or web search OR retrieval pipeline
        -> optional critic node
        -> final answer
    """
    parser = argparse.ArgumentParser(
        description="Run AdaptiveRAG Stage 5 Adaptive router + self-rag pipeline."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print graph state, retrieved chunks, grades, and web results.",
    )

    args = parser.parse_args()

    print("AdaptiveRAG Stage 5 is ready.")
    print(
        "Flow: router node -> direct/retrieval/web -> generator -> critic -> final answer"
    )
    print("Type 'exit' to quit.\n")

    while True:
        question = input("Question: ").strip()

        if question.lower() in {"exit", "quit"}:
            print("Exiting AdaptiveRAG.")
            break

        if not question:
            print("Please enter a question.\n")
            continue

        state = rag_graph.invoke(
            {
                "query": question,
                "retry_count": 0,
            },
            config={
                "tags": ["stage5", "adaptive-rag"],
                "metadata": {
                    "app-stage": "stage-6",
                    "pipeline": "adaptive-router-self-rag",
                },
            },
        )

        if args.debug:
            _print_debug_state(state)

        print("\nAnswer:")
        print(state.get("answer", "I don't know."))
        print()


if __name__ == "__main__":
    main()
