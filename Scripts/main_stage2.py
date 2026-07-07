import argparse

from langchain_core.prompts import ChatPromptTemplate

from src.client import get_llm
from src.config import settings
from src.retrieval.hybrid import hybrid_search
from src.retrieval.reranker import rerank


PROMPT_TEMPLATE = """
Use the following context to answer the question.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{question}
"""


def _print_chunks(title: str, chunks) -> None:
    """
    Print retrieved/reranked chunks for debugging.
    """

    print(f"\n{title}")
    print("=" * 100)

    for i, doc in enumerate(chunks, start=1):
        print(f"\nChunk {i}")
        print("-" * 80)
        print("Chunk ID:", doc.metadata.get("chunk_id"))
        print("Source:", doc.metadata.get("source"))
        print("Page:", doc.metadata.get("page_number"))
        print("Chunk index:", doc.metadata.get("chunk_index"))
        print("BM25 score:", doc.metadata.get("bm25_score"))
        print("RRF score:", doc.metadata.get("rrf_score"))
        print("Reranker score:", doc.metadata.get("reranker_score"))
        print("\nPreview:")
        print(doc.page_content[:400])


def main() -> None:
    """
    Run AdaptiveRAG Stage 2 question-answering loop.

    Flow:
    user question -> hybrid search -> rerank -> context -> LLM answer
    """

    parser = argparse.ArgumentParser(
        description="Run AdaptiveRAG Stage 2 hybrid RAG with reranking."
    )

    parser.add_argument(
        "--k",
        type=int,
        default=settings.top_k_retrieval,
        help="Number of chunks to retrieve from hybrid search.",
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=settings.top_k_rerank,
        help="Number of chunks to keep after reranking.",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print retrieved and reranked chunks.",
    )

    args = parser.parse_args()

    llm = get_llm()

    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    chain = prompt | llm

    print("AdaptiveRAG Stage 2 is ready.")
    print("Flow: hybrid search -> reranker -> LLM")
    print(f"Hybrid retrieval k: {args.k}")
    print(f"Reranker top_n: {args.top_n}")
    print("Type 'exit' to quit.\n")

    while True:
        question = input("Question: ").strip()

        if question.lower() in {"exit", "quit"}:
            print("Exiting AdaptiveRAG.")
            break

        if not question:
            print("Please enter a question.\n")
            continue

        candidates = hybrid_search(question, k=args.k)

        if args.debug:
            # _print_chunks("HYBRID CANDIDATES", candidates)
            pass

        chunks = rerank(
            query=question,
            chunks=candidates,
            top_n=args.top_n,
        )

        if args.debug:
            _print_chunks("RERANKED CHUNKS", chunks)

        context = "\n\n".join([doc.page_content for doc in chunks])

        response = chain.invoke(
            {
                "context": context,
                "question": question,
            }
        )

        print("\nAnswer:")
        print(response.content)
        print()


if __name__ == "__main__":
    main()
