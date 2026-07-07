import logging
from typing import List

from langchain_core.documents import Document

from src.nodes.rewriter import generate_multi_query, rewrite_query
from src.retrieval.hybrid import hybrid_search
from src.retrieval.reranker import rerank
from src.retrieval.semantic import hyde_semantic_retrieval
from src.state import RAGState
from src.utils.documents import documents_to_list_of_dicts

logger = logging.getLogger(__name__)


def retrieve_candidates(
    query: str,
    rewritten_query: str,
    multiple_queries: List[str],
) -> List[Document]:
    """
    Retrieve candidate chunks using rewritten query, multi-query,
    hybrid search, and HyDE.
    """

    search_queries = [rewritten_query] + multiple_queries

    clean_search_queries: List[str] = []
    seen_queries = set()

    for search_query in search_queries:
        cleaned_query = search_query.strip()

        if cleaned_query and cleaned_query not in seen_queries:
            clean_search_queries.append(cleaned_query)
            seen_queries.add(cleaned_query)

    all_chunks: List[Document] = []

    for search_query in clean_search_queries:
        chunks = hybrid_search(search_query)
        all_chunks.extend(chunks)

    hyde_chunks = hyde_semantic_retrieval(query)
    all_chunks.extend(hyde_chunks)

    unique_chunks: dict[str, Document] = {}

    for chunk in all_chunks:
        chunk_id = chunk.metadata.get("chunk_id")
        unique_chunks[chunk_id] = chunk

    deduplicated_chunks = list(unique_chunks.values())

    logger.info(
        "Retriever produced %s unique candidate chunks from %s search queries.",
        len(deduplicated_chunks),
        len(clean_search_queries),
    )

    return deduplicated_chunks


def retrieve(query: str) -> List[Document]:
    """
    Full retrieval pipeline.

    Args:
        query: Original user query.

    Returns:
        List[Document]: Final reranked chunks.
    """

    rewritten_query = rewrite_query(query)
    multiple_queries = generate_multi_query(query)

    deduplicated_chunks = retrieve_candidates(
        query,
        rewritten_query,
        multiple_queries,
    )

    reranked_chunks = rerank(
        query=query,
        chunks=deduplicated_chunks,
    )

    return reranked_chunks


def retriever_node(state: RAGState) -> dict:
    """
    LangGraph node that retrieves candidate chunks.

    Reads:
        - query
        - rewritten_query
        - multiple_queries

    Writes:
        - retrieved_chunks
    """

    query = state.get("query", "").strip()
    rewritten_query = state.get("rewritten_query", "").strip()
    multiple_queries = state.get("multiple_queries", [])

    if not query:
        logger.warning("Retriever node skipped because query is empty.")
        return {"retrieved_chunks": []}

    candidates = retrieve_candidates(
        query,
        rewritten_query,
        multiple_queries,
    )

    return {
        "retrieved_chunks": documents_to_list_of_dicts(candidates),
    }
