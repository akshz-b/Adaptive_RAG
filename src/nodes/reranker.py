import logging

from src.retrieval.reranker import rerank as rerank_documents
from src.state import RAGState
from src.utils.documents import (
    dicts_to_list_of_documents,
    documents_to_list_of_dicts,
)

logger = logging.getLogger(__name__)


def reranker_node(state: RAGState) -> dict:
    """
    LangGraph node that reranks retrieved chunks.

    Reads:
        - query
        - retrieved_chunks

    Writes:
        - reranked_chunks
    """

    query = state.get("query", "").strip()
    retrieved_chunks = state.get("retrieved_chunks", [])

    if not query or not retrieved_chunks:
        logger.warning(
            "Reranker node skipped because query or retrieved chunks are missing."
        )
        return {
            "reranked_chunks": [],
        }

    # logger.info(
    #     "Reranker node started for %s retrieved chunks.",
    #     len(retrieved_chunks),
    # )

    retrieved_documents = dicts_to_list_of_documents(retrieved_chunks)

    reranked_documents = rerank_documents(
        query,
        retrieved_documents,
    )

    reranked_chunks = documents_to_list_of_dicts(reranked_documents)

    # logger.info(
    #     "Reranker node completed with %s reranked chunks.",
    #     len(reranked_chunks),
    # )

    return {
        "reranked_chunks": reranked_chunks,
    }
