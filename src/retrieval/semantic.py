import logging
from typing import Any, List, Optional

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from src.client import get_llm
from src.config import settings
from src.ingestion.vectorstore import get_retriever
from src.prompts import HYDE_PROMPT

logger = logging.getLogger(__name__)


def _get_response_text(response: Any) -> str:
    """Extract text content from an LLM response."""
    return getattr(response, "content", str(response)).strip()


def semantic_retrieval(query: str, k: Optional[int] = None) -> List[Document]:
    """
    Retrieve the top-k most relevant chunks for a query using ChromaDB.

    Args:
        query: User query string.
        k: Number of chunks to retrieve. If not provided,
           settings.top_k_retrieval is used.

    Returns:
        List[Document]: Retrieved chunk documents.
    """
    if not query.strip():
        logger.warning("Semantic retrieval skipped because query is empty.")
        return []

    top_k = k if k is not None else settings.top_k_retrieval

    try:
        retriever = get_retriever(top_k)
        results = retriever.invoke(query)

        logger.info("Semantic retrieval returned %s chunks.", len(results))
        return results

    except Exception:
        logger.exception("Semantic retrieval failed. Returning empty results.")
        return []


def generate_hypothetical_document(query: str) -> str:
    """
    Generate a hypothetical document-style answer for HyDE retrieval.

    Args:
        query: Original user query.

    Returns:
        str: Hypothetical answer paragraph.
    """
    if not query.strip():
        logger.info("HyDE generation skipped because query is empty.")
        return query

    try:
        logger.info("Generating hypothetical document for HyDE retrieval.")

        llm = get_llm()
        prompt = ChatPromptTemplate.from_template(HYDE_PROMPT)
        chain = prompt | llm

        response = chain.invoke({"query": query})
        hypothetical_document = _get_response_text(response)

        logger.info("HyDE hypothetical document generated.")
        return hypothetical_document

    except Exception:
        logger.exception(
            "HyDE generation failed. Falling back to original query for semantic retrieval."
        )
        return query


def hyde_semantic_retrieval(query: str, k: Optional[int] = None) -> List[Document]:
    """
    Retrieve relevant chunks using HyDE.

    HyDE flow:
    1. Generate a hypothetical answer/document from the query.
    2. Use that generated text as the semantic search query.
    3. Return retrieved chunks.

    Args:
        query: Original user query.
        k: Number of chunks to retrieve. If not provided,
           settings.top_k_retrieval is used.

    Returns:
        List[Document]: Retrieved chunk documents.
    """
    hypothetical_document = generate_hypothetical_document(query)

    if not hypothetical_document.strip():
        logger.warning(
            "HyDE produced empty text. Falling back to normal semantic retrieval."
        )
        return semantic_retrieval(query, k=k)

    logger.info("Running semantic retrieval using HyDE-generated document.")
    results = semantic_retrieval(hypothetical_document, k=k)

    logger.info("HyDE semantic retrieval returned %s chunks.", len(results))
    return results
