import logging
from typing import Any, List

from langchain_core.prompts import ChatPromptTemplate

from src.client import get_llm
from src.prompts import MULTI_QUERY_PROMPT, REWRITE_PROMPT
from src.state import RAGState

logger = logging.getLogger(__name__)


def _get_response_text(response: Any) -> str:
    """
    Extract text content from an LLM response.
    """
    return getattr(response, "content", str(response)).strip()


def rewrite_query(query: str) -> str:
    """
    Rewrite a user query into a clearer standalone search query.

    Args:
        query: Original user query.

    Returns:
        str: Rewritten search query.
    """

    if not query.strip():
        logger.info("Query rewriting skipped because query is empty.")
        return query

    try:
        llm = get_llm()

        prompt = ChatPromptTemplate.from_template(REWRITE_PROMPT)
        chain = prompt | llm

        response = chain.invoke({"query": query})
        rewritten_query = _get_response_text(response)

        logger.info("Query rewriting completed.")

        return rewritten_query

    except Exception:
        logger.exception("Query rewriting failed. Falling back to original query.")
        return query


def generate_multi_query(query: str) -> List[str]:
    """
    Generate three search query variants for better retrieval.
    """

    if not query.strip():
        logger.info("Multi-query generation skipped because query is empty.")
        return []

    try:
        llm = get_llm()

        prompt = ChatPromptTemplate.from_template(MULTI_QUERY_PROMPT)

        chain = prompt | llm

        response = chain.invoke({"query": query})
        response_text = _get_response_text(response)

        queries = [line.strip() for line in response_text.splitlines() if line.strip()]
        query_variants = queries[:3]

        logger.info(
            "Generated %s query variants.",
            len(query_variants),
        )

        return query_variants

    except Exception:
        logger.exception(
            "Multi-query generation failed. Continuing without query variants."
        )
        return []


def rewriter_node(state: RAGState) -> dict:
    """
    LangGraph node that rewrites the query and generates multiple query variants.

    Reads:
        - query

    Writes:
        - rewritten_query
        - multiple_queries
    """

    query = state.get("query", "").strip()

    if not query:
        return {
            "rewritten_query": "",
            "multiple_queries": [],
        }

    rewritten_query = rewrite_query(query)
    multiple_queries = generate_multi_query(query)

    return {
        "rewritten_query": rewritten_query,
        "multiple_queries": multiple_queries,
    }
