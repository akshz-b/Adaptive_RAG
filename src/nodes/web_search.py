import logging

from src.state import RAGState
from src.tools import web_search

logger = logging.getLogger(__name__)


def web_search_node(state: RAGState) -> dict:
    """
    LangGraph node that performs web search fallback.

    Reads:
        - query

    Writes:
        - web_results

    Args:
        state: LangGraph state dictionary.

    Returns:
        dict: State update containing web_results.
    """
    query = state.get("query", "").strip()

    if not query:
        logger.warning("Web search node skipped because query is empty.")
        return {
            "web_results": [],
        }

    results = web_search(query)

    return {
        "web_results": results,
    }
