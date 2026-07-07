import logging
import os
from typing import Any, Dict, List, Optional

from langchain_tavily import TavilySearch

from src.config import settings

logger = logging.getLogger(__name__)

_tavily_search_tool: Optional[TavilySearch] = None


def _get_tavily_search_tool() -> TavilySearch:
    """
    Lazily initialize and return Tavily search tool.
    """
    global _tavily_search_tool

    if _tavily_search_tool is None:
        if settings.tavily_api_key:
            os.environ["TAVILY_API_KEY"] = settings.tavily_api_key

        _tavily_search_tool = TavilySearch(
            max_results=3,
            topic="general",
            include_answer=True,
            include_raw_content=False,
            include_images=False,
            search_depth="basic",
        )

    return _tavily_search_tool


def _normalize_tavily_results(raw_response: Any) -> List[Dict[str, str]]:
    """
    Convert Tavily's raw response into the standard web result format.

    Output:
        List of dictionaries with title, url, and content.
    """
    normalized_results: List[Dict[str, str]] = []

    if isinstance(raw_response, dict):
        if raw_response.get("answer"):
            normalized_results.append(
                {
                    "title": "Tavily Answer",
                    "url": "",
                    "content": str(raw_response.get("answer", "")),
                }
            )

        results = raw_response.get("results", [])

    elif isinstance(raw_response, list):
        results = raw_response

    else:
        return normalized_results

    for result in results:
        if not isinstance(result, dict):
            continue

        normalized_results.append(
            {
                "title": str(result.get("title", "")),
                "url": str(result.get("url", "")),
                "content": str(
                    result.get("content") or result.get("raw_content") or ""
                ),
            }
        )

    return normalized_results


def web_search(query: str) -> List[Dict[str, str]]:
    """
    Search the web using Tavily.

    Args:
        query: Search query.

    Returns:
        List[Dict[str, str]]: Normalized web search results.
    """
    if not query.strip():
        logger.warning("Web search skipped because query is empty.")
        return []

    try:
        logger.info("Running Tavily web search.")

        tavily_search_tool = _get_tavily_search_tool()

        raw_response = tavily_search_tool.invoke(
            {
                "query": query,
            }
        )

        results = _normalize_tavily_results(raw_response)

        logger.info("Tavily web search returned %s results.", len(results))
        return results

    except Exception:
        logger.exception("Tavily web search failed. Returning empty web results.")
        return []
