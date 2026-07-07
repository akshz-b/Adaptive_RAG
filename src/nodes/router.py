import logging
from typing import Any, Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.client import get_llm
from src.prompts import ROUTER_PROMPT
from src.state import RAGState

logger = logging.getLogger(__name__)


class RouteDecision(BaseModel):
    """Structured output schema for query routing."""

    route: Literal["direct", "retrieval", "web"] = Field(
        description="The classification route for the query."
    )


def _get_response_text(response: Any) -> str:
    """Extract text content from an LLM response."""
    return getattr(response, "content", str(response)).strip()


def _normalize_route(raw_route: str) -> str:
    """
    Normalize raw LLM output into one of:
    direct, retrieval, web.
    """
    route = raw_route.strip().lower()

    if route == "direct":
        return "direct"

    if route == "retrieval":
        return "retrieval"

    if route == "web":
        return "web"

    if "retrieval" in route:
        return "retrieval"

    if "web" in route:
        return "web"

    if "direct" in route:
        return "direct"

    return "retrieval"


def classify_query(query: str) -> str:
    """
    Classify a user query into one of:
    direct, retrieval, web.

    Uses structured output first. If unsupported, falls back to text parsing.
    Defaults to retrieval if routing fails.
    """
    if not query.strip():
        return "direct"

    try:
        llm = get_llm()
        prompt = ChatPromptTemplate.from_template(ROUTER_PROMPT)

        try:
            structured_llm = llm.with_structured_output(RouteDecision)
            chain = prompt | structured_llm

            result = chain.invoke(
                {
                    "query": query,
                }
            )

            return result.route

        except Exception:
            logger.warning(
                "Structured router output failed. Falling back to text parsing."
            )

            chain = prompt | llm

            response = chain.invoke(
                {
                    "query": query,
                }
            )

            raw_route = _get_response_text(response)
            return _normalize_route(raw_route)

    except Exception:
        logger.exception("Query routing failed. Defaulting route to retrieval.")
        return "retrieval"


def router_node(state: RAGState) -> dict:
    """
    LangGraph node that classifies the incoming query.

    Reads:
        - query

    Writes:
        - route
    """
    query = state.get("query", "").strip()

    if not query:
        logger.warning("Router received empty query. Routing to direct.")
        return {
            "route": "direct",
        }

    route = classify_query(query)

    logger.info("Router selected route: %s", route)

    return {
        "route": route,
    }


def route_after_router(state: RAGState) -> str:
    """
    Decide which graph path to take after router_node.

    Returns:
        direct, retrieval, or web.
    """
    route = state.get("route", "retrieval")

    if route == "direct":
        return "direct"

    if route == "web":
        return "web"

    return "retrieval"
