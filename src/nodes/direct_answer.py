import logging
from typing import Any

from langchain_core.prompts import ChatPromptTemplate

from src.client import get_llm
from src.prompts import DIRECT_ANSWER_PROMPT
from src.state import RAGState

logger = logging.getLogger(__name__)


def _get_response_text(response: Any) -> str:
    """Extract text content from an LLM response."""
    return getattr(response, "content", str(response)).strip()


def direct_answer_node(state: RAGState) -> dict:
    """
    LangGraph node that answers simple/direct questions without retrieval.

    Reads:
        - query

    Writes:
        - answer
        - final_context
        - citations
    """
    query = state.get("query", "").strip()

    if not query:
        logger.warning("Direct answer skipped because query is empty.")
        return {
            "answer": "I don't know.",
            "final_context": "",
            "citations": [],
        }

    try:
        logger.info("Generating direct answer without retrieval.")

        llm = get_llm()
        prompt = ChatPromptTemplate.from_template(DIRECT_ANSWER_PROMPT)
        chain = prompt | llm

        response = chain.invoke(
            {
                "query": query,
            }
        )

        answer = _get_response_text(response)

        logger.info("Direct answer generated successfully.")

        return {
            "answer": answer,
            "final_context": "",
            "citations": [],
        }

    except Exception:
        logger.exception("Direct answer generation failed.")

        return {
            "answer": (
                "I could not generate a direct answer due to an internal LLM error. "
                "Please try again later."
            ),
            "final_context": "",
            "citations": [],
        }
