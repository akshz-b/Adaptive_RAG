import logging
from typing import Any, Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.client import get_llm
from src.prompts import CHUNK_GRADER_PROMPT
from src.state import RAGState

logger = logging.getLogger(__name__)


class ChunkGrade(BaseModel):
    """
    Structured output schema for chunk grading.
    """

    grade: Literal["relevant", "ambiguous", "irrelevant"] = Field(
        description="Relevance grade for the retrieved chunk."
    )


def _get_response_text(response: Any) -> str:
    """
    Extract text content from an LLM response.
    """
    return getattr(response, "content", str(response)).strip()


def _normalize_grade(raw_grade: str) -> str:
    """
    Normalize raw LLM output into one of:
    relevant, ambiguous, irrelevant.

    Args:
        raw_grade: Raw text returned by the LLM.

    Returns:
        str: Normalized grade.
    """

    grade = raw_grade.strip().lower()

    if grade == "relevant":
        return "relevant"

    if grade == "ambiguous":
        return "ambiguous"

    if grade == "irrelevant":
        return "irrelevant"

    if "irrelevant" in grade:
        return "irrelevant"

    if "ambiguous" in grade:
        return "ambiguous"

    if "relevant" in grade:
        return "relevant"

    return "ambiguous"


def grade_chunk(query: str, chunk_text: str) -> str:
    """
    Grade a single retrieved chunk against the user query.

    Args:
        query: Original user query.
        chunk_text: Retrieved chunk text.

    Returns:
        str: One of relevant, ambiguous, irrelevant.
    """

    try:
        llm = get_llm()
        prompt = ChatPromptTemplate.from_template(CHUNK_GRADER_PROMPT)

        try:
            structured_llm = llm.with_structured_output(ChunkGrade)
            chain = prompt | structured_llm

            result = chain.invoke(
                {
                    "query": query,
                    "chunk": chunk_text,
                }
            )

            return result.grade

        except Exception:
            logger.warning(
                "Structured chunk grading failed. Falling back to text parsing."
            )

            chain = prompt | llm

            response = chain.invoke(
                {
                    "query": query,
                    "chunk": chunk_text,
                }
            )

            raw_grade = _get_response_text(response)
            return _normalize_grade(raw_grade)

    except Exception:
        logger.exception("Chunk grading failed. Falling back to ambiguous grade.")
        return "ambiguous"


def grader_node(state: RAGState) -> dict:
    """
    Grade retrieved chunks for relevance.

    Reads:
        - query
        - reranked_chunks

    Writes:
        - chunk_grades
    """

    query = state.get("query", "").strip()
    reranked_chunks = state.get("reranked_chunks", [])

    if not query or not reranked_chunks:
        logger.warning("Chunk grading skipped because query or reranked chunks are empty.")

        return {
            "chunk_grades": [],
            "requires_web_search": True,
        }

    chunk_grades: list[dict] = []

    
    for chunk in reranked_chunks:
        chunk_text = chunk.get("page_content", "")
        grade = grade_chunk(query, chunk_text)

        chunk_grades.append(grade)

    logger.info("Chunk grading completed. Grades: %s", chunk_grades)

    return {
        "chunk_grades": chunk_grades,
    }


def decide_after_grading(state: RAGState) -> str:
    """
    Decide the next step after chunk grading.

    Returns:
        - generate
        - web_search
    """

    grades = state.get("chunk_grades", [])

    if not grades:
        logger.info("No chunk grades available. Defaulting to web search.")
        return "web_search"
    
    if all(grade == "relevant" for grade in grades):
        logger.info("Grader route selected: generate")
        return "generate"
    
    if any(grade == "ambiguous" for grade in grades):
        logger.info("Grader route selected: web_search")
        return "web_search"
    
    if all(grade == "irrelevant" for grade in grades):
        logger.info("Grader route selected: web_search")
        return "web_search"
    
    logger.info("Defaulting to web search after grading.")
    return "web_search"
