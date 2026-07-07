import logging
from typing import Any, List, Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.client import get_llm
from src.prompts import CRITIC_PROMPT
from src.state import RAGState

logger = logging.getLogger(__name__)

MAX_CRITIC_RETRIES = 1


class CriticResult(BaseModel):
    """
    Structured output schema for answer grounding criticism.
    """

    decision: Literal["grounded", "not_grounded"] = Field(
        description="Whether the answer is grounded in the provided context."
    )

    unsupported_claims: List[str] = Field(
        default_factory=list,
        description="Claims from the answer that are not supported by the context.",
    )


def _get_response_text(response: Any) -> str:
    """
    Extract text content from an LLM response.
    """
    return getattr(response, "content", str(response)).strip()


def _normalize_decision(raw_decision: str) -> str:
    """
    Normalize raw LLM output into grounded or not_grounded.
    """
    decision = raw_decision.strip().lower()

    if decision == "grounded":
        return "grounded"

    if decision == "not_grounded":
        return "not_grounded"

    if "not_grounded" in decision or "not grounded" in decision:
        return "not_grounded"

    if "grounded" in decision:
        return "grounded"

    return "not_grounded"


def critique_answer(query: str, context: str, answer: str) -> CriticResult:
    """
    Critique whether the generated answer is grounded in the provided context.
    """

    try:
        llm = get_llm()
        prompt = ChatPromptTemplate.from_template(CRITIC_PROMPT)

        try:
            structured_llm = llm.with_structured_output(CriticResult)
            chain = prompt | structured_llm

            result = chain.invoke(
                {
                    "query": query,
                    "context": context,
                    "answer": answer,
                }
            )

            return result

        except Exception:
            logger.warning(
                "Structured critic output failed. Falling back to text parsing."
            )

            chain = prompt | llm

            response = chain.invoke(
                {
                    "query": query,
                    "context": context,
                    "answer": answer,
                }
            )

            raw_text = _get_response_text(response)
            decision = _normalize_decision(raw_text)

            return CriticResult(
                decision=decision,
                unsupported_claims=[],
            )

    except Exception:
        logger.exception(
            "Answer critique failed. Falling back to not_grounded decision."
        )

        return CriticResult(
            decision="not_grounded",
            unsupported_claims=["Critic failed to verify answer grounding."],
        )


def critic_node(state: RAGState) -> dict:
    """
    LangGraph node that checks whether the generated answer is grounded.

    Reads:
        - query
        - final_context
        - answer
        - retry_count

    Writes:
        - critic_decision
        - unsupported_claims
        - retry_count
    """

    query = state.get("query", "").strip()
    context = state.get("final_context", "").strip()
    answer = state.get("answer", "").strip()
    retry_count = state.get("retry_count", 0)

    if not query or not context or not answer:
        logger.warning(
            "Critic marked answer not grounded because required fields are missing."
        )

        return {
            "critic_decision": "not_grounded",
            "unsupported_claims": ["Missing query, context, or answer."],
            "retry_count": retry_count + 1,
        }

    result = critique_answer(
        query=query,
        context=context,
        answer=answer,
    )

    updated_retry_count = retry_count

    if result.decision == "not_grounded":
        updated_retry_count += 1

    logger.info(
        "Critic decision: %s | retry_count=%s | unsupported_claims=%s",
        result.decision,
        updated_retry_count,
        len(result.unsupported_claims),
    )

    return {
        "critic_decision": result.decision,
        "unsupported_claims": result.unsupported_claims,
        "retry_count": updated_retry_count,
    }


def route_after_critic(state: RAGState) -> str:
    """
    Decide what to do after critic evaluation.

    Returns:
        - end
        - web_search
        - finalize_unverified
    """

    decision = state.get("critic_decision", "not_grounded")
    retry_count = state.get("retry_count", 0)

    if decision == "grounded":
        logger.info("Critic route selected: end")
        return "end"

    if retry_count <= MAX_CRITIC_RETRIES:
        logger.info("Critic route selected: web_search")
        return "web_search"

    logger.info("Critic route selected: finalize_unverified")
    return "finalize_unverified"


def finalizer_node(state: RAGState) -> dict:
    """
    Finalize the graph with a safe fallback when max critic retries are exceeded.

    Returns:
        - answer
    """

    unsupported_claims = state.get("unsupported_claims", [])

    logger.warning(
        "Finalizer triggered after max critic retries. Unsupported claims: %s",
        len(unsupported_claims),
    )

    if unsupported_claims:
        claims_text = "\n".join(f"- {claim}" for claim in unsupported_claims)
    else:
        claims_text = "- Insufficient context to validate the answer."

    return {
        "answer": (
            "I cannot provide this answer as it contains unsupported claims:\n\n"
            f"{claims_text}\n\n"
            "Please refine your question or I can search for more specific information."
        )
    }
