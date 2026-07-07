import logging
from typing import Any, List

from langchain_core.prompts import ChatPromptTemplate

from src.client import get_llm
from src.prompts import GENERATOR_PROMPT
from src.state import RAGState

logger = logging.getLogger(__name__)


def _get_response_text(response: Any) -> str:
    """
    Extract text content from an LLM response.
    """
    return getattr(response, "content", str(response)).strip()


def _format_document_context(chunks: List[dict]) -> str:
    """
    Format reranked document chunks into one context string.

    Args:
        chunks: List of reranked chunk dictionaries.

    Returns:
        str: Formatted document context.
    """

    formatted_chunks: List[str] = []

    for index, chunk in enumerate(chunks, start=1):
        metadata = chunk.get("metadata", {})
        page_content = chunk.get("page_content", "")

        source = metadata.get("source", "unknown source")
        page_number = metadata.get("page_number", "unknown page")
        chunk_id = metadata.get("chunk_id", "unknown chunk")

        formatted_chunks.append(
            f"[DOC {index}]\n"
            f"Source: {source}\n"
            f"Page: {page_number}\n"
            f"Chunk ID: {chunk_id}\n"
            f"Content:\n{page_content}"
        )

    return "\n\n".join(formatted_chunks)


def _format_web_context(web_results: List[dict]) -> str:
    """
    Format web search results into one context string.

    Args:
        web_results: List of normalized web result dictionaries.

    Returns:
        str: Formatted web context.
    """

    formatted_results: List[str] = []

    for index, result in enumerate(web_results, start=1):
        title = result.get("title", "Untitled")
        url = result.get("url", "")
        content = result.get("content", "")

        formatted_results.append(
            f"[WEB {index}]\nTitle: {title}\nURL: {url}\nContent:\n{content}"
        )

    return "\n\n".join(formatted_results)


def _build_citations(
    chunks: List[dict],
    web_results: List[dict],
) -> List[str]:
    """
    Build simple citations from document chunks and web results.
    """

    citations: List[str] = []

    for chunk in chunks:
        metadata = chunk.get("metadata", {})
        source = metadata.get("source")
        page_number = metadata.get("page_number")
        chunk_id = metadata.get("chunk_id")

        if source:
            citations.append(f"{source}, page {page_number}, chunk {chunk_id}")

    for result in web_results:
        title = result.get("title")
        url = result.get("url")

        if title or url:
            citations.append(f"{title} - {url}")

    return citations


def generator_node(state: RAGState) -> dict:
    """
    LangGraph node that generates the final answer.

    Reads:
        - query
        - reranked_chunks
        - web_results

    Writes:
        - final_context
        - answer
        - citations
    """

    query = state.get("query", "").strip()
    reranked_chunks = state.get("reranked_chunks", [])
    web_results = state.get("web_results", [])

    document_context = _format_document_context(reranked_chunks)
    web_context = _format_web_context(web_results)

    context_parts: List[str] = []

    if document_context.strip():
        context_parts.append(document_context)

    if web_context.strip():
        context_parts.append(web_context)

    final_context = "\n\n".join(context_parts)
    citations = _build_citations(reranked_chunks, web_results)

    if not query:
        logger.warning("Answer generation skipped because query is empty.")

        return {
            "final_context": final_context,
            "answer": "I don't know.",
            "citations": citations,
        }

    if not final_context.strip():
        logger.warning("Answer generation skipped because final context is empty.")

        return {
            "final_context": "",
            "answer": "I don't know.",
            "citations": [],
        }

    try:
        logger.info(
            "Generating answer using %s document chunks and %s web results.",
            len(reranked_chunks),
            len(web_results),
        )

        llm = get_llm()
        prompt = ChatPromptTemplate.from_template(GENERATOR_PROMPT)
        chain = prompt | llm

        response = chain.invoke(
            {
                "context": final_context,
                "query": query,
            }
        )

        answer = _get_response_text(response)

        logger.info(
            "Answer generated successfully with %s citations.",
            len(citations),
        )

        return {
            "final_context": final_context,
            "answer": answer,
            "citations": citations,
        }

    except Exception:
        logger.exception("Answer generation failed. Returning safe fallback answer.")

        return {
            "final_context": final_context,
            "answer": (
                "I could not generate an answer due to an internal LLM error. "
                "Please try again later."
            ),
            "citations": citations,
        }
