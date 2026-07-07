import logging
import re
from typing import List, Optional

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from src.config import settings
from src.ingestion.vectorstore import get_vectorstore

logger = logging.getLogger(__name__)

_bm25_index: Optional[BM25Okapi] = None
_bm25_chunks: List[Document] = []


def _tokenize(text: str) -> List[str]:
    """
    Convert text into lowercase word tokens.

    Args:
        text: Raw text.

    Returns:
        List[str]: Tokenized words.
    """
    return re.findall(r"\b\w+\b", text.lower())


def _build_bm25_index() -> None:
    """
    Load all chunks from ChromaDB and build a BM25 index.
    """
    global _bm25_index, _bm25_chunks

    try:
        logger.info("Building BM25 index from ChromaDB chunks.")

        vectorstore = get_vectorstore()
        stored_data = vectorstore.get(include=["documents", "metadatas"])

        documents = stored_data.get("documents") or []
        metadatas = stored_data.get("metadatas") or []

        _bm25_chunks = []
        tokenized_corpus: List[List[str]] = []

        for text, metadata in zip(documents, metadatas):
            if not text or not text.strip():
                continue

            document = Document(page_content=text, metadata=metadata or {})

            _bm25_chunks.append(document)
            tokenized_corpus.append(_tokenize(text))

        if not tokenized_corpus:
            _bm25_index = None
            logger.warning(
                "BM25 index was not built because no valid chunks were found."
            )
            return

        _bm25_index = BM25Okapi(tokenized_corpus)
        logger.info("BM25 index built successfully with %s chunks.", len(_bm25_chunks))

    except Exception:
        _bm25_index = None
        _bm25_chunks = []
        logger.exception("BM25 index build failed. Keyword search will be unavailable.")


def reset_bm25_index() -> None:
    """
    Clear the cached BM25 index.

    Useful if ChromaDB was updated and BM25 needs to be rebuilt.
    """
    global _bm25_index, _bm25_chunks

    _bm25_index = None
    _bm25_chunks = []

    logger.info("BM25 index cache reset.")


def keyword_search(query: str, k: Optional[int] = None) -> List[Document]:
    """
    Search chunks using BM25 keyword matching.

    Args:
        query: User query string.
        k: Number of chunks to return. If not provided,
           settings.top_k_retrieval is used.

    Returns:
        List[Document]: Top-k keyword-matched chunks.
    """
    global _bm25_index

    if not query.strip():
        logger.info("Keyword search skipped because query is empty.")
        return []

    try:
        if _bm25_index is None:
            _build_bm25_index()

        if _bm25_index is None or not _bm25_chunks:
            logger.warning("Keyword search skipped because BM25 index is unavailable.")
            return []

        top_k = k if k is not None else settings.top_k_retrieval
        tokenized_query = _tokenize(query)

        if not tokenized_query:
            logger.info("Keyword search skipped because query produced no tokens.")
            return []

        scores = _bm25_index.get_scores(tokenized_query)

        ranked_indices = sorted(
            range(len(scores)), key=lambda index: scores[index], reverse=True
        )

        results: List[Document] = []

        for index in ranked_indices[:top_k]:
            score = float(scores[index])

            if score <= 0:
                continue

            chunk = _bm25_chunks[index]
            metadata = dict(chunk.metadata)
            metadata["bm25_score"] = score

            results.append(Document(page_content=chunk.page_content, metadata=metadata))

        logger.info("Keyword search returned %s chunks.", len(results))
        return results

    except Exception:
        logger.exception("Keyword search failed. Returning empty results.")
        return []
