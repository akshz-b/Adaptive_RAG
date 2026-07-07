import logging
from typing import List

from src.client import get_embeddings


logger = logging.getLogger(__name__)

_embedding_client = get_embeddings()


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Embed a list of document texts/chunks.

    Args:
        texts: List of text chunks to embed.

    Returns:
        List[List[float]]: One embedding vector per input text.
    """
    if not texts:
        logger.warning("Document embedding skipped because no texts were provided.")
        return []

    try:
        logger.info("Embedding %s document texts.", len(texts))

        embedding_client = _embedding_client
        embeddings = embedding_client.embed_documents(texts)

        logger.info("Generated %s document embeddings.", len(embeddings))

        return embeddings

    except Exception:
        logger.exception("Document embedding generation failed.")
        raise


def embed_query(query: str) -> List[float]:
    """
    Embed a single query string.

    Args:
        query: User query text.

    Returns:
        List[float]: Embedding vector for the query.
    """
    if not query.strip():
        raise ValueError("Query cannot be empty for embedding.")

    try:
        logger.debug("Embedding query text.")

        embedding_client = _embedding_client
        return embedding_client.embed_query(query)

    except Exception:
        logger.exception("Query embedding generation failed.")
        raise
