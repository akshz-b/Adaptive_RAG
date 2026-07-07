import logging
from typing import List, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.client import get_embeddings
from src.config import settings

logger = logging.getLogger(__name__)

COLLECTION_NAME = "adaptive_rag"
_vectorstore: Optional[Chroma] = None


def get_vectorstore() -> Chroma:
    """
    Initialize and return the project's Chroma vector store.

    Returns:
        Chroma: Configured Chroma vector store instance.
    """
    global _vectorstore

    if _vectorstore is None:
        try:
            logger.info(
                "Initializing Chroma vector store: %s",
                COLLECTION_NAME,
            )

            _vectorstore = Chroma(
                collection_name=COLLECTION_NAME,
                embedding_function=get_embeddings(),
                persist_directory=settings.chroma_persist_dir,
            )

            logger.info("Chroma vector store initialized successfully.")

        except Exception:
            logger.exception("Failed to initialize Chroma vector store.")
            raise

    return _vectorstore


def store_chunks(chunks: List[Document]) -> None:
    """
    Store chunked documents in ChromaDB.

    Args:
        chunks: List of LangChain Document chunks to store.
    """

    if not chunks:
        logger.warning("No chunks provided for ChromaDB storage.")
        return

    try:
        logger.info("Storing %s chunks in ChromaDB.", len(chunks))

        vectorstore = get_vectorstore()

        ids = []

        for chunk in chunks:
            chunk_id = (
                f"{chunk.metadata['source']}"
                f"_p{chunk.metadata['page_number']}"
                f"_{chunk.metadata['chunk_index']}"
            )

            chunk.metadata["chunk_id"] = chunk_id
            ids.append(chunk_id)

        vectorstore.add_documents(documents=chunks, ids=ids)

        logger.info("Stored %s chunks in ChromaDB.", len(chunks))

    except Exception:
        logger.exception("Failed to store chunks in ChromaDB.")
        raise


def get_retriever(k: Optional[int] = None):
    """
    Return a retriever for fetching top-k similar chunks.

    Args:
        k: Number of chunks to retrieve. If not provided,
           settings.top_k_retrieval is used.

    Returns:
        BaseRetriever: Chroma retriever configured with top-k search.
    """

    try:
        vectorstore = get_vectorstore()
        top_k = k if k is not None else settings.top_k_retrieval

        logger.debug(
            "Creating Chroma retriever with top_k=%s.",
            top_k,
        )

        return vectorstore.as_retriever(search_kwargs={"k": top_k})

    except Exception:
        logger.exception("Failed to create Chroma retriever.")
        raise
