import logging
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import settings

logger = logging.getLogger(__name__)


def _load_pdf(file_path: str) -> List[Document]:
    """
    Load a PDF file page by page and return LangChain Document objects.

    Args:
        file_path: Path to the PDF file.

    Returns:
        List[Document]: One Document per page.
    """

    pdf_path = Path(file_path)

    if not pdf_path.exists():
        logger.error("PDF file not found: %s", pdf_path)
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    try:
        logger.info("Loading PDF: %s", pdf_path)

        loader = PyPDFLoader(
            file_path=str(pdf_path),
            mode="page",
        )

        documents = loader.load()

        logger.info("Loaded %s pages from PDF.", len(documents))

        return documents

    except Exception:
        logger.exception("Failed to load PDF: %s", pdf_path)
        raise


def _chunk_documents(documents: List[Document]) -> List[Document]:
    """
    Split page-level documents into smaller chunks.

    Args:
        documents: List of page-level Document objects.

    Returns:
        List[Document]: Chunked Document objects.
    """

    if not documents:
        logger.warning("Document chunking skipped because no documents were provided.")
        return []

    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

        chunks = text_splitter.split_documents(documents)

        logger.info(
            "Split %s pages into %s chunks.",
            len(documents),
            len(chunks),
        )

        return chunks

    except Exception:
        logger.exception("Failed to split documents into chunks.")
        raise


def _add_chunk_metadata(
    chunks: List[Document],
    file_path: str,
) -> List[Document]:
    """
    Normalize metadata for each chunk.

    Ensures that every chunk has:
    - source
    - page_number
    - chunk_index

    Args:
        chunks: List of chunked Document objects.
        file_path: Original PDF file path.

    Returns:
        List[Document]: Chunked documents with normalized metadata.
    """

    if not chunks:
        logger.warning(
            "Metadata normalization skipped because no chunks were provided."
        )
        return []

    try:
        normalized_chunks: List[Document] = []
        file_name = Path(file_path).name

        for chunk_index, chunk in enumerate(chunks):
            metadata = dict(chunk.metadata) if chunk.metadata else {}

            raw_page = metadata.get("page", metadata.get("page_number"))

            if isinstance(raw_page, int):
                page_number = raw_page + 1
            else:
                page_number = raw_page

            metadata["source"] = file_name
            metadata["page_number"] = page_number
            metadata["chunk_index"] = chunk_index

            normalized_chunks.append(
                Document(
                    page_content=chunk.page_content,
                    metadata=metadata,
                )
            )

        logger.info(
            "Normalized metadata for %s chunks.",
            len(normalized_chunks),
        )

        return normalized_chunks

    except Exception:
        logger.exception("Failed to normalize chunk metadata.")
        raise


def load_and_chunk_pdf(file_path: str) -> List[Document]:
    """
    Load a PDF, split it into chunks, and attach normalized metadata.

    Args:
        file_path: Path to the PDF file.

    Returns:
        List[Document]: Final chunked documents ready for embedding.
    """

    try:
        page_documents = _load_pdf(file_path)
        chunked_documents = _chunk_documents(page_documents)
        final_chunks = _add_chunk_metadata(
            chunked_documents,
            file_path,
        )

        logger.info(
            "PDF ingestion completed with %s final chunks.",
            len(final_chunks),
        )

        return final_chunks

    except Exception:
        logger.exception(
            "PDF ingestion failed for file: %s",
            file_path,
        )
        raise
