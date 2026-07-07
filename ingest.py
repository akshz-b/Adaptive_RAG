import argparse
import logging

from src.ingestion.loader import load_and_chunk_pdf
from src.ingestion.vectorstore import store_chunks

logger = logging.getLogger(__name__)


def main() -> None:
    """
    Ingest a PDF into the Chroma vector store.
    """

    parser = argparse.ArgumentParser(
        description="Ingest a PDF into the AdaptiveRAG ChromaDB store."
    )

    parser.add_argument(
        "--file",
        required=True,
        help="Path to the PDF file to ingest.",
    )

    args = parser.parse_args()

    try:
        logger.info("Starting ingestion for file: %s", args.file)

        chunks = load_and_chunk_pdf(args.file)
        logger.info("Created %d chunks from: %s", len(chunks))

        store_chunks(chunks)
        logger.info("Chunks stored successfully in ChromaDB.")

    except Exception:
        logger.exception("PDF ingestion failed.")
        raise


if __name__ == "__main__":
    main()
