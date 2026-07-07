from typing import List

from langchain_core.documents import Document


def document_to_dict(document: Document) -> dict:
    """
    Convert a single LangChain Document into a serializable dictionary.
    """
    return {
        "page_content": document.page_content,
        "metadata": dict(document.metadata),
    }


def documents_to_list_of_dicts(documents: List[Document]) -> List[dict]:
    """
    Convert a list of LangChain Documents into a list of dictionaries.
    """
    return [document_to_dict(document) for document in documents]


def dict_to_document(chunk: dict) -> Document:
    """
    Convert a dictionary back into a LangChain Document.
    """
    return Document(
        page_content=chunk["page_content"],
        metadata=chunk.get("metadata", {}),
    )


def dicts_to_list_of_documents(chunks: List[dict]) -> List[Document]:
    """
    Convert a list of dictionaries back into LangChain Documents.
    """
    return [dict_to_document(chunk) for chunk in chunks]
