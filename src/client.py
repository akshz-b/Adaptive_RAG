import logging
import os
from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFaceEmbeddings,
    HuggingFaceEndpoint,
)

from src.config import settings

logger = logging.getLogger(__name__)

_llm_client: Any | None = None
_embedding_client: Any | None = None


def _set_runtime_env() -> None:
    """
    Set runtime environment variables needed by external clients.
    """
    os.environ["GOOGLE_API_KEY"] = settings.google_api_key
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = settings.huggingface_api_key
    os.environ["TAVILY_API_KEY"] = settings.tavily_api_key
    os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    os.environ["LANGSMITH_TRACING"] = str(settings.langsmith_tracing).lower()

    logger.debug("Runtime environment variables configured for external clients.")


_set_runtime_env()


def get_gemini_llm() -> ChatGoogleGenerativeAI:
    """
    Create and return the Gemini chat LLM client.
    """
    logger.info("Creating Gemini LLM client with model: %s", settings.google_llm_model)
    return ChatGoogleGenerativeAI(
        model=settings.google_llm_model,
        temperature=0.1,
    )


def get_hf_llm() -> ChatHuggingFace:
    """
    Create and return the Hugging Face chat LLM client.
    """
    logger.info(
        "Creating Hugging Face LLM client with model: %s", settings.hf_llm_model
    )

    llm = HuggingFaceEndpoint(
        repo_id=settings.hf_llm_model,
        task="text-generation",
        provider="auto",
        temperature=0.1,
    )

    return ChatHuggingFace(llm=llm).with_config(
        {"metadata": {"model": settings.hf_llm_model}, "tags": ["hf", "llm"]}
    )


def get_llm() -> Any:
    """
    Create and return the cached chat LLM client.
    """
    global _llm_client

    if _llm_client is not None:
        return _llm_client

    provider = settings.llm_provider.strip().lower()
    logger.debug("Selected LLM provider: %s", provider)

    if provider == "google":
        _llm_client = get_gemini_llm()
        return _llm_client

    if provider == "huggingface":
        _llm_client = get_hf_llm()
        return _llm_client

    raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")


def get_hf_embeddings() -> HuggingFaceEmbeddings:
    """
    Create and return the Hugging Face embeddings client.
    """
    logger.info(
        "Creating Hugging Face embeddings client with model: %s",
        settings.hf_embedding_model,
    )
    return HuggingFaceEmbeddings(model_name=settings.hf_embedding_model)


def get_google_embeddings() -> GoogleGenerativeAIEmbeddings:
    """
    Create and return the Google embeddings client.
    """
    logger.info(
        "Creating Google embeddings client with model: %s",
        settings.google_embedding_model,
    )
    return GoogleGenerativeAIEmbeddings(model=settings.google_embedding_model)


def get_embeddings() -> Any:
    """
    Create and return the cached embeddings client.
    """
    global _embedding_client

    if _embedding_client is not None:
        return _embedding_client

    provider = settings.embedding_provider.strip().lower()
    logger.debug("Selected embedding provider: %s", provider)

    if provider == "huggingface":
        _embedding_client = get_hf_embeddings()
        return _embedding_client

    if provider == "google":
        _embedding_client = get_google_embeddings()
        return _embedding_client

    raise ValueError(f"Unsupported embedding provider: {settings.embedding_provider}")
