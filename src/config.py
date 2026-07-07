from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Central application settings loaded from .env.

    All configurable values for the AdaptiveRAG project should live here.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API keys
    huggingface_api_key: str = Field(alias="HUGGINGFACEHUB_API_TOKEN")
    tavily_api_key: str = Field(alias="TAVILY_API_KEY")
    langsmith_api_key: str = Field(alias="LANGSMITH_API_KEY")
    google_api_key: str = Field(alias="GOOGLE_API_KEY")

    # LangSmith
    langsmith_project: str = Field(default="adaptive-rag", alias="LANGSMITH_PROJECT")
    langsmith_tracing: bool = Field(default=True, alias="LANGSMITH_TRACING")

    # Models
    llm_provider: str = Field(default="huggingface", alias="LLM_PROVIDER")
    embedding_provider: str = Field(default="huggingface", alias="EMBEDDING_PROVIDER")
    hf_llm_model: str = Field(default="Qwen/Qwen2.5-72B-Instruct", alias="HF_LLM_MODEL")
    hf_embedding_model: str = Field(
        default="BAAI/bge-base-en-v1.5", alias="HF_EMBEDDING_MODEL"
    )
    reranker_model: str = Field(
        default="BAAI/bge-reranker-base", alias="RERANKER_MODEL"
    )
    google_embedding_model: str = Field(
        default="gemini-embedding-001", alias="GOOGLE_EMBEDDING_MODEL"
    )
    google_llm_model: str = Field(default="gemini-2.5-flash", alias="GOOGLE_LLM_MODEL")

    # Storage
    chroma_persist_dir: str = "./data/chroma"

    # Chunking
    chunk_size: int = 2000
    chunk_overlap: int = 300

    # Retrieval
    top_k_retrieval: int = 10
    top_k_rerank: int = 5

    @property
    def chroma_path(self) -> Path:
        """Return Chroma persist directory as a Path object."""
        return Path(self.chroma_persist_dir)


settings = Settings()
