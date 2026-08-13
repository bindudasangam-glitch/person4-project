from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application configuration settings.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )

    APP_NAME: str = "Hallucination Detection System"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True

    API_PREFIX: str = "/api/v1"

    # --------------------------------------------------------------------- #
    # OpenAI / Voice Feature
    # --------------------------------------------------------------------- #
    openai_api_key: str | None = None

    # --------------------------------------------------------------------- #
    # Document ingestion (Person 2)
    # --------------------------------------------------------------------- #
    UPLOAD_DIR: str = "data/uploads"
    PROCESSED_DIR: str = "data/processed"

    max_upload_size_mb: int = 20

    # --------------------------------------------------------------------- #
    # Chunking (Person 2)
    # --------------------------------------------------------------------- #
    chunk_size: int = 500
    chunk_overlap: int = 50

    # --------------------------------------------------------------------- #
    # Embedding (Person 2)
    # --------------------------------------------------------------------- #
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: str = "cpu"
    embedding_batch_size: int = 32

    # --------------------------------------------------------------------- #
    # Vector store / ChromaDB (Person 2)
    # --------------------------------------------------------------------- #
    CHROMA_PERSIST_DIR: str = "data/chroma"
    chroma_collection_name: str = "hallucination_detection_evidence"

    # --------------------------------------------------------------------- #
    # Retrieval (Person 2)
    # --------------------------------------------------------------------- #
    retrieval_top_k: int = 5
    similarity_score_threshold: float = 0.35

    # --------------------------------------------------------------------- #
    # Verification (Person 2)
    # --------------------------------------------------------------------- #
    verification_support_threshold: float = 0.6
    verification_contradiction_threshold: float = 0.6

    def resolved_upload_dir(self) -> Path:
        """Return the absolute, filesystem-ready path for uploaded documents."""
        return Path(self.UPLOAD_DIR).resolve()

    def resolved_processed_dir(self) -> Path:
        """Return the absolute, filesystem-ready path for processed artifacts."""
        return Path(self.PROCESSED_DIR).resolve()

    def resolved_chroma_persist_dir(self) -> Path:
        """Return the absolute, filesystem-ready path for ChromaDB persistence."""
        return Path(self.CHROMA_PERSIST_DIR).resolve()


settings = Settings()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the process-wide cached Settings instance.

    Follows the same lru_cache-backed singleton pattern already used
    throughout the codebase so that all services share one settings
    instance per worker process.
    """
    return settings