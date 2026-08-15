from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application configuration settings.

    The configuration is intentionally lightweight so that importing the
    settings module does not initialize any heavy ML or vector-store
    dependencies.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )

    APP_NAME: str = "Hallucination Detection System"
    APP_VERSION: str = "1.0.0"

    # --------------------------------------------------------------------- #
    # Production / deployment
    # --------------------------------------------------------------------- #

    DEBUG: bool = False

    API_PREFIX: str = "/api/v1"

    # --------------------------------------------------------------------- #
    # OpenAI / Voice Feature
    # --------------------------------------------------------------------- #

    openai_api_key: str | None = None

    # --------------------------------------------------------------------- #
    # Document ingestion
    # --------------------------------------------------------------------- #

    UPLOAD_DIR: str = "data/uploads"
    PROCESSED_DIR: str = "data/processed"

    max_upload_size_mb: int = 20

    # --------------------------------------------------------------------- #
    # Chunking
    # --------------------------------------------------------------------- #

    chunk_size: int = 500
    chunk_overlap: int = 50

    # --------------------------------------------------------------------- #
    # Embedding
    #
    # IMPORTANT:
    # We use FastEmbed/ONNX instead of SentenceTransformers/PyTorch.
    # This substantially reduces the runtime memory footprint on Render.
    # --------------------------------------------------------------------- #

    embedding_model_name: str = "BAAI/bge-small-en-v1.5"

    # Kept for compatibility with the existing application configuration.
    # FastEmbed uses ONNX Runtime and does not require a PyTorch device.
    embedding_device: str = "cpu"

    # Keep this deliberately small because Render Free has a 512 MB
    # memory limit.
    embedding_batch_size: int = 2

    # --------------------------------------------------------------------- #
    # Vector store / ChromaDB
    # --------------------------------------------------------------------- #

    CHROMA_PERSIST_DIR: str = "data/chroma"
    chroma_collection_name: str = "hallucination_detection_evidence"

    # --------------------------------------------------------------------- #
    # Retrieval
    # --------------------------------------------------------------------- #

    retrieval_top_k: int = 5
    similarity_score_threshold: float = 0.35

    # --------------------------------------------------------------------- #
    # Verification
    # --------------------------------------------------------------------- #

    verification_support_threshold: float = 0.6
    verification_contradiction_threshold: float = 0.6

    # --------------------------------------------------------------------- #
    # Resolved filesystem paths
    # --------------------------------------------------------------------- #

    def resolved_upload_dir(self) -> Path:
        """
        Return the absolute filesystem-ready path for uploaded documents.
        """
        return Path(self.UPLOAD_DIR).resolve()

    def resolved_processed_dir(self) -> Path:
        """
        Return the absolute filesystem-ready path for processed artifacts.
        """
        return Path(self.PROCESSED_DIR).resolve()

    def resolved_chroma_persist_dir(self) -> Path:
        """
        Return the absolute filesystem-ready path for persistent ChromaDB.
        """
        return Path(self.CHROMA_PERSIST_DIR).resolve()


# ------------------------------------------------------------------------- #
# Process-wide settings instance
# ------------------------------------------------------------------------- #

settings = Settings()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the process-wide cached Settings instance.

    All application services use the same Settings instance within a
    worker process.
    """
    return settings