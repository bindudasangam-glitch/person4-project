"""
Embedding generation service.

Wraps a Sentence Transformers model behind a small, reusable interface so
the rest of the application never interacts with the underlying ML
library directly. The embedding model, batch size, and inference device
are all configurable via application settings (environment variables),
so swapping models does not require any code changes.

Dependencies:
    pip install sentence-transformers
"""

from __future__ import annotations

import logging
from threading import Lock
from typing import Optional

from sentence_transformers import SentenceTransformer

from app.core.config import Settings, get_settings
from app.core.exceptions import EmbeddingGenerationError, EmbeddingModelLoadError
from app.models.chunk import Chunk

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Generates dense vector embeddings for text using a configurable
    Sentence Transformers model.

    The underlying model is loaded lazily (on first use) and cached for
    the lifetime of the service instance, since model loading is
    comparatively expensive and the same instance is intended to be
    reused across many embedding calls (e.g. injected as a singleton
    dependency in the FastAPI application).
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        """
        Args:
            settings: Application settings providing the embedding model
                name, batch size, and device. Defaults to the cached
                process-wide settings instance.
        """
        self._settings = settings or get_settings()
        self._model: Optional[SentenceTransformer] = None
        self._load_lock = Lock()

    @property
    def model_name(self) -> str:
        """The configured embedding model's name or path."""
        return self._settings.embedding_model_name

    def _get_model(self) -> SentenceTransformer:
        """
        Return the loaded Sentence Transformers model, loading it on
        first access. Thread-safe: concurrent callers will not trigger
        duplicate model loads.

        Returns:
            The loaded :class:`SentenceTransformer` instance.

        Raises:
            EmbeddingModelLoadError: If the model fails to load.
        """
        if self._model is not None:
            return self._model

        with self._load_lock:
            if self._model is not None:
                return self._model

            try:
                logger.info(
                    "Loading embedding model '%s' on device '%s'...",
                    self._settings.embedding_model_name,
                    self._settings.embedding_device,
                )
                self._model = SentenceTransformer(
                    self._settings.embedding_model_name,
                    device=self._settings.embedding_device,
                )
                logger.info("Embedding model '%s' loaded successfully.", self._settings.embedding_model_name)
            except Exception as exc:  # noqa: BLE001 - surface any load failure uniformly
                raise EmbeddingModelLoadError(self._settings.embedding_model_name, str(exc)) from exc

            return self._model

    def embed_text(self, text: str) -> list[float]:
        """
        Generate an embedding vector for a single piece of text.

        Args:
            text: The text to embed. Must be non-empty.

        Returns:
            A dense embedding vector as a list of floats.

        Raises:
            EmbeddingGenerationError: If ``text`` is empty or embedding
                generation fails.
        """
        if not text or not text.strip():
            raise EmbeddingGenerationError("cannot embed empty or whitespace-only text.")

        return self.embed_texts([text])[0]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embedding vectors for a batch of texts, respecting the
        configured embedding batch size.

        Args:
            texts: A list of non-empty text strings to embed.

        Returns:
            A list of embedding vectors, in the same order as ``texts``.

        Raises:
            EmbeddingGenerationError: If ``texts`` is empty, contains any
                empty/whitespace-only entries, or embedding generation fails.
        """
        if not texts:
            raise EmbeddingGenerationError("cannot embed an empty list of texts.")

        blank_indices = [index for index, text in enumerate(texts) if not text or not text.strip()]
        if blank_indices:
            raise EmbeddingGenerationError(
                f"cannot embed empty or whitespace-only text at indices {blank_indices}."
            )

        model = self._get_model()

        try:
            embeddings = model.encode(
                texts,
                batch_size=self._settings.embedding_batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        except Exception as exc:  # noqa: BLE001 - surface any inference failure uniformly
            raise EmbeddingGenerationError(str(exc), batch_size=self._settings.embedding_batch_size) from exc

        logger.debug("Generated %d embeddings using model '%s'.", len(texts), self.model_name)
        return [vector.tolist() for vector in embeddings]

    def embed_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """
        Generate and attach embeddings to a list of chunks, returning new
        :class:`Chunk` instances with the ``embedding`` field populated.

        Args:
            chunks: Chunks to embed. Chunks that already have an
                embedding are skipped (their vectors are reused as-is)
                to avoid redundant computation.

        Returns:
            A new list of :class:`Chunk` instances, in the same order as
            the input, with embeddings populated.

        Raises:
            EmbeddingGenerationError: If ``chunks`` is empty or embedding
                generation fails for any chunk requiring embedding.
        """
        if not chunks:
            raise EmbeddingGenerationError("cannot embed an empty list of chunks.")

        indices_needing_embedding = [
            index for index, chunk in enumerate(chunks) if not chunk.has_embedding()
        ]

        if not indices_needing_embedding:
            logger.debug("All %d chunks already had embeddings; skipping generation.", len(chunks))
            return list(chunks)

        texts_to_embed = [chunks[index].text for index in indices_needing_embedding]
        new_embeddings = self.embed_texts(texts_to_embed)

        embedded_chunks = list(chunks)
        for index, embedding in zip(indices_needing_embedding, new_embeddings):
            embedded_chunks[index] = chunks[index].model_copy(update={"embedding": embedding})

        logger.info(
            "Embedded %d of %d chunks (remainder already had embeddings).",
            len(indices_needing_embedding),
            len(chunks),
        )
        return embedded_chunks