"""
Lightweight embedding generation service.

Production design:
- Uses FastEmbed instead of SentenceTransformers/PyTorch.
- FastEmbed runs through ONNX Runtime.
- The embedding model is imported and loaded lazily.
- The model is cached after first use.
- CPU threading is limited to reduce Render memory pressure.
- Query embeddings use the "query:" prefix.
- Document/chunk embeddings use the "passage:" prefix.
"""

from __future__ import annotations

import logging
import os
from threading import Lock
from typing import Any, Optional

from app.core.config import Settings, get_settings
from app.core.exceptions import (
    EmbeddingGenerationError,
    EmbeddingModelLoadError,
)
from app.models.chunk import Chunk

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Generates text embeddings using FastEmbed.

    FastEmbed is intentionally imported only when an embedding is actually
    requested. This prevents the embedding runtime from being loaded during
    normal FastAPI startup.

    The default model is:
        BAAI/bge-small-en-v1.5

    It produces 384-dimensional embeddings.
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self._settings = settings or get_settings()

        # FastEmbed model is loaded lazily.
        self._model: Any = None

        # Prevent multiple concurrent requests from loading the model twice.
        self._load_lock = Lock()

    @property
    def model_name(self) -> str:
        """Return the configured embedding model name."""
        return self._settings.embedding_model_name

    def _configure_runtime(self) -> None:
        """
        Configure CPU/thread limits before FastEmbed/ONNX Runtime is loaded.

        Render Free has a very small memory limit, so unnecessary thread
        creation should be avoided.
        """

        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    def _get_model(self) -> Any:
        """
        Lazily import and initialize FastEmbed.

        FastEmbed and its model are NOT loaded during application startup.
        They are loaded only when embedding generation is actually needed.
        """

        if self._model is not None:
            return self._model

        with self._load_lock:
            if self._model is not None:
                return self._model

            self._configure_runtime()

            try:
                logger.info(
                    "Loading lightweight embedding model '%s' with FastEmbed...",
                    self._settings.embedding_model_name,
                )

                # IMPORTANT:
                # Keep this import lazy. Do not import FastEmbed at module
                # import time because the application should start without
                # initializing the embedding runtime.
                from fastembed import TextEmbedding

                self._model = TextEmbedding(
                    model_name=self._settings.embedding_model_name,
                )

                logger.info(
                    "Embedding model '%s' loaded successfully with FastEmbed.",
                    self._settings.embedding_model_name,
                )

            except Exception as exc:
                logger.exception(
                    "Failed to load embedding model '%s'.",
                    self._settings.embedding_model_name,
                )

                raise EmbeddingModelLoadError(
                    self._settings.embedding_model_name,
                    str(exc),
                ) from exc

            return self._model

    def embed_text(self, text: str) -> list[float]:
        """
        Generate an embedding for a single query string.

        Retrieval queries are prefixed with ``query:`` as recommended for
        BGE retrieval models.
        """

        if not text or not text.strip():
            raise EmbeddingGenerationError(
                "cannot embed empty or whitespace-only text."
            )

        return self._embed_texts(
            [text],
            prefix="query: ",
        )[0]

    def embed_texts(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """
        Generate embeddings for a list of query strings.

        This method preserves the existing public API and treats the input
        as retrieval queries.
        """

        return self._embed_texts(
            texts,
            prefix="query: ",
        )

    def _embed_texts(
        self,
        texts: list[str],
        *,
        prefix: str,
    ) -> list[list[float]]:
        """
        Internal embedding implementation.

        FastEmbed returns a generator of NumPy arrays. We immediately convert
        the generated vectors into ordinary Python lists because the rest of
        the application expects ``list[list[float]]``.
        """

        if not texts:
            raise EmbeddingGenerationError(
                "cannot embed an empty list of texts."
            )

        blank_indices = [
            index
            for index, text in enumerate(texts)
            if not text or not text.strip()
        ]

        if blank_indices:
            raise EmbeddingGenerationError(
                "cannot embed empty or whitespace-only text "
                f"at indices {blank_indices}."
            )

        model = self._get_model()

        batch_size = max(
            1,
            int(
                getattr(
                    self._settings,
                    "embedding_batch_size",
                    2,
                )
            ),
        )

        prepared_texts = [
            f"{prefix}{text.strip()}"
            for text in texts
        ]

        try:
            embeddings = model.embed(
                prepared_texts,
                batch_size=batch_size,
            )

            vectors = [
                vector.tolist()
                for vector in embeddings
            ]

        except Exception as exc:
            logger.exception(
                "Failed to generate embeddings for %d texts.",
                len(texts),
            )

            raise EmbeddingGenerationError(
                str(exc),
                batch_size=batch_size,
            ) from exc

        if len(vectors) != len(texts):
            raise EmbeddingGenerationError(
                "embedding model returned an unexpected number of vectors: "
                f"expected {len(texts)}, got {len(vectors)}."
            )

        logger.debug(
            "Generated %d embeddings using model '%s'.",
            len(vectors),
            self.model_name,
        )

        return vectors

    def _embed_passages(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """
        Generate passage/document embeddings.

        Chunks stored in ChromaDB are passages, so they receive the
        ``passage:`` prefix.
        """

        return self._embed_texts(
            texts,
            prefix="passage: ",
        )

    def embed_chunks(
        self,
        chunks: list[Chunk],
    ) -> list[Chunk]:
        """
        Generate embeddings for chunks that do not already have one.

        Existing embeddings are preserved.
        """

        if not chunks:
            raise EmbeddingGenerationError(
                "cannot embed an empty list of chunks."
            )

        indices_needing_embedding = [
            index
            for index, chunk in enumerate(chunks)
            if not chunk.has_embedding()
        ]

        if not indices_needing_embedding:
            logger.debug(
                "All %d chunks already have embeddings.",
                len(chunks),
            )
            return list(chunks)

        texts_to_embed = [
            chunks[index].text
            for index in indices_needing_embedding
        ]

        new_embeddings = self._embed_passages(
            texts_to_embed,
        )

        embedded_chunks = list(chunks)

        for index, embedding in zip(
            indices_needing_embedding,
            new_embeddings,
        ):
            embedded_chunks[index] = chunks[index].model_copy(
                update={
                    "embedding": embedding,
                }
            )

        logger.info(
            "Embedded %d of %d chunks using model '%s'.",
            len(indices_needing_embedding),
            len(chunks),
            self.model_name,
        )

        return embedded_chunks