"""
Embedding generation package.

Exposes :class:`EmbeddingService`, the application's single interface
for generating dense vector embeddings from text via a configurable
Sentence Transformers model.
"""

from __future__ import annotations

from app.embeddings.embedding_service import EmbeddingService

__all__ = ["EmbeddingService"]