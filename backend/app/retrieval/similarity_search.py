"""
Vector similarity search.

Wraps :class:`app.vectorstore.chroma_client.ChromaVectorStore` behind a
narrower interface focused purely on ranked similarity search, plus a
couple of small, independently reusable ranking utilities (top-N
limiting and threshold filtering). Kept separate from
:class:`app.retrieval.retriever.Retriever` so consumers that need only
the search step (without query embedding or evidence-model assembly)
can depend on this directly.
"""

from __future__ import annotations

import logging
from typing import Optional

from app.core.config import Settings, get_settings
from app.vectorstore.chroma_client import ChromaVectorStore, VectorQueryResult

logger = logging.getLogger(__name__)


class SimilaritySearch:
    """Performs ranked similarity search against the configured vector store."""

    def __init__(
        self,
        vector_store: Optional[ChromaVectorStore] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        """
        Args:
            vector_store: Vector store to search against. Defaults to a
                new :class:`ChromaVectorStore`.
            settings: Application settings providing the default top-k.
                Defaults to the cached process-wide settings instance.
        """
        self._settings = settings or get_settings()
        self._vector_store = vector_store or ChromaVectorStore(self._settings)

    def search(
        self,
        query_embedding: list[float],
        top_k: Optional[int] = None,
        collection_name: Optional[str] = None,
        document_id: Optional[str] = None,
    ) -> list[VectorQueryResult]:
        """
        Search the vector store for the chunks most similar to a query
        embedding.

        Args:
            query_embedding: The dense embedding vector of the query.
            top_k: Maximum number of results to return. Defaults to the
                configured application retrieval top-k.
            collection_name: Target ChromaDB collection name. Defaults
                to the configured application collection name.
            document_id: If provided, restricts the search to chunks
                belonging to this specific document only.

        Returns:
            A list of :class:`VectorQueryResult` instances, ordered by
            descending similarity score.

        Raises:
            CollectionNotFoundError: If the target collection does not exist.
        """
        effective_top_k = top_k if top_k is not None else self._settings.retrieval_top_k
        where_filter = {"document_id": document_id} if document_id else None

        results = self._vector_store.similarity_search(
            query_embedding=query_embedding,
            top_k=effective_top_k,
            collection_name=collection_name,
            where=where_filter,
        )

        logger.debug("Similarity search returned %d result(s) (top_k=%d).", len(results), effective_top_k)
        return results

    @staticmethod
    def filter_by_threshold(
        results: list[VectorQueryResult], similarity_threshold: float
    ) -> list[VectorQueryResult]:
        """
        Keep only results whose similarity score meets or exceeds a
        minimum threshold.

        Args:
            results: Results to filter.
            similarity_threshold: Minimum similarity score required to
                keep a result.

        Returns:
            A new list containing only qualifying results, preserving
            their original relative order.
        """
        return [result for result in results if result.similarity_score >= similarity_threshold]

    @staticmethod
    def top_n(results: list[VectorQueryResult], n: int) -> list[VectorQueryResult]:
        """
        Return the top ``n`` results by descending similarity score.

        Args:
            results: Results to rank and limit.
            n: Maximum number of results to return. Must be non-negative.

        Returns:
            The top ``n`` results, ordered by descending similarity score.

        Raises:
            ValueError: If ``n`` is negative.
        """
        if n < 0:
            raise ValueError(f"n must be non-negative, got {n}.")
        return sorted(results, key=lambda result: result.similarity_score, reverse=True)[:n]