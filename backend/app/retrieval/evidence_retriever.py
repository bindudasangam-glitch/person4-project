"""
Composable evidence retrieval.

:class:`app.retrieval.retriever.Retriever` provides a single, self-contained
implementation of "embed query, search, build evidence" for the common
case. This module provides the same capability assembled from smaller,
independently reusable pieces -- :class:`app.retrieval.similarity_search.SimilaritySearch`
and :class:`app.retrieval.source_attribution.SourceAttributionBuilder` --
for consumers that need to customize or reuse those individual steps
(e.g. swapping in a different ranking strategy, or attaching source
attribution to results obtained some other way).
"""

from __future__ import annotations

import logging
from typing import Optional

from app.core.config import Settings, get_settings
from app.embeddings.embedding_service import EmbeddingService
from app.models.evidence import Evidence, EvidenceBundle
from app.retrieval.similarity_search import SimilaritySearch
from app.retrieval.source_attribution import SourceAttributionBuilder

logger = logging.getLogger(__name__)


class EvidenceRetriever:
    """
    Retrieves ranked, deduplicated evidence for a query by composing
    query embedding, similarity search, and source attribution as
    separate, independently reusable steps.
    """

    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        similarity_search: Optional[SimilaritySearch] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        """
        Args:
            embedding_service: Service used to embed the query text.
                Defaults to a new :class:`EmbeddingService`.
            similarity_search: Component used to search the vector
                store. Defaults to a new :class:`SimilaritySearch`.
            settings: Application settings providing default top-k and
                similarity threshold. Defaults to the cached process-wide
                settings instance.
        """
        self._settings = settings or get_settings()
        self._embedding_service = embedding_service or EmbeddingService(self._settings)
        self._similarity_search = similarity_search or SimilaritySearch(settings=self._settings)

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        collection_name: Optional[str] = None,
        document_id: Optional[str] = None,
    ) -> EvidenceBundle:
        """
        Retrieve ranked evidence for a query.

        Args:
            query: The natural-language query or claim text to retrieve
                evidence for. Must be non-empty.
            top_k: Maximum number of results to retrieve before
                threshold filtering and deduplication. Defaults to the
                configured application retrieval top-k.
            similarity_threshold: Minimum similarity score required for
                a result to be kept. Defaults to the configured
                application similarity threshold.
            collection_name: Target ChromaDB collection name. Defaults
                to the configured application collection name.
            document_id: If provided, restricts retrieval to chunks
                belonging to this specific document only.

        Returns:
            An :class:`EvidenceBundle` containing evidence results that
            meet the similarity threshold, deduplicated by chunk,
            ordered by descending similarity score.

        Raises:
            EmbeddingGenerationError: If the query cannot be embedded.
            CollectionNotFoundError: If the target collection does not exist.
        """
        if not query or not query.strip():
            logger.warning("Empty query passed to evidence retriever; returning empty evidence bundle.")
            return EvidenceBundle(query=query, results=[])

        effective_threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else self._settings.similarity_score_threshold
        )

        query_embedding = self._embedding_service.embed_text(query)
        search_results = self._similarity_search.search(
            query_embedding=query_embedding,
            top_k=top_k,
            collection_name=collection_name,
            document_id=document_id,
        )

        evidence_results = [
            Evidence(
                chunk_id=result.chunk_id,
                text=result.text,
                similarity_score=result.similarity_score,
                attribution=SourceAttributionBuilder.build(result.chunk_id, result.metadata),
                metadata=result.metadata,
            )
            for result in search_results
        ]

        bundle = EvidenceBundle(query=query, results=evidence_results)
        bundle = bundle.deduplicate_by_chunk()
        bundle = bundle.filter_by_threshold(effective_threshold)

        logger.info(
            "Retrieved %d evidence result(s) for query via composed pipeline (threshold=%.3f).",
            len(bundle.results),
            effective_threshold,
        )
        return bundle