"""
Evidence retrieval service.

Combines query embedding generation with ChromaDB similarity search to
produce a ranked, deduplicated, threshold-filtered bundle of evidence for
a given query or claim. This is the primary interface the fact
verification stage (and any other consumer needing retrieved context)
should depend on, rather than talking to the embedding service or vector
store directly.
"""

from __future__ import annotations

import logging
from typing import Optional

from app.core.config import Settings, get_settings
from app.embeddings.embedding_service import EmbeddingService
from app.models.evidence import Evidence, EvidenceBundle, SourceAttribution
from app.vectorstore.chroma_client import ChromaVectorStore, VectorQueryResult

logger = logging.getLogger(__name__)


class Retriever:
    """
    Retrieves ranked, deduplicated evidence for a query by combining
    embedding generation with vector similarity search.
    """

    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        vector_store: Optional[ChromaVectorStore] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        """
        Args:
            embedding_service: Service used to embed the query text.
                Defaults to a new :class:`EmbeddingService`.
            vector_store: Vector store to search against. Defaults to a
                new :class:`ChromaVectorStore`.
            settings: Application settings providing default top-k and
                similarity threshold. Defaults to the cached process-wide
                settings instance.
        """
        self._settings = settings or get_settings()
        self._embedding_service = embedding_service or EmbeddingService(self._settings)
        self._vector_store = vector_store or ChromaVectorStore(self._settings)

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
            top_k: Maximum number of results to retrieve from the vector
                store before threshold filtering and deduplication.
                Defaults to the configured application retrieval top-k.
            similarity_threshold: Minimum similarity score required for a
                result to be kept. Defaults to the configured application
                similarity threshold.
            collection_name: Target ChromaDB collection name. Defaults to
                the configured application collection name.
            document_id: If provided, restricts retrieval to chunks
                belonging to this specific document only.

        Returns:
            An :class:`EvidenceBundle` containing evidence results that
            meet the similarity threshold, deduplicated by chunk, ordered
            by descending similarity score.

        Raises:
            EmbeddingGenerationError: If the query cannot be embedded.
            CollectionNotFoundError: If the target collection does not exist.
        """
        if not query or not query.strip():
            logger.warning("Empty query passed to retriever; returning empty evidence bundle.")
            return EvidenceBundle(query=query, results=[])

        effective_top_k = top_k if top_k is not None else self._settings.retrieval_top_k
        effective_threshold = (
            similarity_threshold if similarity_threshold is not None else self._settings.similarity_score_threshold
        )

        query_embedding = self._embedding_service.embed_text(query)

        where_filter = {"document_id": document_id} if document_id else None
        raw_results = self._vector_store.similarity_search(
            query_embedding=query_embedding,
            top_k=effective_top_k,
            collection_name=collection_name,
            where=where_filter,
        )

        bundle = EvidenceBundle(query=query, results=[self._to_evidence(result) for result in raw_results])
        bundle = bundle.deduplicate_by_chunk()
        bundle = bundle.filter_by_threshold(effective_threshold)

        logger.info(
            "Retrieved %d evidence result(s) for query (top_k=%d, threshold=%.3f).",
            len(bundle.results),
            effective_top_k,
            effective_threshold,
        )
        return bundle

    def _to_evidence(self, result: VectorQueryResult) -> Evidence:
        """
        Convert a raw vector query result into an :class:`Evidence`
        instance, reconstructing source attribution from stored metadata.

        Args:
            result: A raw result row from the vector store.

        Returns:
            A fully populated :class:`Evidence` instance.
        """
        metadata = result.metadata
        attribution = SourceAttribution(
            document_id=str(metadata.get("document_id", "")),
            source_name=str(metadata.get("source_name", "unknown")),
            chunk_id=result.chunk_id,
            page_number=metadata.get("page_number"),
        )
        return Evidence(
            chunk_id=result.chunk_id,
            text=result.text,
            similarity_score=result.similarity_score,
            attribution=attribution,
            metadata=metadata,
        )