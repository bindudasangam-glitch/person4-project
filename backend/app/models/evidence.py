"""
Evidence domain model.

Defines the models produced by the retrieval layer and consumed by the
verification layer: :class:`SourceAttribution` (where a piece of
evidence came from), :class:`Evidence` (a single retrieved, scored
chunk of text), and :class:`EvidenceBundle` (an ordered, query-scoped
collection of evidence with convenience operations for deduplication
and threshold filtering).
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

__all__ = ["Evidence", "EvidenceBundle", "SourceAttribution"]


class SourceAttribution(BaseModel):
    """
    Identifies exactly where a piece of evidence came from.

    Attributes:
        document_id: Identifier of the source document.
        source_name: Human-readable filename/source of the document.
        chunk_id: Identifier of the specific chunk this evidence was
            retrieved from.
        page_number: 1-indexed page number the chunk falls on, if known.
    """

    document_id: str
    source_name: str
    chunk_id: str
    page_number: Optional[int] = Field(default=None, ge=1)


class Evidence(BaseModel):
    """
    A single piece of retrieved evidence: a scored chunk of text with
    full source attribution.

    Attributes:
        chunk_id: Identifier of the chunk this evidence was retrieved from.
        text: The evidence chunk's text content.
        similarity_score: Similarity of this evidence to the originating
            query, in ``[0, 1]`` (higher is more similar).
        attribution: Where this evidence came from.
        metadata: The raw metadata associated with the underlying chunk
            in the vector store (kept for diagnostic/debugging purposes).
    """

    chunk_id: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)
    similarity_score: float = Field(..., ge=0.0, le=1.0)
    attribution: SourceAttribution
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvidenceBundle(BaseModel):
    """
    An ordered, query-scoped collection of evidence.

    Attributes:
        query: The query or claim text this evidence was retrieved for.
        results: The evidence results, ordered by descending similarity score.
    """

    query: str
    results: list[Evidence] = Field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        """True if this bundle contains no evidence results."""
        return len(self.results) == 0

    def deduplicate_by_chunk(self) -> "EvidenceBundle":
        """
        Remove duplicate evidence results that reference the same chunk,
        keeping only the first (highest-ranked) occurrence of each.

        Returns:
            A new :class:`EvidenceBundle` with duplicate chunk references removed.
        """
        seen_chunk_ids: set[str] = set()
        deduplicated: list[Evidence] = []

        for evidence in self.results:
            if evidence.chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(evidence.chunk_id)
            deduplicated.append(evidence)

        return self.model_copy(update={"results": deduplicated})

    def filter_by_threshold(self, similarity_threshold: float) -> "EvidenceBundle":
        """
        Keep only evidence results whose similarity score meets or
        exceeds a minimum threshold.

        Args:
            similarity_threshold: Minimum similarity score required to
                keep a result.

        Returns:
            A new :class:`EvidenceBundle` containing only qualifying
            results, preserving their original relative order.

        Raises:
            ValueError: If ``similarity_threshold`` is not within ``[0, 1]``.
        """
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError(
                f"similarity_threshold must be within [0, 1], got {similarity_threshold}."
            )

        filtered = [
            evidence for evidence in self.results if evidence.similarity_score >= similarity_threshold
        ]
        return self.model_copy(update={"results": filtered})