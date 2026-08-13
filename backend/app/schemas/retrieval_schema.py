"""
Retrieval API schemas.

Defines the request and response models for the `/retrieval` router
(`app/api/routes/retrival.py`). Kept separate from the internal
`app.models.evidence.EvidenceBundle` domain model so the API's public
shape (a flat, easy-to-consume evidence list) can differ from the
internal representation.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from app.models.evidence import EvidenceBundle

__all__ = ["RetrievalRequest", "RetrievalResponse", "RetrievedEvidence"]


class RetrievalRequest(BaseModel):
    """
    Request payload for a semantic evidence retrieval query.

    Attributes:
        query: The natural-language query to retrieve evidence for.
        top_k: Maximum number of results to retrieve before threshold
            filtering and deduplication. If omitted, the server-side
            configured default is used.
        similarity_threshold: Minimum similarity score required for a
            result to be kept. If omitted, the server-side configured
            default is used.
        document_id: If provided, restricts retrieval to chunks
            belonging to this specific document only.
    """

    query: str = Field(..., min_length=1, max_length=5000)
    top_k: Optional[int] = Field(default=None, ge=1, le=100)
    similarity_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    document_id: Optional[str] = None


class RetrievedEvidence(BaseModel):
    """
    A single retrieved evidence result, flattened for API consumption.

    Attributes:
        chunk_id: Identifier of the chunk this evidence was retrieved from.
        text: The evidence chunk's text content.
        similarity_score: Similarity of this evidence to the query, in ``[0, 1]``.
        document_id: Identifier of the source document.
        source_name: Human-readable filename/source of the document.
        page_number: 1-indexed page number the chunk falls on, if known.
    """

    chunk_id: str
    text: str
    similarity_score: float = Field(..., ge=0.0, le=1.0)
    document_id: str
    source_name: str
    page_number: Optional[int] = None


class RetrievalResponse(BaseModel):
    """
    The response body for a semantic evidence retrieval query.

    Attributes:
        query: The query this response was retrieved for.
        total_results: Number of evidence results returned.
        results: The retrieved evidence, ordered by descending similarity score.
    """

    query: str
    total_results: int = Field(..., ge=0)
    results: list[RetrievedEvidence] = Field(default_factory=list)

    @classmethod
    def from_bundle(cls, bundle: EvidenceBundle) -> "RetrievalResponse":
        """
        Build a response from an internal :class:`EvidenceBundle`.

        Args:
            bundle: The evidence bundle returned by the retriever.

        Returns:
            A populated :class:`RetrievalResponse`.
        """
        results = [
            RetrievedEvidence(
                chunk_id=evidence.chunk_id,
                text=evidence.text,
                similarity_score=evidence.similarity_score,
                document_id=evidence.attribution.document_id,
                source_name=evidence.attribution.source_name,
                page_number=evidence.attribution.page_number,
            )
            for evidence in bundle.results
        ]
        return cls(query=bundle.query, total_results=len(results), results=results)