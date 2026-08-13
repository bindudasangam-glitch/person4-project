"""
Chunk domain model.

Defines :class:`Chunk`, the unit of text that flows through the RAG
pipeline from ingestion through embedding, vector storage, and
retrieval, along with its provenance metadata (:class:`ChunkMetadata`).

Modeled as Pydantic ``BaseModel`` (rather than a plain dataclass, as
Person 1's ``ClaimModel`` is) because chunks cross serialization
boundaries -- they are embedded, written to ChromaDB as
metadata/documents, and reconstructed from vector-store query results --
so validation and JSON (de)serialization support are needed, and
:meth:`~pydantic.BaseModel.model_copy` is relied on to produce an
updated chunk with an embedding attached without mutating the original.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator

__all__ = ["Chunk", "ChunkMetadata"]


class ChunkMetadata(BaseModel):
    """
    Provenance metadata for a single chunk.

    Attributes:
        document_id: Identifier of the document this chunk was derived from.
        source_name: Human-readable filename/source of the owning document.
        chunk_index: Zero-based position of this chunk within its document.
        page_number: 1-indexed page number this chunk falls on, if the
            source format has pages (e.g. PDF) and this could be estimated.
        start_char: Character offset (inclusive) where this chunk begins
            within the document's text.
        end_char: Character offset (exclusive) where this chunk ends
            within the document's text.
        extra: Additional, format-specific metadata not covered above.
    """

    document_id: str
    source_name: str
    chunk_index: int = Field(..., ge=0)
    page_number: Optional[int] = Field(default=None, ge=1)
    start_char: int = Field(..., ge=0)
    end_char: int = Field(..., ge=0)
    extra: dict[str, Any] = Field(default_factory=dict)

    @field_validator("end_char")
    @classmethod
    def _validate_end_after_start(cls, end_char: int, info: Any) -> int:
        """Ensure ``end_char`` is strictly greater than ``start_char`` when both are known."""
        start_char = info.data.get("start_char")
        if start_char is not None and end_char <= start_char:
            raise ValueError(
                f"end_char ({end_char}) must be strictly greater than start_char ({start_char})."
            )
        return end_char


class Chunk(BaseModel):
    """
    A single embeddable, retrievable unit of document text.

    Attributes:
        chunk_id: Deterministic, globally unique identifier for this chunk.
        text: The chunk's text content.
        metadata: Provenance metadata for this chunk.
        embedding: The chunk's dense embedding vector, if it has been
            generated yet. ``None`` for freshly-chunked, not-yet-embedded chunks.
    """

    chunk_id: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)
    metadata: ChunkMetadata
    embedding: Optional[list[float]] = None

    def has_embedding(self) -> bool:
        """Return True if this chunk already has an embedding vector attached."""
        return self.embedding is not None