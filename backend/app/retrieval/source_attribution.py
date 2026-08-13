"""
Source attribution construction.

Factors out the logic for reconstructing a :class:`SourceAttribution`
from the raw metadata dictionary stored alongside each chunk in the
vector store. Kept as an independently reusable, easily testable unit
so provenance-reconstruction logic is not duplicated across every
consumer that needs to turn a vector store result into attributable
evidence.
"""

from __future__ import annotations

import logging
from typing import Any

from app.models.evidence import SourceAttribution
from app.vectorstore.chroma_client import VectorQueryResult

logger = logging.getLogger(__name__)

_UNKNOWN_SOURCE_NAME = "unknown"


class SourceAttributionBuilder:
    """Builds :class:`SourceAttribution` instances from raw chunk metadata."""

    @staticmethod
    def build(chunk_id: str, metadata: dict[str, Any]) -> SourceAttribution:
        """
        Build a :class:`SourceAttribution` from a chunk ID and its
        associated metadata.

        Args:
            chunk_id: The unique identifier of the chunk this evidence
                was retrieved from.
            metadata: The raw metadata dictionary stored alongside the
                chunk in the vector store (expected to contain
                ``document_id``, ``source_name``, and optionally
                ``page_number``).

        Returns:
            A :class:`SourceAttribution` instance. Missing
            ``document_id`` defaults to an empty string; missing
            ``source_name`` defaults to ``"unknown"``; missing
            ``page_number`` defaults to ``None``.
        """
        return SourceAttribution(
            document_id=str(metadata.get("document_id", "")),
            source_name=str(metadata.get("source_name", _UNKNOWN_SOURCE_NAME)),
            chunk_id=chunk_id,
            page_number=metadata.get("page_number"),
        )

    @classmethod
    def build_many(cls, results: list[VectorQueryResult]) -> list[SourceAttribution]:
        """
        Build a :class:`SourceAttribution` for each of a list of vector
        query results.

        Args:
            results: Raw result rows from the vector store.

        Returns:
            A list of :class:`SourceAttribution` instances, in the same
            order as ``results``.
        """
        return [cls.build(result.chunk_id, result.metadata) for result in results]