"""
Fixed-size text chunking.

Splits document text into overlapping, boundary-aware character
windows suitable for embedding and retrieval. Unlike
:class:`app.chunking.semantic_chunker.SemanticChunker`, this does not
require an embedding model and is the default, low-cost chunking
strategy used by the ingestion pipeline.
"""

from __future__ import annotations

import logging
from typing import Optional

from app.core.config import Settings, get_settings
from app.core.exceptions import InvalidChunkConfigurationError
from app.models.chunk import Chunk, ChunkMetadata
from app.models.document import Document

logger = logging.getLogger(__name__)


class TextChunker:
    """Splits document text into fixed-size, overlapping chunks."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        """
        Args:
            settings: Application settings providing the default
                ``chunk_size`` and ``chunk_overlap``. Defaults to the
                cached process-wide settings instance.
        """
        self._settings = settings or get_settings()

    def chunk_document(
        self,
        document: Document,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> list[Chunk]:
        """
        Split a document's cleaned text into fixed-size chunks.

        Args:
            document: The document to chunk. ``cleaned_text`` is used
                if present, otherwise ``raw_text``.
            chunk_size: Maximum number of characters per chunk.
                Defaults to the configured application value.
            chunk_overlap: Number of characters of overlap between
                consecutive chunks. Defaults to the configured
                application value.

        Returns:
            A list of :class:`Chunk` instances, in document order, each
            with provenance metadata (document ID, source name, chunk
            index, and character offsets) populated. Returns an empty
            list if the document has no extractable text.

        Raises:
            InvalidChunkConfigurationError: If ``chunk_size`` is not
                positive, or ``chunk_overlap`` is negative or not
                strictly less than ``chunk_size``.
        """
        effective_chunk_size = chunk_size if chunk_size is not None else self._settings.chunk_size
        effective_chunk_overlap = (
            chunk_overlap if chunk_overlap is not None else self._settings.chunk_overlap
        )
        self._validate_configuration(effective_chunk_size, effective_chunk_overlap)

        text = document.cleaned_text or document.raw_text
        segments = self.split_text(
            text, chunk_size=effective_chunk_size, chunk_overlap=effective_chunk_overlap
        )

        page_count = getattr(document.metadata, "page_count", None)
        text_length = len(text) if text else 0

        chunks: list[Chunk] = []
        for chunk_index, (chunk_text, start_char, end_char) in enumerate(segments):
            page_number = (
                self._estimate_page_number(start_char, text_length, page_count)
                if page_count
                else None
            )
            metadata = ChunkMetadata(
                document_id=document.document_id,
                source_name=document.source_name,
                chunk_index=chunk_index,
                page_number=page_number,
                start_char=start_char,
                end_char=end_char,
            )
            chunks.append(
                Chunk(
                    chunk_id=self._build_chunk_id(document.document_id, chunk_index),
                    text=chunk_text,
                    metadata=metadata,
                )
            )

        logger.info(
            "Chunked document '%s' into %d chunk(s) (chunk_size=%d, chunk_overlap=%d).",
            document.source_name,
            len(chunks),
            effective_chunk_size,
            effective_chunk_overlap,
        )
        return chunks

    def split_text(
        self,
        text: str,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> list[tuple[str, int, int]]:
        """
        Split raw text into fixed-size, overlapping, boundary-aware
        segments.

        Segments are cut as close as possible to ``chunk_size``
        characters, but the cut point is nudged back to the nearest
        preceding whitespace (when one exists within the segment) so
        words are not split mid-token. The final segment always ends
        exactly at the end of the text.

        Args:
            text: The text to split.
            chunk_size: Maximum number of characters per segment.
                Defaults to the configured application value.
            chunk_overlap: Number of characters of overlap between
                consecutive segments. Defaults to the configured
                application value.

        Returns:
            A list of ``(segment_text, start_char, end_char)`` tuples,
            in document order. Returns an empty list for empty or
            whitespace-only text.

        Raises:
            InvalidChunkConfigurationError: If ``chunk_size`` is not
                positive, or ``chunk_overlap`` is negative or not
                strictly less than ``chunk_size``.
        """
        effective_chunk_size = chunk_size if chunk_size is not None else self._settings.chunk_size
        effective_chunk_overlap = (
            chunk_overlap if chunk_overlap is not None else self._settings.chunk_overlap
        )
        self._validate_configuration(effective_chunk_size, effective_chunk_overlap)

        if not text or not text.strip():
            return []

        text_length = len(text)
        if text_length <= effective_chunk_size:
            return [(text, 0, text_length)]

        segments: list[tuple[str, int, int]] = []
        start = 0

        while start < text_length:
            end = min(start + effective_chunk_size, text_length)

            if end < text_length:
                boundary = text.rfind(" ", start, end)
                if boundary > start:
                    end = boundary

            segments.append((text[start:end], start, end))

            if end >= text_length:
                break

            next_start = end - effective_chunk_overlap
            if next_start <= start:
                # Guard against non-progress (e.g. a boundary that fell
                # right at the overlap point) by advancing to the end
                # of the current segment instead.
                next_start = end
            start = next_start

        return segments

    @staticmethod
    def _validate_configuration(chunk_size: int, chunk_overlap: int) -> None:
        """
        Validate chunking configuration.

        Raises:
            InvalidChunkConfigurationError: If ``chunk_size`` is not
                positive, ``chunk_overlap`` is negative, or
                ``chunk_overlap`` is not strictly less than ``chunk_size``.
        """
        if chunk_size <= 0:
            raise InvalidChunkConfigurationError(f"chunk_size must be positive, got {chunk_size}.")
        if chunk_overlap < 0:
            raise InvalidChunkConfigurationError(
                f"chunk_overlap must not be negative, got {chunk_overlap}."
            )
        if chunk_overlap >= chunk_size:
            raise InvalidChunkConfigurationError(
                f"chunk_overlap ({chunk_overlap}) must be strictly less than chunk_size ({chunk_size})."
            )

    @staticmethod
    def _estimate_page_number(start_char: int, total_length: int, page_count: Optional[int]) -> Optional[int]:
        """
        Estimate the 1-indexed page number a character offset falls on,
        assuming characters are distributed evenly across pages.

        Args:
            start_char: The character offset within the document.
            total_length: Total character length of the document text.
            page_count: Total number of pages, if known.

        Returns:
            An estimated 1-indexed page number, or ``None`` if
            ``page_count`` is not known or the document is empty.
        """
        if not page_count or total_length <= 0:
            return None
        estimated = int((start_char / total_length) * page_count) + 1
        return max(1, min(page_count, estimated))

    @staticmethod
    def _build_chunk_id(document_id: str, chunk_index: int) -> str:
        """
        Build a deterministic chunk ID from the document ID and chunk
        index, so re-chunking an unchanged document is idempotent.

        Args:
            document_id: The owning document's unique identifier.
            chunk_index: The chunk's zero-based position within the document.

        Returns:
            A deterministic, human-readable chunk ID string.
        """
        return f"{document_id}_{chunk_index}"