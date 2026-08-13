"""
PDF text and metadata extraction.

Wraps ``pypdf`` behind a small, typed interface so the rest of the
application never depends on the underlying parsing library directly.
Extracts per-page text (joined into a single document body) along with
whatever author/title/creation-date metadata the PDF's document
information dictionary provides.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.core.exceptions import EmptyDocumentError, TextExtractionError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PdfExtractionResult:
    """
    The result of extracting text and metadata from a PDF file.

    Attributes:
        text: The full extracted text, with page texts joined by blank lines.
        page_count: The total number of pages in the PDF.
        page_texts: The extracted text of each page, in page order.
        author: The document author, if present in the PDF metadata.
        title: The document title, if present in the PDF metadata.
        created_at: The document creation timestamp, if present in the PDF metadata.
        extra_metadata: Any additional metadata fields (e.g. producer, subject).
    """

    text: str
    page_count: int
    page_texts: list[str]
    author: Optional[str] = None
    title: Optional[str] = None
    created_at: Optional[datetime] = None
    extra_metadata: dict[str, str] = field(default_factory=dict)


class PdfLoader:
    """Extracts text and metadata from ``.pdf`` files using ``pypdf``."""

    def load(self, file_path: Path) -> PdfExtractionResult:
        """
        Extract text and metadata from a PDF file.

        Args:
            file_path: Path to the PDF file on disk.

        Returns:
            A :class:`PdfExtractionResult` with the extracted text and metadata.

        Raises:
            TextExtractionError: If the file does not exist or cannot be
                parsed as a valid PDF.
            EmptyDocumentError: If the PDF contains no extractable text
                (e.g. a scanned, image-only document).
        """
        if not file_path.exists():
            raise TextExtractionError(str(file_path), "file does not exist.")

        try:
            reader = PdfReader(str(file_path))
        except (PdfReadError, OSError, ValueError) as exc:
            raise TextExtractionError(str(file_path), f"failed to open PDF: {exc}") from exc

        page_texts = self._extract_page_texts(file_path, reader)

        text = "\n\n".join(page_text for page_text in page_texts if page_text)
        if not text.strip():
            raise EmptyDocumentError(file_path.name)

        author, title, created_at, extra_metadata = self._extract_metadata(reader)

        logger.info(
            "Extracted %d character(s) from %d page(s) of PDF '%s'.",
            len(text),
            len(page_texts),
            file_path.name,
        )
        return PdfExtractionResult(
            text=text,
            page_count=len(page_texts),
            page_texts=page_texts,
            author=author,
            title=title,
            created_at=created_at,
            extra_metadata=extra_metadata,
        )

    @staticmethod
    def _extract_page_texts(file_path: Path, reader: PdfReader) -> list[str]:
        """
        Extract text from each page of an opened PDF.

        Args:
            file_path: Path to the PDF file (used for error messages).
            reader: The opened :class:`PdfReader` instance.

        Returns:
            A list of extracted text strings, one per page, in page order.

        Raises:
            TextExtractionError: If text extraction fails for any page.
        """
        page_texts: list[str] = []
        for page_number, page in enumerate(reader.pages, start=1):
            try:
                page_texts.append((page.extract_text() or "").strip())
            except Exception as exc:  # noqa: BLE001 - surface any per-page failure uniformly
                raise TextExtractionError(
                    str(file_path), f"failed to extract text from page {page_number}: {exc}"
                ) from exc
        return page_texts

    @staticmethod
    def _extract_metadata(
        reader: PdfReader,
    ) -> tuple[Optional[str], Optional[str], Optional[datetime], dict[str, str]]:
        """
        Extract author/title/creation-date/extra metadata from a PDF's
        document information dictionary, if present.

        Args:
            reader: The opened :class:`PdfReader` instance.

        Returns:
            A ``(author, title, created_at, extra_metadata)`` tuple.
        """
        info = reader.metadata
        if info is None:
            return None, None, None, {}

        author = info.author or None
        title = info.title or None
        created_at = info.creation_date

        extra_metadata = {
            key: value
            for key, value in {
                "producer": info.producer,
                "creator": info.creator,
                "subject": info.subject,
            }.items()
            if value
        }

        return author, title, created_at, extra_metadata