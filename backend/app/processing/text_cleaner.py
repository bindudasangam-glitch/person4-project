"""
Text cleaning for extracted document content.

Raw text extracted from PDFs, DOCX files, and other formats frequently
contains inconsistent whitespace, stray control characters, and
Unicode variants of otherwise-identical characters (e.g. different
dash or quote glyphs). This module normalizes that text into a clean,
consistent form before it is chunked and embedded, while preserving the
document's paragraph structure.
"""

from __future__ import annotations

import logging
import re
import unicodedata

from app.core.exceptions import EmptyDocumentError
from app.models.document import Document, DocumentStatus

logger = logging.getLogger(__name__)

# Matches three or more consecutive blank lines, collapsed down to one
# blank line (i.e. a single paragraph break) during whitespace normalization.
_EXCESS_BLANK_LINES_RE = re.compile(r"\n\s*\n\s*\n+")

# Matches runs of horizontal whitespace (spaces/tabs), collapsed to a
# single space. Newlines are handled separately so paragraph structure
# is preserved.
_HORIZONTAL_WHITESPACE_RE = re.compile(r"[ \t]+")

# Trailing horizontal whitespace at the end of a line.
_TRAILING_WHITESPACE_RE = re.compile(r"[ \t]+\n")

# Non-printable / control characters (excluding newline and tab, which are
# meaningful for structure) that sometimes leak in from PDF extraction.
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class TextCleaner:
    """Normalizes and validates raw extracted document text."""

    def clean(self, document: Document) -> Document:
        """
        Clean a document's raw extracted text and return an updated
        :class:`Document` with ``cleaned_text`` populated.

        Args:
            document: A document with ``raw_text`` already populated
                (typically with status ``DocumentStatus.TEXT_EXTRACTED``).

        Returns:
            A new :class:`Document` instance with ``cleaned_text`` set and
            ``status`` advanced to ``DocumentStatus.CLEANED``.

        Raises:
            EmptyDocumentError: If the text is empty after cleaning (e.g.
                the raw text consisted only of control characters or
                whitespace).
        """
        cleaned_text = self.clean_text(document.raw_text)

        if not cleaned_text:
            raise EmptyDocumentError(document.source_name)

        updated = document.model_copy(
            update={"cleaned_text": cleaned_text, "status": DocumentStatus.CLEANED}
        )

        logger.info(
            "Cleaned text for '%s': %d raw characters -> %d cleaned characters.",
            document.source_name,
            len(document.raw_text),
            len(cleaned_text),
        )
        return updated

    def clean_text(self, text: str) -> str:
        """
        Apply the full text-cleaning pipeline to a raw string.

        Steps applied, in order:
            1. Unicode normalization (NFKC), so visually/semantically
               equivalent characters (e.g. different quote or dash
               variants) collapse to a single canonical form.
            2. Control character removal.
            3. Line-ending normalization (CRLF/CR -> LF).
            4. Horizontal whitespace collapsing (spaces/tabs).
            5. Trailing-whitespace removal at line ends.
            6. Collapsing three-or-more consecutive blank lines down to one.
            7. Leading/trailing whitespace stripping of the whole text.

        Args:
            text: Raw input text.

        Returns:
            The cleaned text. Returns an empty string if the input is
            empty, ``None``-like, or reduces to nothing after cleaning.
        """
        if not text:
            return ""

        normalized = unicodedata.normalize("NFKC", text)
        normalized = _CONTROL_CHARS_RE.sub("", normalized)
        normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
        normalized = _HORIZONTAL_WHITESPACE_RE.sub(" ", normalized)
        normalized = _TRAILING_WHITESPACE_RE.sub("\n", normalized)
        normalized = _EXCESS_BLANK_LINES_RE.sub("\n\n", normalized)

        return normalized.strip()

    def is_empty(self, text: str) -> bool:
        """Return True if the given text is empty or contains only whitespace."""
        return not text or not text.strip()

    def deduplicate_paragraphs(self, text: str) -> str:
        """
        Remove exact-duplicate paragraphs while preserving the order of
        first occurrence.

        This is useful for documents (particularly PDFs) that
        accidentally repeat running headers, footers, or boilerplate on
        every page, which would otherwise pollute chunk content and skew
        embedding similarity.

        Args:
            text: Cleaned text, with paragraphs separated by blank lines.

        Returns:
            Text with duplicate paragraphs removed.
        """
        paragraphs = text.split("\n\n")
        seen: set[str] = set()
        deduplicated: list[str] = []

        for paragraph in paragraphs:
            normalized_key = paragraph.strip().lower()
            if not normalized_key or normalized_key in seen:
                continue
            seen.add(normalized_key)
            deduplicated.append(paragraph.strip())

        return "\n\n".join(deduplicated)