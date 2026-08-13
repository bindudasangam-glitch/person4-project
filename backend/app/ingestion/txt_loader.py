"""
Loader for plain text (.txt) source documents.

Text files require no format-specific parsing, but still need a robust
encoding-fallback strategy since uploaded files are not guaranteed to
be UTF-8. :class:`app.ingestion.markdown_loader.MarkdownLoader` reuses
the same fallback strategy implemented here.
"""

from __future__ import annotations

import logging
from pathlib import Path

from app.core.exceptions import EmptyDocumentError, TextExtractionError

logger = logging.getLogger(__name__)

_FALLBACK_ENCODINGS: tuple[str, ...] = ("utf-8-sig", "latin-1", "cp1252")


class TxtLoader:
    """Extracts text content from plain text (``.txt``) files."""

    def load(self, file_path: Path) -> str:
        """
        Read and return the text content of a plain text file.

        Args:
            file_path: Path to the ``.txt`` file on disk.

        Returns:
            The extracted text content.

        Raises:
            TextExtractionError: If the file cannot be read or decoded.
            EmptyDocumentError: If the resulting text is empty.
        """
        if not file_path.exists():
            raise TextExtractionError(str(file_path), "file does not exist.")
        if not file_path.is_file():
            raise TextExtractionError(str(file_path), "path is not a regular file.")

        text = self._read_with_encoding_fallback(file_path)

        if not text or not text.strip():
            raise EmptyDocumentError(file_path.name)

        logger.info("Successfully extracted text from '%s' (%d characters).", file_path.name, len(text))
        return text

    def _read_with_encoding_fallback(self, file_path: Path) -> str:
        """
        Attempt to decode the file as UTF-8 first, then fall back through
        a small set of common encodings.

        Args:
            file_path: Path to the file to read.

        Returns:
            The successfully decoded text content.

        Raises:
            TextExtractionError: If none of the attempted encodings succeed.
        """
        encodings_to_try = ("utf-8", *_FALLBACK_ENCODINGS)
        last_error: Exception | None = None

        for encoding in encodings_to_try:
            try:
                return file_path.read_text(encoding=encoding)
            except UnicodeDecodeError as exc:
                last_error = exc
                logger.debug(
                    "Failed to decode '%s' using encoding '%s'; trying next fallback.",
                    file_path.name,
                    encoding,
                )
                continue
            except OSError as exc:
                raise TextExtractionError(str(file_path), f"I/O error while reading file: {exc}") from exc

        raise TextExtractionError(
            str(file_path),
            f"Could not decode file using any of the supported encodings "
            f"{encodings_to_try}. Last error: {last_error}",
        )