"""
Loader for Markdown (.md / .markdown) source documents.

Markdown files are plain text, so extraction reuses the same
encoding-fallback strategy as :class:`app.ingestion.txt_loader.TxtLoader`.
In addition, this loader can optionally strip common Markdown syntax
markers (headers, emphasis, links, code fences, etc.) so that downstream
embedding models see natural-language content rather than markup noise,
while still preserving the readable text itself.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from app.core.exceptions import EmptyDocumentError, TextExtractionError

logger = logging.getLogger(__name__)

_FALLBACK_ENCODINGS: tuple[str, ...] = ("utf-8-sig", "latin-1", "cp1252")

# Precompiled regular expressions used to strip Markdown syntax markers
# while preserving the underlying readable text.
_CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`([^`]*)`")
_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_HEADER_RE = re.compile(r"^\s{0,3}#{1,6}\s*", re.MULTILINE)
_EMPHASIS_RE = re.compile(r"(\*\*\*|\*\*|\*|___|__|_)")
_BLOCKQUOTE_RE = re.compile(r"^\s{0,3}>\s?", re.MULTILINE)
_HORIZONTAL_RULE_RE = re.compile(r"^\s{0,3}([-*_])\1{2,}\s*$", re.MULTILINE)
_LIST_MARKER_RE = re.compile(r"^\s{0,3}([-*+]|\d+\.)\s+", re.MULTILINE)
_TABLE_PIPE_RE = re.compile(r"\|")
_MULTI_BLANK_LINES_RE = re.compile(r"\n{3,}")


class MarkdownLoader:
    """Extracts text content from Markdown (.md / .markdown) files."""

    supported_extensions: tuple[str, ...] = (".md", ".markdown")

    def load(self, file_path: Path, strip_markdown_syntax: bool = True) -> str:
        """
        Read and return the text content of a Markdown file.

        Args:
            file_path: Path to the ``.md``/``.markdown`` file on disk.
            strip_markdown_syntax: If True (default), Markdown syntax
                markers are removed so downstream consumers receive
                natural-language text. If False, the raw Markdown source
                is returned unmodified.

        Returns:
            The extracted (and optionally cleaned) text content.

        Raises:
            TextExtractionError: If the file cannot be read or decoded.
            EmptyDocumentError: If the resulting text is empty.
        """
        if not file_path.exists():
            raise TextExtractionError(str(file_path), "file does not exist.")
        if not file_path.is_file():
            raise TextExtractionError(str(file_path), "path is not a regular file.")

        raw_markdown = self._read_with_encoding_fallback(file_path)

        text = self._strip_markdown_syntax(raw_markdown) if strip_markdown_syntax else raw_markdown

        if not text or not text.strip():
            raise EmptyDocumentError(file_path.name)

        logger.info(
            "Successfully extracted text from '%s' (%d characters, syntax_stripped=%s).",
            file_path.name,
            len(text),
            strip_markdown_syntax,
        )
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

    @staticmethod
    def _strip_markdown_syntax(markdown_text: str) -> str:
        """
        Remove common Markdown syntax markers while preserving the
        underlying readable text.

        This is a lightweight, dependency-free transformation (regex
        based) rather than a full Markdown-to-AST parser, which keeps the
        ingestion pipeline free of an additional third-party dependency
        for a task that does not require full CommonMark fidelity.

        Args:
            markdown_text: Raw Markdown source text.

        Returns:
            Plain text with Markdown syntax markers removed.
        """
        text = markdown_text

        # Remove fenced code blocks entirely first, since their contents
        # are typically not natural language and can contain characters
        # that would otherwise be misinterpreted by later substitutions.
        text = _CODE_FENCE_RE.sub(" ", text)
        text = _INLINE_CODE_RE.sub(r"\1", text)

        # Images and links: keep the human-readable label, drop the URL.
        text = _IMAGE_RE.sub(r"\1", text)
        text = _LINK_RE.sub(r"\1", text)

        # Structural markers.
        text = _HEADER_RE.sub("", text)
        text = _BLOCKQUOTE_RE.sub("", text)
        text = _HORIZONTAL_RULE_RE.sub("", text)
        text = _LIST_MARKER_RE.sub("", text)

        # Inline emphasis markers (bold/italic) and table pipes.
        text = _EMPHASIS_RE.sub("", text)
        text = _TABLE_PIPE_RE.sub(" ", text)

        # Collapse excessive blank lines left behind by the removals above.
        text = _MULTI_BLANK_LINES_RE.sub("\n\n", text)

        return text.strip()