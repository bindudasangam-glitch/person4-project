"""
Upload validation for ingested documents.

Performs the two checks that must happen before any format-specific
text extraction is attempted: that the file exists and has a supported
extension, and that it does not exceed the configured maximum upload
size. Centralizing these checks here (rather than duplicating them in
each format-specific loader) ensures every ingestion path enforces the
same rules.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from app.core.config import Settings, get_settings
from app.core.exceptions import FileTooLargeError, TextExtractionError, UnsupportedFileTypeError
from app.models.document import DocumentFileType

logger = logging.getLogger(__name__)

_BYTES_PER_MEGABYTE = 1024 * 1024


class FileValidator:
    """Validates an uploaded file's existence, extension, and size before extraction."""

    _EXTENSION_TO_FILE_TYPE: dict[str, DocumentFileType] = {
        ".pdf": DocumentFileType.PDF,
        ".docx": DocumentFileType.DOCX,
        ".txt": DocumentFileType.TXT,
        ".md": DocumentFileType.MARKDOWN,
        ".markdown": DocumentFileType.MARKDOWN,
    }

    def __init__(self, settings: Optional[Settings] = None) -> None:
        """
        Args:
            settings: Application settings providing the configured
                maximum upload size (``max_upload_size_mb``). Defaults
                to the cached process-wide settings instance.
        """
        self._settings = settings or get_settings()

    @property
    def supported_extensions(self) -> tuple[str, ...]:
        """The set of file extensions accepted for ingestion, e.g. ``('.pdf', '.docx', ...)``."""
        return tuple(sorted(self._EXTENSION_TO_FILE_TYPE))

    def validate(self, file_path: Path, file_size_bytes: Optional[int] = None) -> DocumentFileType:
        """
        Validate an uploaded file and determine its document type.

        Args:
            file_path: Path to the file on disk.
            file_size_bytes: Size of the file in bytes, if already known
                (e.g. from an HTTP upload). If omitted, it is read from
                the filesystem.

        Returns:
            The detected :class:`DocumentFileType` for the file.

        Raises:
            TextExtractionError: If the file does not exist or is not a
                regular file.
            UnsupportedFileTypeError: If the file's extension is not one
                of the supported types.
            FileTooLargeError: If the file exceeds the configured
                maximum upload size.
        """
        if not file_path.exists():
            raise TextExtractionError(str(file_path), "file does not exist.")
        if not file_path.is_file():
            raise TextExtractionError(str(file_path), "path is not a regular file.")

        file_type = self._resolve_file_type(file_path)
        self._check_size(file_path, file_size_bytes)

        logger.debug(
            "Validated file '%s' as type '%s'.", file_path.name, file_type.value
        )
        return file_type

    def _resolve_file_type(self, file_path: Path) -> DocumentFileType:
        """
        Determine the :class:`DocumentFileType` for a file based on its
        extension.

        Args:
            file_path: Path to the file being validated.

        Returns:
            The matching :class:`DocumentFileType`.

        Raises:
            UnsupportedFileTypeError: If the extension is not supported.
        """
        extension = file_path.suffix.lower()
        file_type = self._EXTENSION_TO_FILE_TYPE.get(extension)

        if file_type is None:
            raise UnsupportedFileTypeError(
                f"Unsupported file extension '{extension}' for file '{file_path.name}'. "
                f"Supported extensions: {', '.join(self.supported_extensions)}."
            )
        return file_type

    def _check_size(self, file_path: Path, file_size_bytes: Optional[int]) -> None:
        """
        Verify that a file does not exceed the configured maximum
        upload size.

        Args:
            file_path: Path to the file being validated.
            file_size_bytes: Size of the file in bytes, if already
                known. If omitted, it is read from the filesystem.

        Raises:
            FileTooLargeError: If the file's size exceeds the configured
                maximum upload size.
        """
        size_bytes = file_size_bytes if file_size_bytes is not None else file_path.stat().st_size
        max_bytes = self._settings.max_upload_size_mb * _BYTES_PER_MEGABYTE

        if size_bytes > max_bytes:
            raise FileTooLargeError(
                f"File '{file_path.name}' is {size_bytes} bytes, exceeding the maximum "
                f"allowed upload size of {max_bytes} bytes ({self._settings.max_upload_size_mb} MB)."
            )