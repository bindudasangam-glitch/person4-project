"""
Document processor: the single entry point for ingesting a source file.

This module ties together file validation and format-specific text
extraction (PDF, DOCX, TXT, Markdown) behind one uniform interface,
producing a fully populated :class:`app.models.document.Document`
instance ready for the text-cleaning and chunking stages.

Callers (e.g. the API upload route or a batch ingestion script) should
depend only on :class:`DocumentProcessor` and never invoke individual
loaders directly, so that validation is never accidentally bypassed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from app.core.config import Settings, get_settings
from app.core.exceptions import DocumentIngestionError, TextExtractionError
from app.ingestion.docx_loader import DocxLoader
from app.ingestion.file_validator import FileValidator
from app.ingestion.markdown_loader import MarkdownLoader
from app.ingestion.pdf_loader import PdfLoader
from app.ingestion.txt_loader import TxtLoader
from app.models.document import Document, DocumentFileType, DocumentMetadata, DocumentStatus

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    Orchestrates validation and text extraction for a single uploaded
    document, regardless of its underlying file format.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        file_validator: Optional[FileValidator] = None,
        pdf_loader: Optional[PdfLoader] = None,
        docx_loader: Optional[DocxLoader] = None,
        txt_loader: Optional[TxtLoader] = None,
        markdown_loader: Optional[MarkdownLoader] = None,
    ) -> None:
        """
        Args:
            settings: Application settings. Defaults to the cached
                process-wide settings instance.
            file_validator: Validator used to check file type/size before
                extraction. Defaults to a new :class:`FileValidator`.
            pdf_loader: Loader used for ``.pdf`` files.
            docx_loader: Loader used for ``.docx`` files.
            txt_loader: Loader used for ``.txt`` files.
            markdown_loader: Loader used for ``.md``/``.markdown`` files.
        """
        self._settings = settings or get_settings()
        self._file_validator = file_validator or FileValidator(self._settings)
        self._pdf_loader = pdf_loader or PdfLoader()
        self._docx_loader = docx_loader or DocxLoader()
        self._txt_loader = txt_loader or TxtLoader()
        self._markdown_loader = markdown_loader or MarkdownLoader()

    def process(self, file_path: Path, file_size_bytes: Optional[int] = None) -> Document:
        """
        Validate and extract text from a single uploaded document,
        returning a fully populated :class:`Document`.

        Args:
            file_path: Path to the uploaded file on disk.
            file_size_bytes: Size of the file in bytes, if already known
                (e.g. from an HTTP upload). If omitted, it is read from
                the filesystem.

        Returns:
            A :class:`Document` instance with ``status`` set to
            ``DocumentStatus.TEXT_EXTRACTED`` and ``raw_text`` populated.

        Raises:
            UnsupportedFileTypeError: If the file's extension is not supported.
            FileTooLargeError: If the file exceeds the configured size limit.
            TextExtractionError: If text extraction fails.
            EmptyDocumentError: If no extractable text is found.
        """
        file_type = self._file_validator.validate(file_path, file_size_bytes)
        resolved_size_bytes = (
            file_size_bytes if file_size_bytes is not None else file_path.stat().st_size
        )

        try:
            document = self._dispatch_extraction(file_path, file_type, resolved_size_bytes)
        except DocumentIngestionError:
            # Already one of our typed exceptions; re-raise as-is so callers
            # can pattern-match on the specific failure category.
            raise
        except Exception as exc:  # noqa: BLE001 - guarantee callers only ever see typed exceptions
            raise TextExtractionError(str(file_path), f"unexpected extraction failure: {exc}") from exc

        logger.info(
            "Processed document '%s' (document_id=%s, type=%s, %d characters).",
            document.source_name,
            document.document_id,
            file_type.value,
            len(document.raw_text),
        )
        return document

    def _dispatch_extraction(
        self, file_path: Path, file_type: DocumentFileType, file_size_bytes: int
    ) -> Document:
        """
        Dispatch to the correct format-specific loader and assemble the
        resulting :class:`Document`.

        Args:
            file_path: Path to the uploaded file on disk.
            file_type: The validated file type.
            file_size_bytes: Size of the file in bytes.

        Returns:
            A fully populated :class:`Document` instance.
        """
        if file_type is DocumentFileType.PDF:
            result = self._pdf_loader.load(file_path)
            metadata = DocumentMetadata(
                original_filename=file_path.name,
                file_type=file_type,
                file_size_bytes=file_size_bytes,
                page_count=result.page_count,
                author=result.author,
                title=result.title,
                created_at=result.created_at,
                extra=result.extra_metadata,
            )
            raw_text = result.text

        elif file_type is DocumentFileType.DOCX:
            result = self._docx_loader.load(file_path)
            metadata = DocumentMetadata(
                original_filename=file_path.name,
                file_type=file_type,
                file_size_bytes=file_size_bytes,
                page_count=None,
                author=result.author,
                title=result.title,
                created_at=result.created_at,
                extra=result.extra_metadata,
            )
            raw_text = result.text

        elif file_type is DocumentFileType.TXT:
            raw_text = self._txt_loader.load(file_path)
            metadata = DocumentMetadata(
                original_filename=file_path.name,
                file_type=file_type,
                file_size_bytes=file_size_bytes,
            )

        elif file_type is DocumentFileType.MARKDOWN:
            raw_text = self._markdown_loader.load(file_path)
            metadata = DocumentMetadata(
                original_filename=file_path.name,
                file_type=file_type,
                file_size_bytes=file_size_bytes,
            )

        else:  # pragma: no cover - defensive; validator already constrains file_type
            raise TextExtractionError(str(file_path), f"no loader registered for file type '{file_type}'.")

        return Document(
            source_path=str(file_path),
            raw_text=raw_text,
            metadata=metadata,
            status=DocumentStatus.TEXT_EXTRACTED,
        )