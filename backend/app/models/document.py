"""
Document domain model.

Defines :class:`Document`, the object representing a single uploaded
source file as it moves through the ingestion pipeline: validation,
format-specific text extraction, cleaning, and (eventually) chunking
and indexing. :class:`DocumentMetadata` carries provenance/format
details, :class:`DocumentFileType` enumerates supported source formats,
and :class:`DocumentStatus` tracks the document's position in that
pipeline.

Modeled as Pydantic ``BaseModel`` (rather than a plain dataclass) since
``Document`` instances are persisted to a JSON-file-backed registry
(``app/api/routes/documents.py::DocumentRegistry``) via
``model_dump_json()`` / ``model_validate()``, and updated immutably at
each pipeline stage via ``model_copy(update={...})``.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

__all__ = ["Document", "DocumentFileType", "DocumentMetadata", "DocumentStatus"]


class DocumentFileType(str, Enum):
    """Source document formats supported by the ingestion pipeline."""

    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    MARKDOWN = "markdown"


class DocumentStatus(str, Enum):
    """A document's position within the ingestion pipeline."""

    TEXT_EXTRACTED = "text_extracted"
    CLEANED = "cleaned"
    INDEXED = "indexed"


class DocumentMetadata(BaseModel):
    """
    Provenance and format metadata for a document.

    Attributes:
        original_filename: The filename as originally uploaded.
        file_type: The detected source format.
        file_size_bytes: Size of the uploaded file, in bytes.
        page_count: Number of pages, if the format has pages (e.g. PDF)
            and this could be determined.
        author: Document author, if present in the source file's
            embedded metadata.
        title: Document title, if present in the source file's embedded metadata.
        created_at: Document creation timestamp, if present in the
            source file's embedded metadata.
        extra: Any additional, format-specific metadata fields (e.g.
            PDF producer/subject, DOCX keywords/category).
    """

    original_filename: str = Field(..., min_length=1)
    file_type: DocumentFileType
    file_size_bytes: int = Field(..., ge=0)
    page_count: Optional[int] = Field(default=None, ge=1)
    author: Optional[str] = None
    title: Optional[str] = None
    created_at: Optional[datetime] = None
    extra: dict[str, Any] = Field(default_factory=dict)


class Document(BaseModel):
    """
    A single source document as it moves through the ingestion pipeline.

    Attributes:
        document_id: Globally unique identifier for this document,
            generated automatically if not provided.
        source_path: Filesystem path the document was ingested from, if
            it originated from disk (as opposed to, e.g., an in-memory upload).
        raw_text: Text extracted directly from the source file, before cleaning.
        cleaned_text: Normalized text after :class:`app.processing.text_cleaner.TextCleaner`
            has run. ``None`` until the cleaning stage has completed.
        metadata: Provenance and format metadata for this document.
        status: The document's current position in the ingestion pipeline.
    """

    document_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_path: Optional[str] = None
    raw_text: str = ""
    cleaned_text: Optional[str] = None
    metadata: DocumentMetadata
    status: DocumentStatus = DocumentStatus.TEXT_EXTRACTED

    @property
    def source_name(self) -> str:
        """The document's human-readable source name (its original filename)."""
        return self.metadata.original_filename