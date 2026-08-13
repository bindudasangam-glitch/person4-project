"""
Document API schemas.

Defines the response models returned by the `/documents` router
(`app/api/routes/documents.py`). Kept separate from the internal
`app.models.document.Document` domain model so the API's public shape
can evolve independently of internal storage/pipeline details.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from app.models.document import Document, DocumentFileType, DocumentStatus

__all__ = [
    "DocumentDeleteResponse",
    "DocumentListResponse",
    "DocumentSummary",
    "DocumentUploadResponse",
]


class DocumentUploadResponse(BaseModel):
    """
    Full details of a single ingested document, returned after upload
    and when fetching a document by ID.

    Attributes:
        document_id: The document's unique identifier.
        source_name: The document's original filename.
        file_type: The detected source format.
        status: The document's current ingestion pipeline status.
        file_size_bytes: Size of the uploaded file, in bytes.
        page_count: Number of pages, if known.
        author: Document author, if present in the source file's metadata.
        title: Document title, if present in the source file's metadata.
        created_at: Document creation timestamp, if present in the
            source file's metadata.
    """

    document_id: str
    source_name: str
    file_type: DocumentFileType
    status: DocumentStatus
    file_size_bytes: int = Field(..., ge=0)
    page_count: Optional[int] = None
    author: Optional[str] = None
    title: Optional[str] = None
    created_at: Optional[datetime] = None

    @classmethod
    def from_document(cls, document: Document) -> "DocumentUploadResponse":
        """
        Build a response from an internal :class:`Document` instance.

        Args:
            document: The document to convert.

        Returns:
            A populated :class:`DocumentUploadResponse`.
        """
        return cls(
            document_id=document.document_id,
            source_name=document.source_name,
            file_type=document.metadata.file_type,
            status=document.status,
            file_size_bytes=document.metadata.file_size_bytes,
            page_count=document.metadata.page_count,
            author=document.metadata.author,
            title=document.metadata.title,
            created_at=document.metadata.created_at,
        )


class DocumentSummary(BaseModel):
    """
    A lightweight summary of a single ingested document, used in list responses.

    Attributes:
        document_id: The document's unique identifier.
        source_name: The document's original filename.
        file_type: The detected source format.
        status: The document's current ingestion pipeline status.
        file_size_bytes: Size of the uploaded file, in bytes.
    """

    document_id: str
    source_name: str
    file_type: DocumentFileType
    status: DocumentStatus
    file_size_bytes: int = Field(..., ge=0)

    @classmethod
    def from_document(cls, document: Document) -> "DocumentSummary":
        """
        Build a summary from an internal :class:`Document` instance.

        Args:
            document: The document to convert.

        Returns:
            A populated :class:`DocumentSummary`.
        """
        return cls(
            document_id=document.document_id,
            source_name=document.source_name,
            file_type=document.metadata.file_type,
            status=document.status,
            file_size_bytes=document.metadata.file_size_bytes,
        )


class DocumentListResponse(BaseModel):
    """
    The response body for listing all ingested documents.

    Attributes:
        total: Total number of documents in the response.
        documents: The document summaries.
    """

    total: int = Field(..., ge=0)
    documents: list[DocumentSummary] = Field(default_factory=list)


class DocumentDeleteResponse(BaseModel):
    """
    Confirmation response for a document deletion request.

    Attributes:
        document_id: The identifier of the deleted document.
        deleted: Whether the deletion succeeded.
    """

    document_id: str
    deleted: bool