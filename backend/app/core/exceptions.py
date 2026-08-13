"""
RAG & Verification module exception hierarchy.

Defines every domain-specific exception raised by Person 2's module
(document ingestion, chunking, embedding, vector storage, retrieval,
and fact verification). All exceptions share a single common base,
:class:`RAGFactVerificationError`, so API route handlers can catch one
type to translate any expected domain failure into a clean HTTP 400
response, while truly unexpected exceptions still propagate to a
generic 500 handler.

This module is intentionally independent of Person 1's exception
hierarchy (``app.services.*`` defines its own exceptions for the
hallucination-detection pipeline) -- the two error domains do not need
to share a base class, since they are caught and handled by entirely
separate API routers.
"""

from __future__ import annotations

from typing import Any, Optional


class RAGFactVerificationError(Exception):
    """
    Base class for all Person 2 (RAG & Verification) domain exceptions.

    Attributes:
        message: A human-readable description of the failure.
        details: Optional structured context about the failure (e.g.
            the offending value, file path, or operation name), useful
            for logging or API error payloads without needing to parse
            the message string.
    """

    def __init__(self, message: str, details: Optional[dict[str, Any]] = None) -> None:
        """
        Args:
            message: A human-readable description of the failure.
            details: Optional structured context about the failure.
        """
        super().__init__(message)
        self.message = message
        self.details: dict[str, Any] = details or {}

    def __str__(self) -> str:
        return self.message


# --------------------------------------------------------------------------- #
# Ingestion errors
# --------------------------------------------------------------------------- #


class DocumentIngestionError(RAGFactVerificationError):
    """Base class for errors occurring while ingesting an uploaded document."""


class UnsupportedFileTypeError(DocumentIngestionError):
    """Raised when an uploaded file's extension is not one of the supported document types."""


class FileTooLargeError(DocumentIngestionError):
    """Raised when an uploaded file exceeds the configured maximum upload size."""


class TextExtractionError(DocumentIngestionError):
    """
    Raised when text cannot be extracted from a source file (missing
    file, corrupt/unreadable format, or an unrecoverable decoding failure).
    """

    def __init__(self, source: str, message: str) -> None:
        """
        Args:
            source: Path or filename of the file that failed extraction.
            message: A human-readable description of what went wrong.
        """
        super().__init__(f"{source}: {message}", details={"source": source})


class EmptyDocumentError(DocumentIngestionError):
    """Raised when a document contains no extractable, non-whitespace text."""

    def __init__(self, source_name: str) -> None:
        """
        Args:
            source_name: Filename or identifier of the empty document.
        """
        super().__init__(
            f"Document '{source_name}' contains no extractable text.",
            details={"source_name": source_name},
        )


# --------------------------------------------------------------------------- #
# Chunking errors
# --------------------------------------------------------------------------- #


class InvalidChunkConfigurationError(RAGFactVerificationError):
    """
    Raised when chunking parameters are invalid (e.g. non-positive chunk
    size, negative overlap, or overlap not strictly less than chunk size).
    """


# --------------------------------------------------------------------------- #
# Embedding errors
# --------------------------------------------------------------------------- #


class EmbeddingGenerationError(RAGFactVerificationError):
    """Raised when embedding generation fails for a piece of text or batch."""

    def __init__(self, message: str, batch_size: Optional[int] = None) -> None:
        """
        Args:
            message: A human-readable description of what went wrong.
            batch_size: The batch size in effect when the failure
                occurred, if applicable.
        """
        details: dict[str, Any] = {}
        if batch_size is not None:
            details["batch_size"] = batch_size
        super().__init__(message, details=details)


class EmbeddingModelLoadError(RAGFactVerificationError):
    """Raised when the underlying embedding model fails to load."""

    def __init__(self, model_name: str, message: str) -> None:
        """
        Args:
            model_name: Name or path of the embedding model that failed to load.
            message: A human-readable description of what went wrong.
        """
        super().__init__(
            f"Failed to load embedding model '{model_name}': {message}",
            details={"model_name": model_name},
        )


# --------------------------------------------------------------------------- #
# Vector store errors
# --------------------------------------------------------------------------- #


class VectorStoreConnectionError(RAGFactVerificationError):
    """Raised when a connection to the persistent vector store cannot be established."""

    def __init__(self, persist_directory: str, message: str) -> None:
        """
        Args:
            persist_directory: Filesystem path the vector store was
                attempting to persist to or connect from.
            message: A human-readable description of what went wrong.
        """
        super().__init__(
            f"Vector store connection error at '{persist_directory}': {message}",
            details={"persist_directory": persist_directory},
        )


class VectorStoreWriteError(RAGFactVerificationError):
    """Raised when a write operation (add/upsert/delete/query) against the vector store fails."""

    def __init__(self, operation: str, message: str) -> None:
        """
        Args:
            operation: Name of the failing operation (e.g. ``"add"``,
                ``"upsert"``, ``"delete"``, ``"query"``).
            message: A human-readable description of what went wrong.
        """
        super().__init__(
            f"Vector store '{operation}' operation failed: {message}",
            details={"operation": operation},
        )


class CollectionNotFoundError(RAGFactVerificationError):
    """Raised when a requested vector store collection does not exist."""

    def __init__(self, collection_name: str) -> None:
        """
        Args:
            collection_name: Name of the collection that could not be found.
        """
        super().__init__(
            f"Collection '{collection_name}' does not exist. "
            f"Upload and index at least one document before querying.",
            details={"collection_name": collection_name},
        )


# --------------------------------------------------------------------------- #
# Verification errors
# --------------------------------------------------------------------------- #


class InvalidClaimError(RAGFactVerificationError):
    """Raised when a claim submitted for verification is invalid (e.g. empty text)."""


__all__ = [
    "RAGFactVerificationError",
    "DocumentIngestionError",
    "UnsupportedFileTypeError",
    "FileTooLargeError",
    "TextExtractionError",
    "EmptyDocumentError",
    "InvalidChunkConfigurationError",
    "EmbeddingGenerationError",
    "EmbeddingModelLoadError",
    "VectorStoreConnectionError",
    "VectorStoreWriteError",
    "CollectionNotFoundError",
    "InvalidClaimError",
]