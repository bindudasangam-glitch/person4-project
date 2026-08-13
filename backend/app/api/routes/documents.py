"""
Document ingestion API routes.

Exposes endpoints to upload a document (running it through validation,
text extraction, cleaning, chunking, embedding, and vector storage in a
single request), list previously ingested documents, retrieve a single
document's details, and delete a document along with its stored chunks.

A lightweight JSON-file-backed registry is used to persist document
metadata across process restarts (chunk vectors themselves are already
persisted by ChromaDB; this registry only tracks document-level
bookkeeping such as filenames and status).
"""

from __future__ import annotations

import json
import logging
import shutil
import uuid
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, UploadFile, status

from app.core.config import Settings, get_settings
from app.core.exceptions import RAGFactVerificationError
from app.embeddings.embedding_service import EmbeddingService
from app.ingestion.document_processor import DocumentProcessor
from app.models.document import Document, DocumentStatus
from app.processing.chunker import TextChunker
from app.processing.text_cleaner import TextCleaner
from app.schemas.document_schema import (
    DocumentDeleteResponse,
    DocumentListResponse,
    DocumentSummary,
    DocumentUploadResponse,
)
from app.vectorstore.chroma_client import ChromaVectorStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])


class DocumentRegistry:
    """
    A minimal, JSON-file-backed registry of ingested document metadata.

    This keeps document bookkeeping (filenames, status, timestamps)
    available across process restarts without requiring a full database
    dependency. Chunk-level data lives in ChromaDB; this registry only
    tracks document-level records needed for list/detail/delete
    endpoints.
    """

    def __init__(self, registry_path: Path) -> None:
        self._registry_path = registry_path
        self._lock = Lock()
        self._registry_path.parent.mkdir(parents=True, exist_ok=True)
        if not self._registry_path.exists():
            self._write_all({})

    def save(self, document: Document) -> None:
        """Persist or update a document's metadata record."""
        with self._lock:
            records = self._read_all()
            records[document.document_id] = json.loads(document.model_dump_json())
            self._write_all(records)

    def get(self, document_id: str) -> Optional[Document]:
        """Retrieve a single document's metadata record by ID, if it exists."""
        with self._lock:
            records = self._read_all()
        raw = records.get(document_id)
        return Document.model_validate(raw) if raw else None

    def list_all(self) -> list[Document]:
        """Return all registered documents."""
        with self._lock:
            records = self._read_all()
        return [Document.model_validate(raw) for raw in records.values()]

    def delete(self, document_id: str) -> bool:
        """Remove a document's metadata record. Returns True if it existed."""
        with self._lock:
            records = self._read_all()
            existed = document_id in records
            records.pop(document_id, None)
            self._write_all(records)
        return existed

    def _read_all(self) -> dict:
        try:
            return json.loads(self._registry_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _write_all(self, records: dict) -> None:
        self._registry_path.write_text(json.dumps(records, indent=2), encoding="utf-8")


@lru_cache(maxsize=1)
def get_document_registry() -> DocumentRegistry:
    """Return the process-wide cached document registry instance."""
    settings = get_settings()
    return DocumentRegistry(settings.resolved_processed_dir() / "document_registry.json")


@lru_cache(maxsize=1)
def get_document_processor() -> DocumentProcessor:
    """Return the process-wide cached document processor instance."""
    return DocumentProcessor()


@lru_cache(maxsize=1)
def get_text_cleaner() -> TextCleaner:
    """Return the process-wide cached text cleaner instance."""
    return TextCleaner()


@lru_cache(maxsize=1)
def get_text_chunker() -> TextChunker:
    """Return the process-wide cached text chunker instance."""
    return TextChunker()


@lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService:
    """Return the process-wide cached embedding service instance (model loaded once)."""
    return EmbeddingService()


@lru_cache(maxsize=1)
def get_vector_store() -> ChromaVectorStore:
    """Return the process-wide cached ChromaDB vector store instance."""
    return ChromaVectorStore()


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and fully ingest a document (PDF, DOCX, TXT, or Markdown).",
)
async def upload_document(
    file: UploadFile,
    settings: Settings = Depends(get_settings),
    processor: DocumentProcessor = Depends(get_document_processor),
    cleaner: TextCleaner = Depends(get_text_cleaner),
    chunker: TextChunker = Depends(get_text_chunker),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    vector_store: ChromaVectorStore = Depends(get_vector_store),
    registry: DocumentRegistry = Depends(get_document_registry),
) -> DocumentUploadResponse:
    """
    Upload a document and run it through the full ingestion pipeline:
    validation, text extraction, cleaning, chunking, embedding, and
    vector storage.

    Args:
        file: The uploaded file.

    Returns:
        A :class:`DocumentUploadResponse` describing the ingested document.

    Raises:
        HTTPException: 400 if the file fails validation or extraction;
            500 if an unexpected error occurs during processing.
    """
    upload_dir = settings.resolved_upload_dir()
    upload_dir.mkdir(parents=True, exist_ok=True)

    safe_filename = f"{uuid.uuid4().hex}_{Path(file.filename or 'upload').name}"
    destination_path = upload_dir / safe_filename

    try:
        with destination_path.open("wb") as destination_file:
            shutil.copyfileobj(file.file, destination_file)

        file_size_bytes = destination_path.stat().st_size

        document = processor.process(destination_path, file_size_bytes)
        document = cleaner.clean(document)

        chunks = chunker.chunk_document(document)
        chunks = embedding_service.embed_chunks(chunks)
        vector_store.upsert_chunks(chunks)

        document = document.model_copy(update={"status": DocumentStatus.INDEXED})
        registry.save(document)

        logger.info(
            "Document '%s' fully ingested (document_id=%s, %d chunks).",
            document.source_name,
            document.document_id,
            len(chunks),
        )
        return DocumentUploadResponse.from_document(document)

    except RAGFactVerificationError as exc:
        logger.warning("Document ingestion failed for '%s': %s", file.filename, exc)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - guarantee a clean 500 rather than a raw traceback
        logger.exception("Unexpected error while ingesting document '%s'.", file.filename)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error while processing document: {exc}",
        ) from exc


@router.get(
    "",
    response_model=DocumentListResponse,
    summary="List all previously ingested documents.",
)
async def list_documents(
    registry: DocumentRegistry = Depends(get_document_registry),
) -> DocumentListResponse:
    """
    List all documents currently tracked by the ingestion registry.

    Returns:
        A :class:`DocumentListResponse` containing document summaries.
    """
    documents = registry.list_all()
    summaries = [DocumentSummary.from_document(document) for document in documents]
    return DocumentListResponse(total=len(summaries), documents=summaries)


@router.get(
    "/{document_id}",
    response_model=DocumentUploadResponse,
    summary="Retrieve details for a single ingested document.",
)
async def get_document(
    document_id: str,
    registry: DocumentRegistry = Depends(get_document_registry),
) -> DocumentUploadResponse:
    """
    Retrieve details for a single ingested document by ID.

    Args:
        document_id: The document's unique identifier.

    Returns:
        A :class:`DocumentUploadResponse` describing the document.

    Raises:
        HTTPException: 404 if no document with this ID is registered.
    """
    document = registry.get(document_id)
    if document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Document '{document_id}' not found.")
    return DocumentUploadResponse.from_document(document)


@router.delete(
    "/{document_id}",
    response_model=DocumentDeleteResponse,
    summary="Delete a document and all of its stored chunks.",
)
async def delete_document(
    document_id: str,
    registry: DocumentRegistry = Depends(get_document_registry),
    vector_store: ChromaVectorStore = Depends(get_vector_store),
) -> DocumentDeleteResponse:
    """
    Delete a document's registry record and remove all of its chunks
    from the vector store.

    Args:
        document_id: The document's unique identifier.

    Returns:
        A :class:`DocumentDeleteResponse` confirming deletion.

    Raises:
        HTTPException: 404 if no document with this ID is registered.
    """
    document = registry.get(document_id)
    if document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Document '{document_id}' not found.")

    vector_store.delete_document_chunks(document_id)
    registry.delete(document_id)

    logger.info("Deleted document '%s' (document_id=%s) and its chunks.", document.source_name, document_id)
    return DocumentDeleteResponse(document_id=document_id, deleted=True)