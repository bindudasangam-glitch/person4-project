"""
Document ingestion API routes.

Heavy dependencies are imported lazily so that FastAPI startup remains
lightweight on memory-constrained environments such as Render Free.

Heavy components such as:
- SentenceTransformer
- ChromaDB
- PDF/document processing dependencies
- embedding services

are created only when their endpoints actually need them.
"""

from __future__ import annotations

import json
import logging
import shutil
import uuid
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, UploadFile, status

from app.core.config import Settings, get_settings
from app.core.exceptions import RAGFactVerificationError
from app.models.document import Document, DocumentStatus
from app.schemas.document_schema import (
    DocumentDeleteResponse,
    DocumentListResponse,
    DocumentSummary,
    DocumentUploadResponse,
)


logger = logging.getLogger(__name__)


router = APIRouter(
    prefix="/documents",
    tags=["documents"],
)


# ============================================================================
# DOCUMENT REGISTRY
# ============================================================================


class DocumentRegistry:
    """
    JSON-file-backed registry for document metadata.
    """

    def __init__(self, registry_path: Path) -> None:
        self._registry_path = registry_path
        self._lock = Lock()

        self._registry_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if not self._registry_path.exists():
            self._write_all({})

    def save(self, document: Document) -> None:
        """Persist or update a document's metadata."""

        with self._lock:
            records = self._read_all()

            document_id = str(
                document.document_id
            ).strip()

            records[document_id] = json.loads(
                document.model_dump_json()
            )

            self._write_all(records)

    def get(
        self,
        document_id: str,
    ) -> Optional[Document]:
        """
        Retrieve a document by ID.
        """

        requested_id = str(
            document_id
        ).strip()

        with self._lock:
            records = self._read_all()

        # Exact key lookup.
        raw = records.get(
            requested_id
        )

        if isinstance(raw, dict):
            try:
                return Document.model_validate(raw)
            except Exception:
                logger.exception(
                    "Invalid document registry record for key '%s'.",
                    requested_id,
                )

        # Robust fallback lookup.
        for key, record in records.items():

            if not isinstance(record, dict):
                continue

            key_id = str(key).strip()

            stored_id = str(
                record.get(
                    "document_id",
                    "",
                )
            ).strip()

            if (
                key_id == requested_id
                or stored_id == requested_id
            ):
                try:
                    return Document.model_validate(
                        record
                    )
                except Exception:
                    logger.exception(
                        "Invalid document registry record "
                        "for document_id '%s'.",
                        requested_id,
                    )
                    return None

        return None

    def list_all(self) -> list[Document]:
        """Return all registered documents."""

        with self._lock:
            records = self._read_all()

        documents: list[Document] = []

        for raw in records.values():

            if not isinstance(raw, dict):
                continue

            try:
                documents.append(
                    Document.model_validate(raw)
                )
            except Exception:
                logger.exception(
                    "Skipping invalid document registry record."
                )

        return documents

    def delete(
        self,
        document_id: str,
    ) -> bool:
        """
        Remove a document from the registry.
        """

        requested_id = str(
            document_id
        ).strip()

        with self._lock:
            records = self._read_all()

            # Exact dictionary-key match.
            if requested_id in records:
                records.pop(
                    requested_id,
                    None,
                )

                self._write_all(records)

                return True

            # Fallback lookup.
            matching_key: Optional[str] = None

            for key, record in records.items():

                if not isinstance(record, dict):
                    continue

                key_id = str(key).strip()

                stored_id = str(
                    record.get(
                        "document_id",
                        "",
                    )
                ).strip()

                if (
                    key_id == requested_id
                    or stored_id == requested_id
                ):
                    matching_key = key
                    break

            if matching_key is None:
                return False

            records.pop(
                matching_key,
                None,
            )

            self._write_all(records)

            return True

    def _read_all(self) -> dict:
        """Read all registry records."""

        try:
            content = self._registry_path.read_text(
                encoding="utf-8"
            )

            data = json.loads(content)

            if isinstance(data, dict):
                return data

            return {}

        except (
            FileNotFoundError,
            json.JSONDecodeError,
        ):
            return {}

    def _write_all(
        self,
        records: dict,
    ) -> None:
        """Write all registry records."""

        self._registry_path.write_text(
            json.dumps(
                records,
                indent=2,
            ),
            encoding="utf-8",
        )


# ============================================================================
# LIGHTWEIGHT DEPENDENCIES
# ============================================================================


@lru_cache(maxsize=1)
def get_document_registry() -> DocumentRegistry:
    """
    Return the cached document registry.

    This is lightweight and safe during startup.
    """

    settings = get_settings()

    return DocumentRegistry(
        settings.resolved_processed_dir()
        / "document_registry.json"
    )


# ============================================================================
# LAZY HEAVY DEPENDENCIES
# ============================================================================


@lru_cache(maxsize=1)
def get_document_processor() -> Any:
    """
    Lazily create the document processor.

    The heavy document-processing module is imported only when
    the upload endpoint actually requests this dependency.
    """

    from app.ingestion.document_processor import (
        DocumentProcessor,
    )

    return DocumentProcessor()


@lru_cache(maxsize=1)
def get_text_cleaner() -> Any:
    """
    Lazily create the text cleaner.
    """

    from app.processing.text_cleaner import (
        TextCleaner,
    )

    return TextCleaner()


@lru_cache(maxsize=1)
def get_text_chunker() -> Any:
    """
    Lazily create the text chunker.
    """

    from app.processing.chunker import (
        TextChunker,
    )

    return TextChunker()


@lru_cache(maxsize=1)
def get_embedding_service() -> Any:
    """
    Lazily create the embedding service.

    SentenceTransformer itself is already lazy inside
    EmbeddingService.
    """

    from app.embeddings.embedding_service import (
        EmbeddingService,
    )

    return EmbeddingService()


@lru_cache(maxsize=1)
def get_vector_store() -> Any:
    """
    Lazily create the ChromaDB vector store.
    """

    from app.vectorstore.chroma_client import (
        ChromaVectorStore,
    )

    return ChromaVectorStore()


# ============================================================================
# UPLOAD
# ============================================================================


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and fully ingest a document.",
)
async def upload_document(
    file: UploadFile,
    settings: Settings = Depends(get_settings),
    processor: Any = Depends(
        get_document_processor
    ),
    cleaner: Any = Depends(
        get_text_cleaner
    ),
    chunker: Any = Depends(
        get_text_chunker
    ),
    embedding_service: Any = Depends(
        get_embedding_service
    ),
    vector_store: Any = Depends(
        get_vector_store
    ),
    registry: DocumentRegistry = Depends(
        get_document_registry
    ),
) -> DocumentUploadResponse:
    """
    Upload and ingest a document.

    Pipeline:

        upload
        -> extraction
        -> cleaning
        -> chunking
        -> embedding
        -> ChromaDB
        -> registry
    """

    upload_dir = settings.resolved_upload_dir()

    upload_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    safe_filename = (
        f"{uuid.uuid4().hex}_"
        f"{Path(file.filename or 'upload').name}"
    )

    destination_path = (
        upload_dir / safe_filename
    )

    try:

        # ------------------------------------------------------------------
        # Save uploaded file
        # ------------------------------------------------------------------

        with destination_path.open(
            "wb"
        ) as destination_file:

            shutil.copyfileobj(
                file.file,
                destination_file,
            )

        file_size_bytes = (
            destination_path.stat().st_size
        )

        # ------------------------------------------------------------------
        # Process document
        # ------------------------------------------------------------------

        document = processor.process(
            destination_path,
            file_size_bytes,
        )

        # ------------------------------------------------------------------
        # Clean extracted text
        # ------------------------------------------------------------------

        document = cleaner.clean(
            document
        )

        # ------------------------------------------------------------------
        # Chunk
        # ------------------------------------------------------------------

        chunks = chunker.chunk_document(
            document
        )

        # ------------------------------------------------------------------
        # Embeddings
        # ------------------------------------------------------------------

        chunks = embedding_service.embed_chunks(
            chunks
        )

        # ------------------------------------------------------------------
        # Vector database
        # ------------------------------------------------------------------

        vector_store.upsert_chunks(
            chunks
        )

        # ------------------------------------------------------------------
        # Mark as indexed
        # ------------------------------------------------------------------

        document = document.model_copy(
            update={
                "status": DocumentStatus.INDEXED
            }
        )

        # ------------------------------------------------------------------
        # Save metadata
        # ------------------------------------------------------------------

        registry.save(
            document
        )

        logger.info(
            "Document '%s' fully ingested "
            "(document_id=%s, %d chunks).",
            document.source_name,
            document.document_id,
            len(chunks),
        )

        return DocumentUploadResponse.from_document(
            document
        )

    except RAGFactVerificationError as exc:

        logger.warning(
            "Document ingestion failed for '%s': %s",
            file.filename,
            exc,
        )

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    except Exception as exc:

        logger.exception(
            "Unexpected error while ingesting "
            "document '%s'.",
            file.filename,
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "Unexpected error while processing "
                f"document: {exc}"
            ),
        ) from exc


# ============================================================================
# LIST DOCUMENTS
# ============================================================================


@router.get(
    "",
    response_model=DocumentListResponse,
    summary="List all previously ingested documents.",
)
async def list_documents(
    registry: DocumentRegistry = Depends(
        get_document_registry
    ),
) -> DocumentListResponse:
    """List all registered documents."""

    documents = registry.list_all()

    summaries = [
        DocumentSummary.from_document(
            document
        )
        for document in documents
    ]

    return DocumentListResponse(
        total=len(summaries),
        documents=summaries,
    )


# ============================================================================
# GET DOCUMENT
# ============================================================================


@router.get(
    "/{document_id}",
    response_model=DocumentUploadResponse,
    summary="Retrieve details for a single ingested document.",
)
async def get_document(
    document_id: str,
    registry: DocumentRegistry = Depends(
        get_document_registry
    ),
) -> DocumentUploadResponse:
    """Retrieve one document by ID."""

    document = registry.get(
        document_id
    )

    if document is None:

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"Document '{document_id}' not found."
            ),
        )

    return DocumentUploadResponse.from_document(
        document
    )


# ============================================================================
# DELETE DOCUMENT
# ============================================================================


@router.delete(
    "/{document_id}",
    response_model=DocumentDeleteResponse,
    summary="Delete a document and its stored chunks.",
)
async def delete_document(
    document_id: str,
    registry: DocumentRegistry = Depends(
        get_document_registry
    ),
    vector_store: Any = Depends(
        get_vector_store
    ),
) -> DocumentDeleteResponse:
    """Delete document metadata and its vector chunks."""

    document = registry.get(
        document_id
    )

    if document is None:

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"Document '{document_id}' not found."
            ),
        )

    # Remove chunks from ChromaDB.
    vector_store.delete_document_chunks(
        document_id
    )

    # Remove metadata from registry.
    deleted = registry.delete(
        document_id
    )

    if not deleted:

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"Document '{document_id}' not found."
            ),
        )

    logger.info(
        "Deleted document '%s' "
        "(document_id=%s) and its chunks.",
        document.source_name,
        document_id,
    )

    return DocumentDeleteResponse(
        document_id=document_id,
        deleted=True,
    )