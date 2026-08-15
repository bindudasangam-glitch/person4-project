"""
ChromaDB vector store client.

The ChromaDB persistent client is initialized lazily.

Important for Render:
- Creating ChromaVectorStore does NOT open the database.
- The Chroma client is created only when a vector-store operation
  actually needs it.
- The client is cached after first initialization.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from threading import Lock
from typing import Any, Optional

from app.core.config import Settings, get_settings
from app.core.exceptions import (
    CollectionNotFoundError,
    VectorStoreConnectionError,
    VectorStoreWriteError,
)
from app.models.chunk import Chunk

logger = logging.getLogger(__name__)

# ChromaDB HNSW index uses cosine distance.
# Similarity is calculated as 1 - distance.
_HNSW_SPACE = "cosine"


@dataclass
class VectorQueryResult:
    """
    A single raw result row returned from a similarity query.
    """

    chunk_id: str
    text: str
    similarity_score: float
    metadata: dict[str, Any]


class ChromaVectorStore:
    """
    Persistent ChromaDB-backed vector store.

    ChromaDB itself is loaded lazily so constructing this service does
    not perform database initialization.
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self._settings = settings or get_settings()

        # Do NOT initialize Chroma here.
        self._client: Any = None
        self._client_lock = Lock()

    # =====================================================================
    # LAZY CHROMA CLIENT
    # =====================================================================

    def _get_client(self) -> Any:
        """
        Lazily create and cache the persistent ChromaDB client.

        This method is the only place where PersistentClient is created.
        """

        if self._client is not None:
            return self._client

        with self._client_lock:
            if self._client is not None:
                return self._client

            persist_directory = str(
                self._settings.resolved_chroma_persist_dir()
            )

            try:
                logger.info(
                    "Initializing persistent ChromaDB at '%s'...",
                    persist_directory,
                )

                # Heavy Chroma import is delayed until it is actually needed.
                import chromadb
                from chromadb.config import Settings as ChromaSettings

                self._client = chromadb.PersistentClient(
                    path=persist_directory,
                    settings=ChromaSettings(
                        anonymized_telemetry=False,
                    ),
                )

                logger.info(
                    "Connected to persistent ChromaDB store at '%s'.",
                    persist_directory,
                )

            except Exception as exc:
                logger.exception(
                    "Failed to initialize persistent ChromaDB at '%s'.",
                    persist_directory,
                )

                raise VectorStoreConnectionError(
                    persist_directory,
                    str(exc),
                ) from exc

            return self._client

    # =====================================================================
    # COLLECTION MANAGEMENT
    # =====================================================================

    def get_or_create_collection(
        self,
        collection_name: Optional[str] = None,
    ) -> Any:
        """
        Return the named collection, creating it when necessary.
        """

        name = (
            collection_name
            or self._settings.chroma_collection_name
        )

        try:
            client = self._get_client()

            return client.get_or_create_collection(
                name=name,
                metadata={
                    "hnsw:space": _HNSW_SPACE,
                },
            )

        except VectorStoreConnectionError:
            raise

        except Exception as exc:
            raise VectorStoreConnectionError(
                str(
                    self._settings.resolved_chroma_persist_dir()
                ),
                (
                    f"failed to get or create collection "
                    f"'{name}': {exc}"
                ),
            ) from exc

    def get_collection(
        self,
        collection_name: Optional[str] = None,
    ) -> Any:
        """
        Return an existing collection without creating it.
        """

        name = (
            collection_name
            or self._settings.chroma_collection_name
        )

        try:
            client = self._get_client()

            return client.get_collection(
                name=name,
            )

        except CollectionNotFoundError:
            raise

        except Exception as exc:
            raise CollectionNotFoundError(
                name
            ) from exc

    def collection_exists(
        self,
        collection_name: Optional[str] = None,
    ) -> bool:
        """
        Check whether a collection exists.
        """

        name = (
            collection_name
            or self._settings.chroma_collection_name
        )

        client = self._get_client()

        existing_collections = client.list_collections()

        existing_names = {
            collection.name
            for collection in existing_collections
        }

        return name in existing_names

    # =====================================================================
    # WRITE OPERATIONS
    # =====================================================================

    def add_chunks(
        self,
        chunks: list[Chunk],
        collection_name: Optional[str] = None,
    ) -> int:
        """
        Add embedded chunks to the vector store.
        """

        return self._write_chunks(
            chunks,
            collection_name,
            operation="add",
        )

    def upsert_chunks(
        self,
        chunks: list[Chunk],
        collection_name: Optional[str] = None,
    ) -> int:
        """
        Add or update embedded chunks.
        """

        return self._write_chunks(
            chunks,
            collection_name,
            operation="upsert",
        )

    def _write_chunks(
        self,
        chunks: list[Chunk],
        collection_name: Optional[str],
        operation: str,
    ) -> int:
        """
        Shared implementation for add/upsert.
        """

        if not chunks:
            return 0

        missing_embeddings = [
            chunk.chunk_id
            for chunk in chunks
            if not chunk.has_embedding()
        ]

        if missing_embeddings:
            raise VectorStoreWriteError(
                operation,
                (
                    "chunks missing embeddings: "
                    f"{missing_embeddings}"
                ),
            )

        collection = self.get_or_create_collection(
            collection_name
        )

        ids = [
            chunk.chunk_id
            for chunk in chunks
        ]

        embeddings = [
            chunk.embedding
            for chunk in chunks
        ]

        documents = [
            chunk.text
            for chunk in chunks
        ]

        metadatas = [
            self._build_metadata_dict(chunk)
            for chunk in chunks
        ]

        try:
            if operation == "add":
                collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=metadatas,
                )
            else:
                collection.upsert(
                    ids=ids,
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=metadatas,
                )

        except Exception as exc:
            raise VectorStoreWriteError(
                operation,
                str(exc),
            ) from exc

        logger.info(
            "%s %d chunks into collection '%s'.",
            operation.capitalize(),
            len(chunks),
            collection.name,
        )

        return len(chunks)

    # =====================================================================
    # DELETE OPERATIONS
    # =====================================================================

    def delete_chunks(
        self,
        chunk_ids: list[str],
        collection_name: Optional[str] = None,
    ) -> None:
        """
        Delete chunks by chunk ID.
        """

        if not chunk_ids:
            return

        collection = self.get_or_create_collection(
            collection_name
        )

        try:
            collection.delete(
                ids=chunk_ids,
            )

        except Exception as exc:
            raise VectorStoreWriteError(
                "delete",
                str(exc),
            ) from exc

        logger.info(
            "Deleted %d chunks from collection '%s'.",
            len(chunk_ids),
            collection.name,
        )

    def delete_document_chunks(
        self,
        document_id: str,
        collection_name: Optional[str] = None,
    ) -> None:
        """
        Delete all chunks belonging to a document.
        """

        collection = self.get_or_create_collection(
            collection_name
        )

        try:
            collection.delete(
                where={
                    "document_id": document_id,
                },
            )

        except Exception as exc:
            raise VectorStoreWriteError(
                "delete_by_document",
                str(exc),
            ) from exc

        logger.info(
            "Deleted all chunks for document_id='%s' "
            "from collection '%s'.",
            document_id,
            collection.name,
        )

    # =====================================================================
    # SIMILARITY SEARCH
    # =====================================================================

    def similarity_search(
        self,
        query_embedding: list[float],
        top_k: Optional[int] = None,
        collection_name: Optional[str] = None,
        where: Optional[dict[str, Any]] = None,
    ) -> list[VectorQueryResult]:
        """
        Perform similarity search against ChromaDB.
        """

        effective_top_k = (
            top_k
            if top_k is not None
            else self._settings.retrieval_top_k
        )

        collection = self.get_collection(
            collection_name
        )

        try:
            raw_results = collection.query(
                query_embeddings=[
                    query_embedding
                ],
                n_results=effective_top_k,
                where=where,
            )

        except Exception as exc:
            raise VectorStoreWriteError(
                "query",
                str(exc),
            ) from exc

        return self._parse_query_results(
            raw_results
        )

    # =====================================================================
    # RESULT PARSING
    # =====================================================================

    def _parse_query_results(
        self,
        raw_results: dict[str, Any],
    ) -> list[VectorQueryResult]:
        """
        Convert ChromaDB query results into typed results.
        """

        ids = raw_results.get(
            "ids",
            [[]],
        )[0]

        documents = raw_results.get(
            "documents",
            [[]],
        )[0]

        metadatas = raw_results.get(
            "metadatas",
            [[]],
        )[0]

        distances = raw_results.get(
            "distances",
            [[]],
        )[0]

        results: list[VectorQueryResult] = []

        for (
            chunk_id,
            text,
            metadata,
            distance,
        ) in zip(
            ids,
            documents,
            metadatas,
            distances,
        ):
            similarity_score = max(
                0.0,
                min(
                    1.0,
                    1.0 - distance,
                ),
            )

            results.append(
                VectorQueryResult(
                    chunk_id=chunk_id,
                    text=text,
                    similarity_score=similarity_score,
                    metadata=dict(
                        metadata or {}
                    ),
                )
            )

        return results

    # =====================================================================
    # METADATA
    # =====================================================================

    def _build_metadata_dict(
        self,
        chunk: Chunk,
    ) -> dict[str, Any]:
        """
        Flatten structured chunk metadata into Chroma-compatible metadata.
        """

        metadata: dict[str, Any] = {
            "document_id": chunk.metadata.document_id,
            "source_name": chunk.metadata.source_name,
            "chunk_index": chunk.metadata.chunk_index,
        }

        if chunk.metadata.page_number is not None:
            metadata["page_number"] = (
                chunk.metadata.page_number
            )

        if chunk.metadata.start_char is not None:
            metadata["start_char"] = (
                chunk.metadata.start_char
            )

        if chunk.metadata.end_char is not None:
            metadata["end_char"] = (
                chunk.metadata.end_char
            )

        for key, value in chunk.metadata.extra.items():
            if isinstance(
                value,
                (
                    str,
                    int,
                    float,
                    bool,
                ),
            ):
                metadata[key] = value
            else:
                metadata[key] = str(value)

        return metadata