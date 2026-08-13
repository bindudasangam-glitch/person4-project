"""
ChromaDB vector store client.

Wraps ChromaDB's persistent client behind a small, typed interface
focused on the operations this application actually needs: collection
management, adding/upserting/deleting chunk embeddings, and similarity
search. Keeping this wrapper thin (rather than exposing the raw
ChromaDB client everywhere) means the rest of the codebase depends on a
stable interface even if the underlying vector database is swapped out
later.

Dependencies:
    pip install chromadb
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings as ChromaSettings

from app.core.config import Settings, get_settings
from app.core.exceptions import (
    CollectionNotFoundError,
    VectorStoreConnectionError,
    VectorStoreWriteError,
)
from app.models.chunk import Chunk

logger = logging.getLogger(__name__)

# ChromaDB's HNSW index is configured to use cosine distance, so that
# similarity scores can be derived consistently as (1 - distance).
_HNSW_SPACE = "cosine"


@dataclass
class VectorQueryResult:
    """
    A single raw result row returned from a similarity query, before it
    has been assembled into an :class:`app.models.evidence.Evidence`.
    """

    chunk_id: str
    text: str
    similarity_score: float
    metadata: dict[str, Any]


class ChromaVectorStore:
    """Persistent ChromaDB-backed vector store for document chunks."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        """
        Args:
            settings: Application settings providing the ChromaDB
                persistence directory and default collection name.
                Defaults to the cached process-wide settings instance.

        Raises:
            VectorStoreConnectionError: If the persistent ChromaDB client
                cannot be initialized.
        """
        self._settings = settings or get_settings()
        self._client = self._create_client()

    def _create_client(self) -> "chromadb.api.ClientAPI":
        """Initialize the persistent ChromaDB client."""
        persist_directory = str(self._settings.resolved_chroma_persist_dir())
        try:
            client = chromadb.PersistentClient(
                path=persist_directory,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
        except Exception as exc:  # noqa: BLE001 - surface any connection failure uniformly
            raise VectorStoreConnectionError(persist_directory, str(exc)) from exc

        logger.info("Connected to persistent ChromaDB store at '%s'.", persist_directory)
        return client

    def get_or_create_collection(self, collection_name: Optional[str] = None) -> Collection:
        """
        Return the named collection, creating it (with cosine distance
        configured) if it does not already exist.

        Args:
            collection_name: Name of the collection. Defaults to the
                configured application collection name.

        Returns:
            The ChromaDB :class:`Collection` handle.

        Raises:
            VectorStoreConnectionError: If the collection cannot be
                created or retrieved.
        """
        name = collection_name or self._settings.chroma_collection_name
        try:
            return self._client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": _HNSW_SPACE},
            )
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreConnectionError(
                str(self._settings.resolved_chroma_persist_dir()),
                f"failed to get or create collection '{name}': {exc}",
            ) from exc

    def get_collection(self, collection_name: Optional[str] = None) -> Collection:
        """
        Return an existing collection without creating it.

        Args:
            collection_name: Name of the collection. Defaults to the
                configured application collection name.

        Returns:
            The ChromaDB :class:`Collection` handle.

        Raises:
            CollectionNotFoundError: If no collection with this name exists.
        """
        name = collection_name or self._settings.chroma_collection_name
        try:
            return self._client.get_collection(name=name)
        except Exception as exc:  # noqa: BLE001 - ChromaDB raises a generic error for missing collections
            raise CollectionNotFoundError(name) from exc

    def collection_exists(self, collection_name: Optional[str] = None) -> bool:
        """
        Check whether a collection with the given name exists.

        Args:
            collection_name: Name of the collection. Defaults to the
                configured application collection name.

        Returns:
            True if the collection exists, False otherwise.
        """
        name = collection_name or self._settings.chroma_collection_name
        existing_names = {collection.name for collection in self._client.list_collections()}
        return name in existing_names

    def add_chunks(self, chunks: list[Chunk], collection_name: Optional[str] = None) -> int:
        """
        Add embedded chunks to the vector store. Chunks without an
        embedding are rejected.

        Args:
            chunks: Chunks to add. Each must already have an ``embedding``
                populated (see :class:`app.embeddings.EmbeddingService`).
            collection_name: Target collection name. Defaults to the
                configured application collection name.

        Returns:
            The number of chunks added.

        Raises:
            VectorStoreWriteError: If any chunk lacks an embedding, or if
                the underlying ChromaDB add operation fails.
        """
        return self._write_chunks(chunks, collection_name, operation="add")

    def upsert_chunks(self, chunks: list[Chunk], collection_name: Optional[str] = None) -> int:
        """
        Add or update embedded chunks in the vector store, replacing any
        existing entries that share the same chunk ID.

        Since :class:`Chunk` IDs are deterministic (derived from
        document_id and chunk_index), re-processing an unchanged
        document is idempotent when using this method.

        Args:
            chunks: Chunks to upsert. Each must already have an
                ``embedding`` populated.
            collection_name: Target collection name. Defaults to the
                configured application collection name.

        Returns:
            The number of chunks upserted.

        Raises:
            VectorStoreWriteError: If any chunk lacks an embedding, or if
                the underlying ChromaDB upsert operation fails.
        """
        return self._write_chunks(chunks, collection_name, operation="upsert")

    def _write_chunks(self, chunks: list[Chunk], collection_name: Optional[str], operation: str) -> int:
        """Shared implementation for ``add_chunks`` and ``upsert_chunks``."""
        if not chunks:
            return 0

        missing_embeddings = [chunk.chunk_id for chunk in chunks if not chunk.has_embedding()]
        if missing_embeddings:
            raise VectorStoreWriteError(
                operation, f"chunks missing embeddings: {missing_embeddings}"
            )

        collection = self.get_or_create_collection(collection_name)

        ids = [chunk.chunk_id for chunk in chunks]
        embeddings = [chunk.embedding for chunk in chunks]
        documents = [chunk.text for chunk in chunks]
        metadatas = [self._build_metadata_dict(chunk) for chunk in chunks]

        try:
            if operation == "add":
                collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
            else:
                collection.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreWriteError(operation, str(exc)) from exc

        logger.info("%s %d chunks into collection '%s'.", operation.capitalize(), len(chunks), collection.name)
        return len(chunks)

    def delete_chunks(self, chunk_ids: list[str], collection_name: Optional[str] = None) -> None:
        """
        Delete chunks from the vector store by their chunk IDs.

        Args:
            chunk_ids: Chunk IDs to delete.
            collection_name: Target collection name. Defaults to the
                configured application collection name.

        Raises:
            VectorStoreWriteError: If the delete operation fails.
        """
        if not chunk_ids:
            return

        collection = self.get_or_create_collection(collection_name)
        try:
            collection.delete(ids=chunk_ids)
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreWriteError("delete", str(exc)) from exc

        logger.info("Deleted %d chunks from collection '%s'.", len(chunk_ids), collection.name)

    def delete_document_chunks(self, document_id: str, collection_name: Optional[str] = None) -> None:
        """
        Delete all chunks belonging to a specific document.

        Args:
            document_id: Identifier of the document whose chunks should
                be removed.
            collection_name: Target collection name. Defaults to the
                configured application collection name.

        Raises:
            VectorStoreWriteError: If the delete operation fails.
        """
        collection = self.get_or_create_collection(collection_name)
        try:
            collection.delete(where={"document_id": document_id})
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreWriteError("delete_by_document", str(exc)) from exc

        logger.info("Deleted all chunks for document_id='%s' from collection '%s'.", document_id, collection.name)

    def similarity_search(
        self,
        query_embedding: list[float],
        top_k: Optional[int] = None,
        collection_name: Optional[str] = None,
        where: Optional[dict[str, Any]] = None,
    ) -> list[VectorQueryResult]:
        """
        Perform a similarity search against the vector store.

        Args:
            query_embedding: The dense embedding vector for the query.
            top_k: Number of results to return. Defaults to the
                configured application retrieval top-k.
            collection_name: Target collection name. Defaults to the
                configured application collection name.
            where: Optional ChromaDB metadata filter, e.g.
                ``{"document_id": "abc123"}``.

        Returns:
            A list of :class:`VectorQueryResult` ordered by descending
            similarity score.

        Raises:
            CollectionNotFoundError: If the target collection does not exist.
            VectorStoreWriteError: If the underlying query operation fails.
        """
        effective_top_k = top_k if top_k is not None else self._settings.retrieval_top_k
        collection = self.get_collection(collection_name)

        try:
            raw_results = collection.query(
                query_embeddings=[query_embedding],
                n_results=effective_top_k,
                where=where,
            )
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreWriteError("query", str(exc)) from exc

        return self._parse_query_results(raw_results)

    def _parse_query_results(self, raw_results: dict[str, Any]) -> list[VectorQueryResult]:
        """
        Convert ChromaDB's raw query response into a list of
        :class:`VectorQueryResult` instances, converting cosine distance
        into a cosine similarity score.

        Args:
            raw_results: The raw dictionary returned by ``collection.query``.

        Returns:
            A list of parsed, similarity-scored results.
        """
        ids = raw_results.get("ids", [[]])[0]
        documents = raw_results.get("documents", [[]])[0]
        metadatas = raw_results.get("metadatas", [[]])[0]
        distances = raw_results.get("distances", [[]])[0]

        results: list[VectorQueryResult] = []
        for chunk_id, text, metadata, distance in zip(ids, documents, metadatas, distances):
            similarity_score = max(0.0, min(1.0, 1.0 - distance))
            results.append(
                VectorQueryResult(
                    chunk_id=chunk_id,
                    text=text,
                    similarity_score=similarity_score,
                    metadata=dict(metadata or {}),
                )
            )
        return results

    def _build_metadata_dict(self, chunk: Chunk) -> dict[str, Any]:
        """
        Flatten a chunk's structured metadata into the plain string/number
        dictionary that ChromaDB requires for stored metadata.

        Args:
            chunk: The chunk whose metadata should be serialized.

        Returns:
            A ChromaDB-compatible metadata dictionary.
        """
        metadata: dict[str, Any] = {
            "document_id": chunk.metadata.document_id,
            "source_name": chunk.metadata.source_name,
            "chunk_index": chunk.metadata.chunk_index,
        }
        if chunk.metadata.page_number is not None:
            metadata["page_number"] = chunk.metadata.page_number
        if chunk.metadata.start_char is not None:
            metadata["start_char"] = chunk.metadata.start_char
        if chunk.metadata.end_char is not None:
            metadata["end_char"] = chunk.metadata.end_char

        for key, value in chunk.metadata.extra.items():
            if isinstance(value, (str, int, float, bool)):
                metadata[key] = value
            else:
                metadata[key] = str(value)

        return metadata