"""
Document indexing service.

Provides a document-level facade over
:class:`app.embeddings.embedding_service.EmbeddingService` and
:class:`app.vectorstore.chroma_client.ChromaVectorStore`, so callers
that need to index or remove a document's chunks do not have to
manually sequence "embed, then upsert" (or "delete by document ID")
themselves.
"""

from __future__ import annotations

import logging
from typing import Optional

from app.core.config import Settings, get_settings
from app.embeddings.embedding_service import EmbeddingService
from app.models.chunk import Chunk
from app.vectorstore.chroma_client import ChromaVectorStore

logger = logging.getLogger(__name__)


class ChromaService:
    """Orchestrates embedding and vector-store persistence for a document's chunks."""

    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        vector_store: Optional[ChromaVectorStore] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        """
        Args:
            embedding_service: Service used to embed chunks that do not
                already have an embedding. Defaults to a new
                :class:`EmbeddingService`.
            vector_store: Vector store to persist chunks to. Defaults to
                a new :class:`ChromaVectorStore`.
            settings: Application settings. Defaults to the cached
                process-wide settings instance.
        """
        self._settings = settings or get_settings()
        self._embedding_service = embedding_service or EmbeddingService(self._settings)
        self._vector_store = vector_store or ChromaVectorStore(self._settings)

    def index_chunks(self, chunks: list[Chunk], collection_name: Optional[str] = None) -> list[Chunk]:
        """
        Embed (as needed) and upsert a document's chunks into the vector
        store.

        Args:
            chunks: Chunks to index. Chunks that already have an
                embedding are not re-embedded.
            collection_name: Target collection name. Defaults to the
                configured application collection name.

        Returns:
            The chunks, with embeddings populated, in the same order as
            the input. Returns an empty list if ``chunks`` is empty.

        Raises:
            EmbeddingGenerationError: If embedding generation fails.
            VectorStoreWriteError: If the underlying upsert operation fails.
        """
        if not chunks:
            logger.debug("No chunks provided to index; nothing to do.")
            return []

        embedded_chunks = self._embedding_service.embed_chunks(chunks)
        self._vector_store.upsert_chunks(embedded_chunks, collection_name=collection_name)

        logger.info(
            "Indexed %d chunk(s) into collection '%s'.",
            len(embedded_chunks),
            collection_name or self._settings.chroma_collection_name,
        )
        return embedded_chunks

    def delete_document(self, document_id: str, collection_name: Optional[str] = None) -> None:
        """
        Remove all indexed chunks belonging to a document.

        Args:
            document_id: Identifier of the document whose chunks should
                be removed.
            collection_name: Target collection name. Defaults to the
                configured application collection name.

        Raises:
            VectorStoreWriteError: If the underlying delete operation fails.
        """
        self._vector_store.delete_document_chunks(document_id, collection_name=collection_name)
        logger.info("Removed indexed chunks for document_id='%s'.", document_id)

    def reindex_document(
        self, document_id: str, chunks: list[Chunk], collection_name: Optional[str] = None
    ) -> list[Chunk]:
        """
        Replace a document's indexed chunks with a new set (e.g. after
        re-processing the source document).

        Args:
            document_id: Identifier of the document being reindexed.
            chunks: The new set of chunks to index in its place.
            collection_name: Target collection name. Defaults to the
                configured application collection name.

        Returns:
            The newly indexed chunks, with embeddings populated.

        Raises:
            EmbeddingGenerationError: If embedding generation fails.
            VectorStoreWriteError: If either the delete or upsert operation fails.
        """
        self.delete_document(document_id, collection_name=collection_name)
        return self.index_chunks(chunks, collection_name=collection_name)

    def get_collection_stats(self, collection_name: Optional[str] = None) -> dict[str, object]:
        """
        Return basic diagnostic statistics for a collection.

        Args:
            collection_name: Target collection name. Defaults to the
                configured application collection name.

        Returns:
            A dictionary with the collection's ``name`` and ``chunk_count``.
            ``chunk_count`` is ``0`` if the collection does not yet exist.
        """
        name = collection_name or self._settings.chroma_collection_name

        if not self._vector_store.collection_exists(name):
            return {"name": name, "chunk_count": 0}

        collection = self._vector_store.get_collection(name)
        return {"name": name, "chunk_count": collection.count()}