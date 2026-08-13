"""
Vector store package.

Exposes :class:`ChromaVectorStore`, the low-level persistent vector
store client, along with :class:`ChromaService`, a document-level
indexing facade built on top of it.
"""

from __future__ import annotations

from app.vectorstore.chroma_client import ChromaVectorStore, VectorQueryResult
from app.vectorstore.chroma_service import ChromaService

__all__ = ["ChromaVectorStore", "VectorQueryResult", "ChromaService"]