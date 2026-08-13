"""
Evidence retrieval package.

Exposes :class:`Retriever`, the primary interface for retrieving ranked
evidence for a query, along with the composable building blocks
(:class:`EvidenceRetriever`, :class:`SimilaritySearch`,
:class:`SourceAttributionBuilder`) that consumers needing a custom
retrieval pipeline can assemble directly.
"""

from __future__ import annotations

from app.retrieval.evidence_retriever import EvidenceRetriever
from app.retrieval.retriever import Retriever
from app.retrieval.similarity_search import SimilaritySearch
from app.retrieval.source_attribution import SourceAttributionBuilder

__all__ = [
    "Retriever",
    "EvidenceRetriever",
    "SimilaritySearch",
    "SourceAttributionBuilder",
]