"""
Knowledge Graph package (Person 3).

Provides Neo4j-backed knowledge graph validation for the multi-agent
hallucination-detection workflow, with an automatic, transparent
fallback to an in-process NetworkX graph whenever Neo4j is not
configured, not installed, or not reachable.

Consumers (e.g. the LangGraph knowledge-graph-validation agent) should
depend only on :class:`~app.knowledge_graph.kg_service.KnowledgeGraphService`,
which hides the backend-selection logic entirely.
"""

from __future__ import annotations

from app.knowledge_graph.graph_fallback import InMemoryGraphFallback
from app.knowledge_graph.kg_service import KnowledgeGraphService
from app.knowledge_graph.neo4j_client import (
    KnowledgeGraphError,
    Neo4jClient,
    Neo4jConnectionError,
    Neo4jQueryError,
)

__all__ = [
    "KnowledgeGraphError",
    "Neo4jConnectionError",
    "Neo4jQueryError",
    "Neo4jClient",
    "InMemoryGraphFallback",
    "KnowledgeGraphService",
]
