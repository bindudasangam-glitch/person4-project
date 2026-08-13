"""
In-Memory Graph Fallback
==========================

A dependency-free, in-process knowledge graph backed by NetworkX
(already a transitive dependency of this project's stack). It mirrors
the subset of :class:`~app.knowledge_graph.neo4j_client.Neo4jClient`'s
public interface that :class:`~app.knowledge_graph.kg_service.KnowledgeGraphService`
depends on, so it is a transparent drop-in substitute whenever Neo4j is
not configured, not installed, or not reachable.

Design notes
------------
* Backed by ``networkx.MultiDiGraph`` so that, in principle, more than
  one distinct relationship type can exist between the same pair of
  entities (e.g. both a ``LOCATED_IN`` and a ``PART_OF`` edge), matching
  Neo4j's own relationship model rather than collapsing them to a single
  edge.
* Entirely in-process and in-memory: it does *not* persist across
  restarts and is *not* shared across multiple worker processes. This
  is an explicit, documented trade-off of the fallback path -- the
  primary goal is that knowledge-graph validation keeps functioning
  (degrading gracefully) rather than becoming a hard dependency on a
  running Neo4j instance.
* Method signatures intentionally match :class:`Neo4jClient` wherever
  both exist, so :class:`~app.knowledge_graph.kg_service.KnowledgeGraphService`
  can call either backend without an ``if`` branch at each call site.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import networkx as nx

logger = logging.getLogger(__name__)

__all__ = ["InMemoryGraphFallback"]


class InMemoryGraphFallback:
    """
    In-process, NetworkX-backed knowledge graph used as the automatic
    fallback backend when Neo4j is unavailable.

    Unlike :class:`~app.knowledge_graph.neo4j_client.Neo4jClient`, this
    backend is always "available" once constructed -- there is no
    external server to be unreachable -- so :meth:`connect` always
    succeeds and :attr:`is_connected` is always ``True``.
    """

    def __init__(self) -> None:
        self._graph: nx.MultiDiGraph = nx.MultiDiGraph()
        logger.info("Initialized in-memory NetworkX fallback knowledge graph.")

    @property
    def is_connected(self) -> bool:
        """Always True: an in-process graph has no connectivity to lose."""
        return True

    def connect(self) -> bool:
        """No-op connect for interface parity with :class:`Neo4jClient`. Always succeeds."""
        return True

    def close(self) -> None:
        """No-op close for interface parity with :class:`Neo4jClient`. Nothing to release."""
        return None

    # ------------------------------------------------------------------ #
    # Entity / relationship mutation
    # ------------------------------------------------------------------ #
    def upsert_entity(self, entity_id: str, label: str, properties: Optional[dict[str, Any]] = None) -> None:
        """
        Create or update a node representing an entity.

        Args:
            entity_id: Stable, normalized identifier for the entity.
            label: Free-form semantic label (e.g. ``"GPE"``, ``"PERSON"``).
            properties: Additional attributes to merge onto the node.
        """
        attributes = dict(properties or {})
        attributes["label"] = label
        if self._graph.has_node(entity_id):
            self._graph.nodes[entity_id].update(attributes)
        else:
            self._graph.add_node(entity_id, **attributes)

    def upsert_relationship(
        self,
        from_id: str,
        to_id: str,
        rel_type: str,
        properties: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Create or update a directed, typed relationship between two entities.

        Both endpoint entities are auto-created with a placeholder label
        if they do not already exist, mirroring the practical effect of
        Neo4j's ``MERGE`` (though :class:`Neo4jClient` itself requires
        both endpoints to pre-exist and raises otherwise -- this fallback
        favors availability over strict referential validation, since it
        has no separate constraint-enforcement layer to lean on).

        Args:
            from_id: Source entity id.
            to_id: Target entity id.
            rel_type: Relationship type name. Used verbatim as the
                MultiDiGraph edge key, so re-upserting the same
                ``(from_id, to_id, rel_type)`` triple updates the
                existing edge's properties rather than creating a
                duplicate parallel edge.
            properties: Additional attributes to merge onto the relationship.
        """
        if not self._graph.has_node(from_id):
            self.upsert_entity(from_id, label="Unknown")
        if not self._graph.has_node(to_id):
            self.upsert_entity(to_id, label="Unknown")

        attributes = dict(properties or {})
        attributes["relationship"] = rel_type
        self._graph.add_edge(from_id, to_id, key=rel_type, **attributes)

    def clear(self) -> None:
        """Remove every node and edge from the graph. Intended for tests."""
        self._graph.clear()
        logger.info("Cleared in-memory fallback knowledge graph.")

    # ------------------------------------------------------------------ #
    # Reads
    # ------------------------------------------------------------------ #
    def entity_exists(self, entity_id: str) -> bool:
        """Return True if a node with this entity id exists."""
        return self._graph.has_node(entity_id)

    def relationship_exists(self, from_id: str, to_id: str, rel_type: Optional[str] = None) -> bool:
        """
        Return True if a directed relationship exists from ``from_id`` to ``to_id``.

        Args:
            from_id: Source entity id.
            to_id: Target entity id.
            rel_type: If provided, only match this specific relationship
                type; otherwise match any relationship type between the pair.
        """
        if not self._graph.has_edge(from_id, to_id):
            return False
        if rel_type is None:
            return True
        edge_data = self._graph.get_edge_data(from_id, to_id)
        return any(
            key == rel_type or data.get("relationship") == rel_type for key, data in edge_data.items()
        )

    def get_neighbors(self, entity_id: str) -> list[dict[str, Any]]:
        """Return the direct outgoing neighbors of an entity, with their relationship type."""
        if not self._graph.has_node(entity_id):
            return []
        neighbors: list[dict[str, Any]] = []
        for _, target, data in self._graph.out_edges(entity_id, data=True):
            neighbors.append(
                {
                    "entity_id": target,
                    "label": self._graph.nodes[target].get("label", "Unknown"),
                    "relationship": data.get("relationship", "RELATED_TO"),
                }
            )
        return neighbors

    def find_relationships(self, from_id: str, to_id: str) -> list[str]:
        """Return the type names of all relationships directed from ``from_id`` to ``to_id``."""
        if not self._graph.has_edge(from_id, to_id):
            return []
        edge_data = self._graph.get_edge_data(from_id, to_id)
        return [data.get("relationship", key) for key, data in edge_data.items()]

    def node_count(self) -> int:
        """Return the total number of entity nodes in the graph."""
        return self._graph.number_of_nodes()

    def relationship_count(self) -> int:
        """Return the total number of relationship edges in the graph."""
        return self._graph.number_of_edges()
