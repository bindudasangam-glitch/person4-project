"""
Neo4j Client
=============

A thin wrapper around the official ``neo4j`` Python driver, exposing only
the small set of graph operations the knowledge-graph-validation agent
needs: upserting entities/relationships, checking existence, walking
neighbors, and basic graph statistics.

Design notes
------------
* The ``neo4j`` package is an *optional* runtime dependency from this
  module's point of view: importing it is wrapped in a ``try/except`` so
  that this file (and everything that imports it) can be imported even
  in an environment where the driver package has not been installed yet
  or Neo4j is not deployed. :class:`Neo4jClient` simply reports itself
  as unavailable in that case, and callers (see
  ``app.knowledge_graph.kg_service.KnowledgeGraphService``) are expected
  to fall back to :class:`~app.knowledge_graph.graph_fallback.InMemoryGraphFallback`.
* Connection failures are never allowed to raise out of :meth:`connect`
  -- they are logged and reported via a boolean return value instead,
  mirroring this project's existing "best-effort startup" pattern (see
  the lifespan hook in ``app/main.py`` pre-warming the NLP pipeline).
* Query-level failures (a reachable server that nonetheless rejects a
  query) *do* raise, via :class:`Neo4jQueryError`, so callers can
  distinguish "never connected" from "connected but this call failed"
  and decide how to react (e.g. degrade to the fallback backend).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

try:
    from neo4j import Driver, GraphDatabase
    from neo4j.exceptions import AuthError, Neo4jError, ServiceUnavailable

    _NEO4J_DRIVER_INSTALLED = True
except ImportError:  # pragma: no cover - exercised only when neo4j isn't installed
    Driver = None  # type: ignore[assignment, misc]
    GraphDatabase = None  # type: ignore[assignment]

    class Neo4jError(Exception):  # type: ignore[no-redef]
        """Placeholder standing in for ``neo4j.exceptions.Neo4jError`` when the driver isn't installed."""

    class ServiceUnavailable(Neo4jError):  # type: ignore[no-redef]
        """Placeholder standing in for ``neo4j.exceptions.ServiceUnavailable`` when the driver isn't installed."""

    class AuthError(Neo4jError):  # type: ignore[no-redef]
        """Placeholder standing in for ``neo4j.exceptions.AuthError`` when the driver isn't installed."""

    _NEO4J_DRIVER_INSTALLED = False

__all__ = [
    "KnowledgeGraphError",
    "Neo4jConnectionError",
    "Neo4jQueryError",
    "Neo4jClient",
]


class KnowledgeGraphError(Exception):
    """Base class for all Person 3 knowledge-graph domain exceptions."""


class Neo4jConnectionError(KnowledgeGraphError):
    """Raised when an operation is attempted against a Neo4j client that isn't connected."""


class Neo4jQueryError(KnowledgeGraphError):
    """Raised when a query against a reachable Neo4j server fails."""


class Neo4jClient:
    """
    Minimal Neo4j graph client for entity/relationship upserts and reads.

    All entities are stored as nodes labeled ``Entity`` with a unique
    ``entity_id`` property (a normalized, lowercased form of the entity's
    display text) plus a free-form ``label`` property describing its
    semantic type (e.g. ``"GPE"``, ``"PERSON"``, ``"ORG"`` — reusing
    spaCy's entity label vocabulary from Person 1's ``ClaimModel`` where
    convenient, though this client itself has no dependency on it).
    Relationships are stored as directed, typed Neo4j relationships
    between two ``Entity`` nodes.

    Args:
        uri: The Neo4j connection URI (e.g. ``"bolt://localhost:7687"``
            or ``"neo4j+s://<host>"``). If falsy, :meth:`connect` reports
            unavailability without attempting a connection.
        user: Username for basic auth. May be ``None`` for unauthenticated
            deployments.
        password: Password for basic auth. May be ``None`` for
            unauthenticated deployments.
        database: Target Neo4j database name. Defaults to ``"neo4j"``,
            the default database name in Neo4j 4.0+.
        connection_timeout: Seconds to wait for the initial connectivity
            check before giving up and reporting unavailability.
    """

    def __init__(
        self,
        uri: Optional[str],
        user: Optional[str] = None,
        password: Optional[str] = None,
        database: str = "neo4j",
        connection_timeout: float = 5.0,
    ) -> None:
        self._uri = uri
        self._user = user
        self._password = password
        self._database = database
        self._connection_timeout = connection_timeout
        self._driver: Optional[Driver] = None
        self._connected = False

    @property
    def is_connected(self) -> bool:
        """True if this client currently holds a verified connection to Neo4j."""
        return self._connected

    @property
    def is_driver_installed(self) -> bool:
        """True if the ``neo4j`` driver package is importable in this environment."""
        return _NEO4J_DRIVER_INSTALLED

    def connect(self) -> bool:
        """
        Attempt to establish and verify a connection to Neo4j.

        Never raises: any failure (missing driver package, no URI
        configured, unreachable server, bad credentials, or any other
        unexpected error) is logged and reported via the return value,
        so callers can safely try this as a best-effort probe at startup.

        Returns:
            ``True`` if a connection was established and verified,
            ``False`` otherwise.
        """
        if not _NEO4J_DRIVER_INSTALLED:
            logger.warning(
                "The 'neo4j' driver package is not installed; cannot connect to Neo4j. "
                "Add 'neo4j' to requirements.txt to enable the Neo4j backend."
            )
            self._connected = False
            return False

        if not self._uri:
            logger.info("No Neo4j URI configured; skipping Neo4j connection attempt.")
            self._connected = False
            return False

        try:
            auth = (self._user, self._password) if self._user else None
            self._driver = GraphDatabase.driver(
                self._uri,
                auth=auth,
                connection_timeout=self._connection_timeout,
            )
            self._driver.verify_connectivity()
            self._connected = True
            logger.info("Connected to Neo4j at '%s' (database='%s').", self._uri, self._database)
        except (ServiceUnavailable, AuthError, Neo4jError, OSError, ValueError) as exc:
            logger.warning("Could not connect to Neo4j at '%s': %s", self._uri, exc)
            self._driver = None
            self._connected = False
        except Exception as exc:  # noqa: BLE001 - connection probing must never crash the app
            logger.warning("Unexpected error connecting to Neo4j at '%s': %s", self._uri, exc)
            self._driver = None
            self._connected = False

        return self._connected

    def close(self) -> None:
        """Release the underlying driver's connection pool, if one is held."""
        if self._driver is not None:
            try:
                self._driver.close()
            except Exception:  # noqa: BLE001 - shutdown must never raise
                logger.exception("Error while closing the Neo4j driver.")
            finally:
                self._driver = None
                self._connected = False

    # ------------------------------------------------------------------ #
    # Low-level query execution
    # ------------------------------------------------------------------ #
    def _run(self, query: str, parameters: Optional[dict[str, Any]] = None) -> list[dict[str, Any]]:
        """
        Execute a Cypher query and return its records as plain dicts.

        Args:
            query: The Cypher query text.
            parameters: Query parameters.

        Returns:
            A list of result records, each as a plain dict.

        Raises:
            Neo4jConnectionError: If this client is not currently connected.
            Neo4jQueryError: If the query itself fails against a reachable server.
        """
        if not self._connected or self._driver is None:
            raise Neo4jConnectionError("Neo4j client is not connected; call connect() first.")

        try:
            with self._driver.session(database=self._database) as session:
                result = session.run(query, parameters or {})
                return [record.data() for record in result]
        except (Neo4jError, ServiceUnavailable) as exc:
            raise Neo4jQueryError(f"Neo4j query failed: {exc}") from exc

    # ------------------------------------------------------------------ #
    # Entity / relationship mutation
    # ------------------------------------------------------------------ #
    def upsert_entity(self, entity_id: str, label: str, properties: Optional[dict[str, Any]] = None) -> None:
        """
        Create or update an ``Entity`` node.

        Args:
            entity_id: Stable, normalized identifier for the entity.
            label: Free-form semantic label (e.g. ``"GPE"``, ``"PERSON"``).
            properties: Additional properties to set on the node.
        """
        query = "MERGE (e:Entity {entity_id: $entity_id}) SET e.label = $label, e += $properties"
        self._run(query, {"entity_id": entity_id, "label": label, "properties": properties or {}})

    def upsert_relationship(
        self,
        from_id: str,
        to_id: str,
        rel_type: str,
        properties: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Create or update a directed relationship between two existing entities.

        Args:
            from_id: Source entity id.
            to_id: Target entity id.
            rel_type: Relationship type name (sanitized into a valid,
                upper-cased Cypher relationship type token).
            properties: Additional properties to set on the relationship.

        Raises:
            Neo4jQueryError: If either endpoint entity does not already exist.
        """
        safe_rel_type = self._sanitize_relationship_type(rel_type)
        query = (
            "MATCH (a:Entity {entity_id: $from_id}) "
            "MATCH (b:Entity {entity_id: $to_id}) "
            f"MERGE (a)-[r:{safe_rel_type}]->(b) "
            "SET r += $properties"
        )
        self._run(query, {"from_id": from_id, "to_id": to_id, "properties": properties or {}})

    def clear(self) -> None:
        """Delete every ``Entity`` node (and their relationships). Intended for tests."""
        self._run("MATCH (n:Entity) DETACH DELETE n")

    # ------------------------------------------------------------------ #
    # Reads
    # ------------------------------------------------------------------ #
    def entity_exists(self, entity_id: str) -> bool:
        """Return True if an ``Entity`` node with this id exists."""
        records = self._run(
            "MATCH (e:Entity {entity_id: $entity_id}) RETURN count(e) AS count",
            {"entity_id": entity_id},
        )
        return bool(records) and records[0]["count"] > 0

    def relationship_exists(self, from_id: str, to_id: str, rel_type: Optional[str] = None) -> bool:
        """
        Return True if a directed relationship exists from ``from_id`` to ``to_id``.

        Args:
            from_id: Source entity id.
            to_id: Target entity id.
            rel_type: If provided, only match this specific relationship
                type; otherwise match any relationship type.
        """
        if rel_type:
            safe_rel_type = self._sanitize_relationship_type(rel_type)
            query = (
                "MATCH (a:Entity {entity_id: $from_id})-"
                f"[r:{safe_rel_type}]->"
                "(b:Entity {entity_id: $to_id}) RETURN count(r) AS count"
            )
        else:
            query = (
                "MATCH (a:Entity {entity_id: $from_id})-[r]->(b:Entity {entity_id: $to_id}) "
                "RETURN count(r) AS count"
            )
        records = self._run(query, {"from_id": from_id, "to_id": to_id})
        return bool(records) and records[0]["count"] > 0

    def get_neighbors(self, entity_id: str) -> list[dict[str, Any]]:
        """Return the direct outgoing neighbors of an entity, with their relationship type."""
        query = (
            "MATCH (a:Entity {entity_id: $entity_id})-[r]->(b:Entity) "
            "RETURN b.entity_id AS entity_id, b.label AS label, type(r) AS relationship"
        )
        return self._run(query, {"entity_id": entity_id})

    def find_relationships(self, from_id: str, to_id: str) -> list[str]:
        """Return the type names of all relationships directed from ``from_id`` to ``to_id``."""
        records = self._run(
            "MATCH (a:Entity {entity_id: $from_id})-[r]->(b:Entity {entity_id: $to_id}) "
            "RETURN type(r) AS relationship",
            {"from_id": from_id, "to_id": to_id},
        )
        return [record["relationship"] for record in records]

    def node_count(self) -> int:
        """Return the total number of ``Entity`` nodes in the graph."""
        records = self._run("MATCH (e:Entity) RETURN count(e) AS count")
        return records[0]["count"] if records else 0

    def relationship_count(self) -> int:
        """Return the total number of relationships between ``Entity`` nodes in the graph."""
        records = self._run("MATCH (:Entity)-[r]->(:Entity) RETURN count(r) AS count")
        return records[0]["count"] if records else 0

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _sanitize_relationship_type(rel_type: str) -> str:
        """
        Convert an arbitrary string into a syntactically valid, safe Cypher
        relationship type token (upper snake case, no injection risk since
        Cypher does not support parameterized relationship/label types).
        """
        cleaned = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in rel_type.strip().upper())
        cleaned = cleaned.strip("_") or "RELATED_TO"
        if cleaned[0].isdigit():
            cleaned = f"REL_{cleaned}"
        return cleaned
