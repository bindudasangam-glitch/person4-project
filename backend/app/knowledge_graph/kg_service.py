"""
Knowledge Graph Service
=========================

Unified, high-level entry point for knowledge-graph-backed entity and
relationship validation. Transparently prefers a real Neo4j backend when
configured and reachable, and falls back to an in-process NetworkX graph
otherwise, so the rest of Person 3's LangGraph workflow (in particular
the knowledge-graph-validation agent) never needs to know or care which
backend is actually in use.

Integration with the existing configuration system
----------------------------------------------------
This module reads its Neo4j connection settings from the application's
existing :mod:`app.core.config` ``Settings`` singleton via
:func:`app.core.config.get_settings`, following the exact same
``lru_cache``-backed singleton pattern already used throughout the
codebase (e.g. ``get_document_registry``, ``get_embedding_service``).

At the time this module was written, ``Settings`` does not yet declare
``NEO4J_URI`` / ``NEO4J_USER`` / ``NEO4J_PASSWORD`` / ``NEO4J_DATABASE`` /
``neo4j_enabled`` fields (that addition is a separate, minimal change to
``app/core/config.py``). To avoid a hard dependency ordering between
these two changes, configuration values are looked up defensively via
``getattr(settings, ..., None)`` first and an environment variable of
the same name second -- so this service works correctly *today* (falling
back to the in-memory graph, since no Neo4j settings exist yet) and will
automatically start using real Neo4j settings the moment they are added
to ``Settings``, with no further change required here.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Iterable, Optional, Union

from app.knowledge_graph.graph_fallback import InMemoryGraphFallback
from app.knowledge_graph.neo4j_client import KnowledgeGraphError, Neo4jClient

logger = logging.getLogger(__name__)

__all__ = ["KnowledgeGraphService", "get_knowledge_graph_service"]

# The two interchangeable graph backends this service can be running on.
_GraphBackend = Union[Neo4jClient, InMemoryGraphFallback]


def _normalize_entity_id(text: str) -> str:
    """
    Normalize free-text entity mentions into a stable graph node key.

    Collapses case and internal whitespace differences (e.g. "Paris",
    "paris", "  Paris  ") so the same real-world entity always maps to
    the same graph node regardless of exact surface form.

    Args:
        text: The raw entity mention text.

    Returns:
        A normalized identifier suitable for use as a graph node key.
    """
    return " ".join(text.strip().lower().split())


class KnowledgeGraphService:
    """
    High-level knowledge graph interface used by the knowledge-graph
    validation agent (and available to any other Person 3 component that
    needs entity/relationship validation).

    Handles backend selection (Neo4j vs. in-memory fallback) once, at
    construction time, and transparently degrades to the fallback
    backend if a previously-working Neo4j connection starts failing
    mid-request, so callers never need their own try/except around
    backend availability.

    Args:
        settings: Application settings instance. Defaults to the cached
            process-wide :class:`app.core.config.Settings` singleton via
            :func:`app.core.config.get_settings`.
        neo4j_client: Explicit Neo4j client to use instead of building
            one from settings. Primarily useful for tests.
        fallback: Explicit in-memory fallback graph to use. Primarily
            useful for tests that want to inspect or pre-seed the fallback
            graph directly. Defaults to a fresh, empty
            :class:`~app.knowledge_graph.graph_fallback.InMemoryGraphFallback`.
    """

    def __init__(
        self,
        settings: Any | None = None,
        neo4j_client: Optional[Neo4jClient] = None,
        fallback: Optional[InMemoryGraphFallback] = None,
    ) -> None:
        self._settings = settings if settings is not None else self._load_default_settings()
        self._fallback: InMemoryGraphFallback = fallback or InMemoryGraphFallback()
        self._neo4j_client: Neo4jClient = neo4j_client or self._build_neo4j_client_from_settings(self._settings)
        self._backend: _GraphBackend = self._select_backend()

    # ------------------------------------------------------------------ #
    # Construction / backend selection
    # ------------------------------------------------------------------ #
    @staticmethod
    def _load_default_settings() -> Any | None:
        """
        Best-effort import of the process-wide application settings.

        Isolated into its own method (rather than a bare module-level
        import) so this module has no hard import-time dependency on
        ``app.core.config`` -- useful for unit tests that construct a
        ``KnowledgeGraphService`` with explicit fakes and never need the
        real settings module loaded at all.
        """
        try:
            from app.core.config import get_settings

            return get_settings()
        except Exception:  # noqa: BLE001 - settings loading must never block KG service construction
            logger.warning(
                "Could not load application settings for KnowledgeGraphService; "
                "falling back to environment-variable-only configuration.",
                exc_info=True,
            )
            return None

    @staticmethod
    def _get_config_value(settings: Any | None, settings_attr: str, env_var: str, default: Any) -> Any:
        """
        Resolve a configuration value, preferring ``settings.<settings_attr>``
        (if present and truthy) and falling back to the environment
        variable ``env_var``, then finally ``default``.
        """
        if settings is not None and getattr(settings, settings_attr, None):
            return getattr(settings, settings_attr)
        env_value = os.getenv(env_var)
        if env_value is not None:
            return env_value
        return default

    def _build_neo4j_client_from_settings(self, settings: Any | None) -> Neo4jClient:
        """Construct a :class:`Neo4jClient` from whatever configuration is currently available."""
        uri = self._get_config_value(settings, "NEO4J_URI", "NEO4J_URI", default=None)
        user = self._get_config_value(settings, "NEO4J_USER", "NEO4J_USER", default="neo4j")
        password = self._get_config_value(settings, "NEO4J_PASSWORD", "NEO4J_PASSWORD", default=None)
        database = self._get_config_value(settings, "NEO4J_DATABASE", "NEO4J_DATABASE", default="neo4j")
        return Neo4jClient(uri=uri, user=user, password=password, database=database)

    def _is_neo4j_enabled(self) -> bool:
        """
        Return whether Neo4j should even be attempted, honoring an
        explicit opt-out (``neo4j_enabled=False`` / ``NEO4J_ENABLED=false``)
        without requiring a URI to be absent as the only way to disable it.

        Note: this deliberately does *not* go through
        :meth:`_get_config_value`, since that helper treats any falsy
        value (including an intentionally-set ``False``) as "not
        configured" and falls through to its default -- which would
        silently ignore an explicit ``neo4j_enabled=False``. Booleans
        need their own presence check via ``hasattr``/``getattr`` instead.
        """
        if self._settings is not None and hasattr(self._settings, "neo4j_enabled"):
            value = getattr(self._settings, "neo4j_enabled")
            if value is not None:
                return self._coerce_bool(value)

        env_value = os.getenv("NEO4J_ENABLED")
        if env_value is not None:
            return self._coerce_bool(env_value)

        return True

    @staticmethod
    def _coerce_bool(value: Any) -> bool:
        """Coerce a bool, or a string/other value representing one, into an actual bool."""
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() not in {"0", "false", "no", "off", ""}
        return bool(value)

    def _select_backend(self) -> _GraphBackend:
        """
        Choose the active backend at construction time: Neo4j if enabled
        and reachable, the in-memory fallback otherwise. Never raises.
        """
        if not self._is_neo4j_enabled():
            logger.info("Neo4j explicitly disabled via configuration; using in-memory fallback graph.")
            return self._fallback

        try:
            connected = self._neo4j_client.connect()
        except Exception as exc:  # noqa: BLE001 - backend selection must never crash startup
            logger.warning("Unexpected error while probing Neo4j connectivity: %s", exc)
            connected = False

        if connected:
            logger.info("KnowledgeGraphService is using the Neo4j backend.")
            return self._neo4j_client

        logger.warning(
            "Neo4j is unavailable; KnowledgeGraphService is falling back to an in-memory "
            "NetworkX graph. Knowledge graph validation will still function, but results "
            "will not persist across restarts or be shared across worker processes."
        )
        return self._fallback

    def _degrade_to_fallback(self) -> None:
        """Switch the active backend to the in-memory fallback after a live Neo4j failure."""
        if self._backend is not self._fallback:
            logger.warning("Degrading KnowledgeGraphService to the in-memory fallback after a backend failure.")
            self._backend = self._fallback

    @property
    def backend_name(self) -> str:
        """``"neo4j"`` or ``"in_memory_fallback"``, whichever backend is currently active."""
        return "neo4j" if self._backend is self._neo4j_client else "in_memory_fallback"

    @property
    def is_using_neo4j(self) -> bool:
        """True if the active backend is a live Neo4j connection."""
        return self._backend is self._neo4j_client

    def close(self) -> None:
        """Release any resources held by the Neo4j backend, if it was ever connected."""
        try:
            self._neo4j_client.close()
        except Exception:  # noqa: BLE001 - shutdown must never raise
            logger.exception("Error closing the Neo4j client during KnowledgeGraphService shutdown.")

    # ------------------------------------------------------------------ #
    # Graph mutation
    # ------------------------------------------------------------------ #
    def add_entity(
        self,
        entity_text: str,
        label: str = "Entity",
        properties: Optional[dict[str, Any]] = None,
    ) -> str:
        """
        Add (or update) an entity node for a piece of free text.

        Args:
            entity_text: The entity's display text (e.g. ``"Paris"``).
            label: Semantic label for the entity (e.g. a spaCy NER label
                such as ``"GPE"``, or any caller-defined category).
            properties: Additional properties to store on the node.

        Returns:
            The normalized entity id this entity was stored under.

        Raises:
            ValueError: If ``entity_text`` is empty or whitespace-only.
        """
        if not entity_text or not entity_text.strip():
            raise ValueError("entity_text must not be empty.")

        entity_id = _normalize_entity_id(entity_text)
        node_properties = dict(properties or {})
        node_properties.setdefault("display_text", entity_text.strip())

        try:
            self._backend.upsert_entity(entity_id, label, node_properties)
        except KnowledgeGraphError as exc:
            logger.warning(
                "Failed to upsert entity '%s' on '%s' backend: %s", entity_text, self.backend_name, exc
            )
            self._degrade_to_fallback()
            self._backend.upsert_entity(entity_id, label, node_properties)

        return entity_id

    def add_relationship(
        self,
        source_text: str,
        target_text: str,
        relation: str,
        properties: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Add (or update) a directed relationship between two entities,
        creating either endpoint entity first if it does not already exist.

        Args:
            source_text: Display text of the relationship's source entity.
            target_text: Display text of the relationship's target entity.
            relation: Relationship type name (e.g. ``"LOCATED_IN"``).
            properties: Additional properties to store on the relationship.
        """
        source_id = self.add_entity(source_text)
        target_id = self.add_entity(target_text)

        try:
            self._backend.upsert_relationship(source_id, target_id, relation, properties or {})
        except KnowledgeGraphError as exc:
            logger.warning(
                "Failed to upsert relationship '%s' -[%s]-> '%s' on '%s' backend: %s",
                source_text,
                relation,
                target_text,
                self.backend_name,
                exc,
            )
            self._degrade_to_fallback()
            self._backend.upsert_relationship(source_id, target_id, relation, properties or {})

    def ingest_entities(self, entities: Iterable[str], label: str = "Entity") -> list[str]:
        """
        Add multiple entities in one call, skipping any empty/whitespace-only strings.

        Args:
            entities: Iterable of entity display texts.
            label: Semantic label applied to every entity added this way.

        Returns:
            The list of normalized entity ids that were added.
        """
        return [self.add_entity(text, label=label) for text in entities if text and text.strip()]

    def clear(self) -> None:
        """Remove all entities and relationships from the active backend. Intended for tests."""
        self._backend.clear()

    # ------------------------------------------------------------------ #
    # Validation / reads
    # ------------------------------------------------------------------ #
    def validate_entity(self, entity_text: str) -> bool:
        """
        Return True if an entity matching this text already exists in the graph.

        Args:
            entity_text: The entity's display text to look up.
        """
        if not entity_text or not entity_text.strip():
            return False

        entity_id = _normalize_entity_id(entity_text)
        try:
            return self._backend.entity_exists(entity_id)
        except KnowledgeGraphError as exc:
            logger.warning("Entity validation failed on '%s' backend: %s", self.backend_name, exc)
            self._degrade_to_fallback()
            return self._backend.entity_exists(entity_id)

    def validate_relationship(
        self,
        source_text: str,
        target_text: str,
        relation: Optional[str] = None,
    ) -> bool:
        """
        Return True if a relationship (of the given type, if specified)
        exists from the source entity to the target entity.

        Args:
            source_text: Display text of the candidate source entity.
            target_text: Display text of the candidate target entity.
            relation: If provided, only match this specific relationship
                type; otherwise match any relationship type between the pair.
        """
        source_id = _normalize_entity_id(source_text)
        target_id = _normalize_entity_id(target_text)
        try:
            return self._backend.relationship_exists(source_id, target_id, relation)
        except KnowledgeGraphError as exc:
            logger.warning("Relationship validation failed on '%s' backend: %s", self.backend_name, exc)
            self._degrade_to_fallback()
            return self._backend.relationship_exists(source_id, target_id, relation)

    def get_relationships_between(self, source_text: str, target_text: str) -> list[str]:
        """Return the relationship type names directed from ``source_text`` to ``target_text``."""
        source_id = _normalize_entity_id(source_text)
        target_id = _normalize_entity_id(target_text)
        try:
            return self._backend.find_relationships(source_id, target_id)
        except KnowledgeGraphError as exc:
            logger.warning("Relationship lookup failed on '%s' backend: %s", self.backend_name, exc)
            self._degrade_to_fallback()
            return self._backend.find_relationships(source_id, target_id)

    def get_neighbors(self, entity_text: str) -> list[dict[str, Any]]:
        """Return the direct outgoing neighbors of an entity, with their relationship type."""
        entity_id = _normalize_entity_id(entity_text)
        try:
            return self._backend.get_neighbors(entity_id)
        except KnowledgeGraphError as exc:
            logger.warning("Neighbor lookup failed on '%s' backend: %s", self.backend_name, exc)
            self._degrade_to_fallback()
            return self._backend.get_neighbors(entity_id)

    def get_stats(self) -> dict[str, Any]:
        """Return basic graph statistics (active backend, node count, relationship count)."""
        try:
            node_count = self._backend.node_count()
            relationship_count = self._backend.relationship_count()
        except KnowledgeGraphError as exc:
            logger.warning("Stats lookup failed on '%s' backend: %s", self.backend_name, exc)
            self._degrade_to_fallback()
            node_count = self._backend.node_count()
            relationship_count = self._backend.relationship_count()

        return {
            "backend": self.backend_name,
            "node_count": node_count,
            "relationship_count": relationship_count,
        }


_service_singleton: Optional[KnowledgeGraphService] = None


def get_knowledge_graph_service() -> KnowledgeGraphService:
    """
    Return the process-wide cached :class:`KnowledgeGraphService` instance.

    Follows the same singleton pattern already used throughout the
    codebase (e.g. ``get_document_registry``, ``get_embedding_service``,
    ``get_settings``), but is implemented with a plain module-level
    variable rather than ``functools.lru_cache`` since, unlike those
    functions, this one takes no arguments to key a cache on and a
    simple singleton is clearer here.

    Returns:
        The shared :class:`KnowledgeGraphService` instance for this process.
    """
    global _service_singleton
    if _service_singleton is None:
        _service_singleton = KnowledgeGraphService()
    return _service_singleton
