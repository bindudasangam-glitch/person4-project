"""
Knowledge Graph Validation Agent
====================================

Cross-checks the entities (and, where possible, entity relationships)
named in a response's claims against
:class:`~app.knowledge_graph.kg_service.KnowledgeGraphService`, which
transparently prefers a live Neo4j backend and automatically falls back
to an in-process NetworkX graph when Neo4j is unavailable (see
``app/knowledge_graph/kg_service.py``, Batch 1). This agent adds no
backend-selection logic of its own -- it depends only on the service's
public interface, so it works identically regardless of which backend
happens to be active.

Validation policy
-------------------
* **Entities**: every distinct named entity across all supplied claims
  is checked for existence in the graph. Entities not found are
  reported in ``unvalidated_entities`` -- this alone does *not* make the
  overall result inconsistent, since an entity simply not yet being in
  the graph is a much weaker signal than an outright contradiction (the
  graph may just be incomplete, especially on a fresh in-memory fallback).
* **Relationships**: for claims with two or more named entities, each
  adjacent pair (ordered by character position) is checked for *any*
  known relationship between them. This is a deliberately conservative,
  transparent heuristic -- like ``HallucinationDetector``'s own lexical
  approach -- since determining the *specific* semantic relationship a
  claim's sentence asserts (e.g. "located in" vs. "capital of") would
  require dependency-parse-based relation extraction outside this
  batch's scope. When a claim was already marked ``CONTRADICTED`` by
  Person 1's hallucination detector *and* the graph has no relationship
  at all between the claim's entities, that absence is recorded as
  independent, graph-based corroboration of the contradiction.
* **Consistency**: the overall result is only ever reported as
  inconsistent (``is_consistent=False``) when such a corroborated
  contradiction was found -- matching
  :class:`~app.models.workflow_models.KGValidationResult`'s own
  documented invariant.

This agent also exposes an explicit, opt-in :meth:`KnowledgeGraphAgent.seed_supported_claims`
method for growing the graph from claims Person 1 already confirmed as
``SUPPORTED`` -- validation itself never mutates the graph as a side effect.
"""

from __future__ import annotations

from app.agents.exceptions import KnowledgeGraphValidationError
from app.core.logging import logger
from app.knowledge_graph.kg_service import KnowledgeGraphService, get_knowledge_graph_service
from app.models.claim_model import ClaimModel, Entity, VerificationStatus
from app.models.workflow_models import (
    EntityValidation,
    KGValidationResult,
    RelationshipValidation,
    WorkflowModelValidationError,
)

__all__ = ["KnowledgeGraphAgent"]


class KnowledgeGraphAgent:
    """
    Validates response claims against a knowledge graph backend.

    Args:
        kg_service: The knowledge graph service to validate against.
            Defaults to the process-wide cached
            :func:`~app.knowledge_graph.kg_service.get_knowledge_graph_service`
            singleton, so repeated agent construction (e.g. one per
            workflow run) does not repeatedly re-probe Neo4j connectivity.
            Injected explicitly here (rather than only relying on the
            module-level singleton) so tests can supply a fake or a
            fresh, isolated :class:`KnowledgeGraphService`.
    """

    def __init__(self, kg_service: KnowledgeGraphService | None = None) -> None:
        self._kg_service = kg_service or get_knowledge_graph_service()

    def validate(self, claims: list[ClaimModel]) -> KGValidationResult:
        """
        Validate every entity (and adjacent entity pair) named across a
        batch of claims against the knowledge graph.

        Args:
            claims: Post-detection claims from Person 1 (each already
                carrying its final ``verification_status`` and extracted
                ``entities``).

        Returns:
            A single :class:`~app.models.workflow_models.KGValidationResult`
            aggregating every entity/relationship check performed.

        Raises:
            KnowledgeGraphValidationError: If ``claims`` is ``None``, or
                if a check against the graph fails unexpectedly (i.e. a
                failure the underlying ``KnowledgeGraphService`` could
                not itself recover from by degrading to its fallback backend).
        """
        if claims is None:
            raise KnowledgeGraphValidationError("claims must not be None.")

        if not claims:
            logger.info("KnowledgeGraphAgent: no claims supplied; returning a trivially consistent result.")
            return KGValidationResult(
                backend_used=self._kg_service.backend_name,
                is_consistent=True,
                consistency_score=1.0,
                explanation="No claims were supplied for knowledge graph validation.",
            )

        entity_checks: list[EntityValidation] = []
        relationship_checks: list[RelationshipValidation] = []
        unvalidated_entities: list[str] = []
        contradicted_relationships: list[str] = []
        seen_entity_keys: set[str] = set()

        for claim in claims:
            for entity in claim.entities:
                normalized_key = entity.text.strip().lower()
                if not normalized_key or normalized_key in seen_entity_keys:
                    continue
                seen_entity_keys.add(normalized_key)

                exists = self._safe_validate_entity(entity)
                entity_checks.append(
                    EntityValidation(entity_text=entity.text, exists_in_graph=exists, label=entity.label)
                )
                if not exists:
                    unvalidated_entities.append(entity.text)

            for source_entity, target_entity in self._extract_entity_pairs(claim):
                exists, found_relationships = self._safe_validate_relationship(source_entity, target_entity)
                relationship_checks.append(
                    RelationshipValidation(
                        source=source_entity.text,
                        target=target_entity.text,
                        relation=None,
                        exists_in_graph=exists,
                        found_relationships=tuple(found_relationships),
                    )
                )

                if claim.verification_status is VerificationStatus.CONTRADICTED and not exists:
                    contradicted_relationships.append(
                        f"Claim {claim.id}: no known relationship between "
                        f"'{source_entity.text}' and '{target_entity.text}' in the knowledge "
                        "graph, corroborating the contradiction flagged by hallucination detection."
                    )

        total_checks = len(entity_checks) + len(relationship_checks)
        validated_checks = sum(1 for e in entity_checks if e.exists_in_graph) + sum(
            1 for r in relationship_checks if r.exists_in_graph
        )
        consistency_score = round(validated_checks / total_checks, 4) if total_checks else 1.0
        is_consistent = len(contradicted_relationships) == 0

        explanation = self._build_explanation(entity_checks, relationship_checks, contradicted_relationships)

        try:
            result = KGValidationResult(
                backend_used=self._kg_service.backend_name,
                is_consistent=is_consistent,
                consistency_score=consistency_score,
                entities_checked=tuple(entity_checks),
                relationships_checked=tuple(relationship_checks),
                unvalidated_entities=tuple(unvalidated_entities),
                contradicted_relationships=tuple(contradicted_relationships),
                explanation=explanation,
            )
        except WorkflowModelValidationError as exc:
            logger.exception("KnowledgeGraphAgent produced an invalid KGValidationResult.")
            raise KnowledgeGraphValidationError("Failed to construct a valid KGValidationResult.") from exc

        logger.info(
            "KnowledgeGraphAgent validation complete: backend=%s, entities=%d, relationships=%d, "
            "is_consistent=%s, consistency_score=%.3f.",
            result.backend_used,
            len(entity_checks),
            len(relationship_checks),
            is_consistent,
            consistency_score,
        )
        return result

    def seed_supported_claims(self, claims: list[ClaimModel]) -> int:
        """
        Populate the knowledge graph with entities (and naive
        co-occurrence relationships) drawn from claims Person 1's
        hallucination detector already confirmed as ``SUPPORTED``.

        This is an explicit, opt-in step -- it is never called
        automatically by :meth:`validate` -- so callers (e.g. the
        LangGraph workflow, in a later batch) decide when it is
        appropriate to let verified response content grow the trusted
        graph, rather than this agent silently mutating shared graph
        state as a side effect of every validation call.

        Args:
            claims: Claims to seed from. Only claims with
                ``verification_status == VerificationStatus.SUPPORTED`` are used.

        Returns:
            The number of distinct entities successfully ingested.
        """
        if not claims:
            return 0

        ingested = 0
        supported_claims = [c for c in claims if c.verification_status is VerificationStatus.SUPPORTED]

        for claim in supported_claims:
            for entity in claim.entities:
                try:
                    self._kg_service.add_entity(entity.text, label=entity.label)
                    ingested += 1
                except Exception:  # noqa: BLE001 - seeding is best-effort, never fatal
                    logger.exception("KnowledgeGraphAgent failed to seed entity '%s'.", entity.text)

            for source_entity, target_entity in self._extract_entity_pairs(claim):
                try:
                    self._kg_service.add_relationship(source_entity.text, target_entity.text, "RELATED_TO")
                except Exception:  # noqa: BLE001 - seeding is best-effort, never fatal
                    logger.exception(
                        "KnowledgeGraphAgent failed to seed relationship '%s' -> '%s'.",
                        source_entity.text,
                        target_entity.text,
                    )

        logger.info(
            "KnowledgeGraphAgent seeded %d entity mention(s) from %d supported claim(s).",
            ingested,
            len(supported_claims),
        )
        return ingested

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _safe_validate_entity(self, entity: Entity) -> bool:
        """Validate a single entity, normalizing unexpected failures into ``KnowledgeGraphValidationError``."""
        try:
            return self._kg_service.validate_entity(entity.text)
        except Exception as exc:  # noqa: BLE001 - normalize to domain error
            logger.exception("KnowledgeGraphAgent failed validating entity '%s'.", entity.text)
            raise KnowledgeGraphValidationError(f"Entity validation failed for '{entity.text}'.") from exc

    def _safe_validate_relationship(self, source: Entity, target: Entity) -> tuple[bool, list[str]]:
        """Validate a single entity pair's relationship, normalizing unexpected failures."""
        try:
            exists = self._kg_service.validate_relationship(source.text, target.text)
            found_relationships = self._kg_service.get_relationships_between(source.text, target.text)
        except Exception as exc:  # noqa: BLE001 - normalize to domain error
            logger.exception(
                "KnowledgeGraphAgent failed validating relationship '%s' -> '%s'.", source.text, target.text
            )
            raise KnowledgeGraphValidationError(
                f"Relationship validation failed for '{source.text}' -> '{target.text}'."
            ) from exc
        return exists, found_relationships

    @staticmethod
    def _extract_entity_pairs(claim: ClaimModel) -> list[tuple[Entity, Entity]]:
        """
        Return adjacent entity pairs (ordered by character position)
        within a single claim, for relationship-existence checking.

        A claim with fewer than two entities yields no pairs. A claim
        with three or more entities yields ``n - 1`` consecutive pairs
        (e.g. entities [A, B, C] yields (A, B) and (B, C)), a
        conservative choice that avoids the combinatorial blow-up (and
        much higher false-positive rate) of checking every possible pair.
        """
        if len(claim.entities) < 2:
            return []
        ordered = sorted(claim.entities, key=lambda entity: entity.start_char)
        return [(ordered[i], ordered[i + 1]) for i in range(len(ordered) - 1)]

    @staticmethod
    def _build_explanation(
        entity_checks: list[EntityValidation],
        relationship_checks: list[RelationshipValidation],
        contradicted_relationships: list[str],
    ) -> str:
        """Build a short, human-readable summary of the validation pass."""
        total_entities = len(entity_checks)
        validated_entities = sum(1 for e in entity_checks if e.exists_in_graph)
        total_relationships = len(relationship_checks)
        validated_relationships = sum(1 for r in relationship_checks if r.exists_in_graph)

        entity_noun = "entity" if total_entities == 1 else "entities"
        parts = [f"{validated_entities}/{total_entities} {entity_noun} found in the knowledge graph"]

        if total_relationships:
            parts.append(
                f"{validated_relationships}/{total_relationships} entity-pair relationship(s) confirmed"
            )
        if contradicted_relationships:
            parts.append(f"{len(contradicted_relationships)} contradiction(s) corroborated by the graph")

        return "; ".join(parts) + "."
