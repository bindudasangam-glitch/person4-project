"""
Workflow Domain Models
=========================

Defines the structured output objects produced by each Person 3 agent
in the LangGraph multi-agent workflow:

* :class:`CorrectionResult`   -- output of the self-correction agent.
* :class:`KGValidationResult` -- output of the knowledge-graph-validation agent.
* :class:`DriftReport`        -- output of the knowledge drift detection agent.
* :class:`RiskAssessment`     -- output of the risk analysis module.
* :class:`WorkflowResult`     -- the final aggregation of all of the above,
  returned by ``app.agents.workflow`` and serialized by the ``/workflow``
  API route added in a later batch.

Design notes
------------
* Modeled as plain ``dataclass`` types, not Pydantic ``BaseModel``,
  matching the precedent set by Person 1's ``app/models/claim_model.py``
  (see that module's own design-notes docstring): these are internal
  domain objects constructed and consumed entirely within Person 3's
  own code (agents -> ``WorkflowState`` -> ``WorkflowResult``), so they
  don't need Pydantic's request/response validation machinery.
  API-boundary validation, when the workflow route is added, is the
  responsibility of a separate ``app.schemas`` model, exactly as Person
  1's ``ClaimModel`` (dataclass) vs. ``app.schemas.claim.Claim``
  (Pydantic) are kept separate. Person 2's flow-through models
  (``Claim``, ``VerificationResult``) use Pydantic instead because they
  are constructed directly from API-adjacent, less-trusted inputs;
  Person 3's models here are constructed only from already-validated
  internal agent outputs, so that justification does not apply.
* Every model validates its own invariants in ``__post_init__``, exactly
  like ``ClaimModel`` / ``Entity`` do, so an inconsistent instance (e.g.
  ``is_consistent=True`` alongside a non-empty ``contradicted_relationships``)
  can never exist in memory.
* Every model exposes ``to_dict()`` for direct, explicit JSON-compatible
  serialization, matching ``ClaimModel.to_dict()`` /
  ``ResponseAnalysis.to_dict()``, rather than relying on callers knowing
  an alternate (Pydantic-only) serialization path.
* Collections are stored as immutable ``tuple`` fields (never mutable
  ``list``), matching ``ClaimModel.entities`` / ``ClaimModel.evidence``,
  so a `WorkflowResult` (or any nested result) can't be mutated out from
  under callers after construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from app.core.logging import logger

__all__ = [
    "WorkflowModelValidationError",
    "SeverityLevel",
    "CorrectionAction",
    "ClaimCorrection",
    "CorrectionResult",
    "EntityValidation",
    "RelationshipValidation",
    "KGValidationResult",
    "DriftedClaim",
    "DriftReport",
    "RiskFactor",
    "RiskAssessment",
    "WorkflowResult",
]


class WorkflowModelValidationError(ValueError):
    """Raised when a Person 3 workflow model is constructed with invalid or inconsistent data."""


class SeverityLevel(str, Enum):
    """
    Coarse-grained severity/risk categorization shared by
    :class:`RiskAssessment` and :class:`DriftReport`, so both agents
    report their findings on the same scale rather than each inventing
    an incompatible one.
    """

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class CorrectionAction(str, Enum):
    """The action the self-correction agent took for a single claim."""

    NONE = "none"
    REWORDED = "reworded"
    REPLACED = "replaced"
    REMOVED = "removed"
    FLAGGED = "flagged"


def _validate_unit_interval(value: float, field_name: str, owner: str) -> None:
    """Shared helper: raise if ``value`` is not within the closed interval [0.0, 1.0]."""
    if not 0.0 <= value <= 1.0:
        raise WorkflowModelValidationError(
            f"{owner}.{field_name} must be within [0.0, 1.0], got {value}."
        )


def _validate_non_empty(value: str, field_name: str, owner: str) -> None:
    """Shared helper: raise if ``value`` is empty or whitespace-only."""
    if not value or not value.strip():
        raise WorkflowModelValidationError(f"{owner}.{field_name} must not be empty.")


# --------------------------------------------------------------------------- #
# Self-correction
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ClaimCorrection:
    """
    A single per-claim correction action taken (or explicitly not taken)
    by the self-correction agent.

    Attributes:
        claim_id: Identifier of the claim this correction applies to
            (matches Person 1's ``ClaimModel.id``).
        original_text: The claim's original text before correction.
        corrected_text: The claim's replacement text, if the action
            rewrote or replaced it. ``None`` if the action was
            ``REMOVED``, ``FLAGGED``, or ``NONE``.
        action: Which kind of correction (if any) was applied.
        reason: Human-readable explanation of why this action was chosen.
    """

    claim_id: int
    original_text: str
    action: CorrectionAction
    corrected_text: str | None = None
    reason: str = ""

    def __post_init__(self) -> None:
        if self.claim_id <= 0:
            raise WorkflowModelValidationError(
                f"ClaimCorrection.claim_id must be positive, got {self.claim_id}."
            )
        _validate_non_empty(self.original_text, "original_text", "ClaimCorrection")

        requires_text = self.action in (CorrectionAction.REWORDED, CorrectionAction.REPLACED)
        has_text = bool(self.corrected_text and self.corrected_text.strip())
        if requires_text and not has_text:
            raise WorkflowModelValidationError(
                f"ClaimCorrection.action={self.action.value!r} requires a non-empty corrected_text."
            )
        if not requires_text and self.corrected_text is not None:
            raise WorkflowModelValidationError(
                f"ClaimCorrection.action={self.action.value!r} must not set corrected_text."
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize this correction to a plain, JSON-compatible dictionary."""
        return {
            "claim_id": self.claim_id,
            "original_text": self.original_text,
            "corrected_text": self.corrected_text,
            "action": self.action.value,
            "reason": self.reason,
        }


@dataclass(slots=True)
class CorrectionResult:
    """
    Output of the self-correction agent: whether and how the response
    text was rewritten to remove, replace, or flag unsupported or
    contradicted claims.

    Attributes:
        original_response: The full, unmodified LLM response text that
            was analyzed.
        corrected_response: The response text after correction. Equal to
            ``original_response`` when ``was_corrected`` is ``False``.
        was_corrected: Whether any claim-level correction was applied.
        correction_confidence: Confidence (0.0-1.0) in the overall
            correction pass -- how much the agent trusts that
            ``corrected_response`` is now more reliable than the original.
        corrections: Per-claim correction actions taken, in claim order.
        explanation: Human-readable summary of the correction pass.
        corrected_at: UTC timestamp of when correction was performed.
    """

    original_response: str
    corrected_response: str
    was_corrected: bool
    correction_confidence: float
    corrections: tuple[ClaimCorrection, ...] = field(default_factory=tuple)
    explanation: str = ""
    corrected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        _validate_non_empty(self.original_response, "original_response", "CorrectionResult")
        _validate_non_empty(self.corrected_response, "corrected_response", "CorrectionResult")
        _validate_unit_interval(self.correction_confidence, "correction_confidence", "CorrectionResult")

        applied_actions = [c for c in self.corrections if c.action is not CorrectionAction.NONE]
        if applied_actions and not self.was_corrected:
            raise WorkflowModelValidationError(
                "CorrectionResult.was_corrected=False but one or more corrections "
                "has a non-NONE action."
            )

        logger.debug(
            "CorrectionResult constructed: was_corrected=%s, corrections=%d, confidence=%.3f",
            self.was_corrected,
            len(self.corrections),
            self.correction_confidence,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize this result to a plain, JSON-compatible dictionary."""
        return {
            "original_response": self.original_response,
            "corrected_response": self.corrected_response,
            "was_corrected": self.was_corrected,
            "correction_confidence": self.correction_confidence,
            "corrections": [c.to_dict() for c in self.corrections],
            "explanation": self.explanation,
            "corrected_at": self.corrected_at.isoformat(),
        }


# --------------------------------------------------------------------------- #
# Knowledge graph validation
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class EntityValidation:
    """
    The result of checking a single entity mention against the knowledge graph.

    Attributes:
        entity_text: The entity's display text as it appeared in the claim.
        exists_in_graph: Whether a matching node was found in the graph.
        label: The entity's semantic label, if known (e.g. from spaCy NER).
    """

    entity_text: str
    exists_in_graph: bool
    label: str | None = None

    def __post_init__(self) -> None:
        _validate_non_empty(self.entity_text, "entity_text", "EntityValidation")

    def to_dict(self) -> dict[str, Any]:
        """Serialize this validation to a plain, JSON-compatible dictionary."""
        return {
            "entity_text": self.entity_text,
            "exists_in_graph": self.exists_in_graph,
            "label": self.label,
        }


@dataclass(frozen=True, slots=True)
class RelationshipValidation:
    """
    The result of checking a single claimed relationship between two
    entities against the knowledge graph.

    Attributes:
        source: Display text of the relationship's source entity.
        target: Display text of the relationship's target entity.
        relation: The specific relationship type that was checked for,
            if any (``None`` means "any relationship at all" was checked).
        exists_in_graph: Whether a matching relationship was found.
        found_relationships: The actual relationship type(s) found
            between ``source`` and ``target`` in the graph, regardless of
            whether they match ``relation`` (useful for surfacing a
            contradiction, e.g. the graph says ``CAPITAL_OF`` but the
            claim asserted ``LOCATED_IN`` and no such edge exists).
    """

    source: str
    target: str
    exists_in_graph: bool
    relation: str | None = None
    found_relationships: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        _validate_non_empty(self.source, "source", "RelationshipValidation")
        _validate_non_empty(self.target, "target", "RelationshipValidation")

    def to_dict(self) -> dict[str, Any]:
        """Serialize this validation to a plain, JSON-compatible dictionary."""
        return {
            "source": self.source,
            "target": self.target,
            "relation": self.relation,
            "exists_in_graph": self.exists_in_graph,
            "found_relationships": list(self.found_relationships),
        }


@dataclass(slots=True)
class KGValidationResult:
    """
    Output of the knowledge-graph-validation agent for a single analyzed response.

    Attributes:
        backend_used: Which knowledge graph backend produced this result
            -- ``"neo4j"`` or ``"in_memory_fallback"`` (see
            ``KnowledgeGraphService.backend_name``).
        is_consistent: Overall consistency verdict: ``True`` iff no
            contradicted relationships were found (unvalidated/unknown
            entities alone do not make the response inconsistent -- they
            simply couldn't be confirmed).
        consistency_score: Fraction (0.0-1.0) of checked entities and
            relationships that were successfully validated against the graph.
        entities_checked: Per-entity validation results.
        relationships_checked: Per-relationship validation results.
        unvalidated_entities: Display text of entities that could not be
            found in the graph at all.
        contradicted_relationships: Human-readable descriptions of
            relationships the response asserted that directly conflict
            with what the graph contains.
        explanation: Human-readable summary of the validation pass.
        validated_at: UTC timestamp of when validation was performed.
    """

    backend_used: str
    is_consistent: bool
    consistency_score: float
    entities_checked: tuple[EntityValidation, ...] = field(default_factory=tuple)
    relationships_checked: tuple[RelationshipValidation, ...] = field(default_factory=tuple)
    unvalidated_entities: tuple[str, ...] = field(default_factory=tuple)
    contradicted_relationships: tuple[str, ...] = field(default_factory=tuple)
    explanation: str = ""
    validated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        _validate_non_empty(self.backend_used, "backend_used", "KGValidationResult")
        _validate_unit_interval(self.consistency_score, "consistency_score", "KGValidationResult")

        if self.contradicted_relationships and self.is_consistent:
            raise WorkflowModelValidationError(
                "KGValidationResult.is_consistent=True but contradicted_relationships "
                "is non-empty."
            )

        logger.debug(
            "KGValidationResult constructed: backend=%s, is_consistent=%s, score=%.3f",
            self.backend_used,
            self.is_consistent,
            self.consistency_score,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize this result to a plain, JSON-compatible dictionary."""
        return {
            "backend_used": self.backend_used,
            "is_consistent": self.is_consistent,
            "consistency_score": self.consistency_score,
            "entities_checked": [e.to_dict() for e in self.entities_checked],
            "relationships_checked": [r.to_dict() for r in self.relationships_checked],
            "unvalidated_entities": list(self.unvalidated_entities),
            "contradicted_relationships": list(self.contradicted_relationships),
            "explanation": self.explanation,
            "validated_at": self.validated_at.isoformat(),
        }


# --------------------------------------------------------------------------- #
# Knowledge drift detection
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class DriftedClaim:
    """
    A single claim identified as having drifted from a prior/baseline
    knowledge state.

    Attributes:
        claim_id: Identifier of the drifted claim (matches Person 1's
            ``ClaimModel.id``).
        claim_text: The claim's text.
        drift_score: Degree of drift detected for this claim, in [0, 1]
            (higher means more divergence from the baseline).
        reason: Human-readable explanation of what changed.
    """

    claim_id: int
    claim_text: str
    drift_score: float
    reason: str = ""

    def __post_init__(self) -> None:
        if self.claim_id <= 0:
            raise WorkflowModelValidationError(
                f"DriftedClaim.claim_id must be positive, got {self.claim_id}."
            )
        _validate_non_empty(self.claim_text, "claim_text", "DriftedClaim")
        _validate_unit_interval(self.drift_score, "drift_score", "DriftedClaim")

    def to_dict(self) -> dict[str, Any]:
        """Serialize this drifted claim to a plain, JSON-compatible dictionary."""
        return {
            "claim_id": self.claim_id,
            "claim_text": self.claim_text,
            "drift_score": self.drift_score,
            "reason": self.reason,
        }


@dataclass(slots=True)
class DriftReport:
    """
    Output of the knowledge drift detection agent: whether the claims in
    a response have diverged from a previously established baseline
    (e.g. an earlier analysis of the same topic, or a reference snapshot
    of the evidence corpus), which can indicate either stale source
    material or an LLM response drifting away from grounded facts over
    a conversation.

    Attributes:
        has_drift: Whether any meaningful drift was detected.
        drift_severity: Coarse severity bucket for the overall drift.
        overall_drift_score: Aggregate drift score across all claims, in [0, 1].
        drifted_claims: The specific claims found to have drifted.
        baseline_source: Human-readable identifier of what the response
            was compared against (e.g. a document id, a prior analysis
            timestamp, or ``None`` if no baseline was available).
        explanation: Human-readable summary of the drift analysis.
        detected_at: UTC timestamp of when drift detection was performed.
    """

    has_drift: bool
    drift_severity: SeverityLevel
    overall_drift_score: float
    drifted_claims: tuple[DriftedClaim, ...] = field(default_factory=tuple)
    baseline_source: str | None = None
    explanation: str = ""
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        _validate_unit_interval(self.overall_drift_score, "overall_drift_score", "DriftReport")

        if self.drifted_claims and not self.has_drift:
            raise WorkflowModelValidationError(
                "DriftReport.drifted_claims is non-empty but has_drift=False."
            )
        if self.has_drift and self.drift_severity is SeverityLevel.NONE:
            raise WorkflowModelValidationError(
                "DriftReport.has_drift=True but drift_severity is SeverityLevel.NONE."
            )
        if not self.has_drift and self.drift_severity is not SeverityLevel.NONE:
            raise WorkflowModelValidationError(
                "DriftReport.has_drift=False but drift_severity is not SeverityLevel.NONE."
            )

        logger.debug(
            "DriftReport constructed: has_drift=%s, severity=%s, score=%.3f, drifted_claims=%d",
            self.has_drift,
            self.drift_severity.value,
            self.overall_drift_score,
            len(self.drifted_claims),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize this report to a plain, JSON-compatible dictionary."""
        return {
            "has_drift": self.has_drift,
            "drift_severity": self.drift_severity.value,
            "overall_drift_score": self.overall_drift_score,
            "drifted_claims": [c.to_dict() for c in self.drifted_claims],
            "baseline_source": self.baseline_source,
            "explanation": self.explanation,
            "detected_at": self.detected_at.isoformat(),
        }


# --------------------------------------------------------------------------- #
# Risk analysis
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class RiskFactor:
    """
    A single contributing factor to an overall risk assessment.

    Attributes:
        name: Short identifier for the factor (e.g.
            ``"hallucination_probability"``, ``"knowledge_graph_inconsistency"``,
            ``"knowledge_drift"``).
        weight: This factor's contribution to the overall risk score, in [0, 1].
        description: Human-readable explanation of this factor's contribution.
    """

    name: str
    weight: float
    description: str = ""

    def __post_init__(self) -> None:
        _validate_non_empty(self.name, "name", "RiskFactor")
        _validate_unit_interval(self.weight, "weight", "RiskFactor")

    def to_dict(self) -> dict[str, Any]:
        """Serialize this risk factor to a plain, JSON-compatible dictionary."""
        return {"name": self.name, "weight": self.weight, "description": self.description}


@dataclass(slots=True)
class RiskAssessment:
    """
    Output of the risk analysis module: an aggregated risk assessment
    combining hallucination-detection, knowledge-graph-consistency, and
    knowledge-drift signals into a single actionable verdict.

    Attributes:
        risk_level: Coarse overall risk categorization.
        risk_score: Aggregate numeric risk score, in [0, 1] (higher is riskier).
        recommendation: A short, human-readable recommended action
            (e.g. "Safe to use as-is.", "Review before use.", "Do not
            use without human verification.").
        requires_human_review: Whether this response should be routed to
            human review before being used, per the configured risk policy.
        risk_factors: The individual signals that were combined to
            produce ``risk_score``, with their relative weights.
        explanation: Human-readable summary of the overall risk assessment.
        assessed_at: UTC timestamp of when the assessment was performed.
    """

    risk_level: SeverityLevel
    risk_score: float
    recommendation: str
    requires_human_review: bool
    risk_factors: tuple[RiskFactor, ...] = field(default_factory=tuple)
    explanation: str = ""
    assessed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        _validate_unit_interval(self.risk_score, "risk_score", "RiskAssessment")
        _validate_non_empty(self.recommendation, "recommendation", "RiskAssessment")

        if self.risk_level is SeverityLevel.CRITICAL and not self.requires_human_review:
            raise WorkflowModelValidationError(
                "RiskAssessment.risk_level=CRITICAL must have requires_human_review=True."
            )

        logger.debug(
            "RiskAssessment constructed: level=%s, score=%.3f, requires_review=%s",
            self.risk_level.value,
            self.risk_score,
            self.requires_human_review,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize this assessment to a plain, JSON-compatible dictionary."""
        return {
            "risk_level": self.risk_level.value,
            "risk_score": self.risk_score,
            "recommendation": self.recommendation,
            "requires_human_review": self.requires_human_review,
            "risk_factors": [f.to_dict() for f in self.risk_factors],
            "explanation": self.explanation,
            "assessed_at": self.assessed_at.isoformat(),
        }


# --------------------------------------------------------------------------- #
# Aggregated workflow output
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class WorkflowResult:
    """
    Final, structured output of running the full Person 3 LangGraph
    multi-agent workflow on a single LLM response.

    This is the object the ``/workflow`` API route (added in a later
    batch) serializes and returns to the caller -- the Person 3
    equivalent of Person 1's ``ResponseAnalysis``.

    Attributes:
        response_text: The original LLM response text that was analyzed.
        final_verdict: The workflow's overall verdict string (e.g. one
            of Person 1's ``Verdict`` values, or a Person-3-specific
            refinement of it once risk analysis has run).
        final_verdict_reason: Human-readable explanation of the final verdict.
        total_claims: Total number of claims extracted from the response.
        correction: The self-correction agent's output, if that stage ran.
        kg_validation: The knowledge-graph-validation agent's output, if that stage ran.
        drift_report: The drift detection agent's output, if that stage ran.
        risk_assessment: The risk analysis module's output, if that stage ran.
        completed_steps: Names of every workflow step that ran successfully,
            in execution order (matches ``app.agents.state.WorkflowStep`` values).
        errors: Human-readable descriptions of any node-level failures
            that occurred during the run but did not abort the whole
            workflow (partial results are still returned when possible).
        processed_at: UTC timestamp of when the workflow run completed.
    """

    response_text: str
    final_verdict: str
    final_verdict_reason: str = ""
    total_claims: int = 0
    correction: CorrectionResult | None = None
    kg_validation: KGValidationResult | None = None
    drift_report: DriftReport | None = None
    risk_assessment: RiskAssessment | None = None
    completed_steps: tuple[str, ...] = field(default_factory=tuple)
    errors: tuple[str, ...] = field(default_factory=tuple)
    processed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        _validate_non_empty(self.response_text, "response_text", "WorkflowResult")
        _validate_non_empty(self.final_verdict, "final_verdict", "WorkflowResult")
        if self.total_claims < 0:
            raise WorkflowModelValidationError(
                f"WorkflowResult.total_claims must be >= 0, got {self.total_claims}."
            )

        logger.debug(
            "WorkflowResult constructed: verdict=%s, total_claims=%d, steps=%d, errors=%d",
            self.final_verdict,
            self.total_claims,
            len(self.completed_steps),
            len(self.errors),
        )

    @property
    def has_errors(self) -> bool:
        """True if one or more workflow steps recorded an error during this run."""
        return len(self.errors) > 0

    @property
    def requires_human_review(self) -> bool:
        """
        True if the risk assessment (when present) flagged this response
        as needing human review. ``False`` (rather than raising) when no
        risk assessment was produced, since callers typically want a
        simple boolean gate without first checking for ``None``.
        """
        return bool(self.risk_assessment and self.risk_assessment.requires_human_review)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the full workflow result to a plain, JSON-compatible dictionary."""
        return {
            "response_text": self.response_text,
            "final_verdict": self.final_verdict,
            "final_verdict_reason": self.final_verdict_reason,
            "total_claims": self.total_claims,
            "correction": self.correction.to_dict() if self.correction is not None else None,
            "kg_validation": self.kg_validation.to_dict() if self.kg_validation is not None else None,
            "drift_report": self.drift_report.to_dict() if self.drift_report is not None else None,
            "risk_assessment": (
                self.risk_assessment.to_dict() if self.risk_assessment is not None else None
            ),
            "completed_steps": list(self.completed_steps),
            "errors": list(self.errors),
            "requires_human_review": self.requires_human_review,
            "processed_at": self.processed_at.isoformat(),
        }
