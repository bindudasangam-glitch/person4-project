"""
Workflow API Schemas
========================

Defines the request and response models for the ``/workflow`` router
(``app/api/routes/workflow.py``). Kept separate from the internal
dataclass domain models in ``app.models.workflow_models`` -- exactly
the same separation Person 2 already established between
``app.models.verification.VerificationResult`` and
``app.schemas.verification_schema.ClaimVerificationResponse`` -- so the
API's public JSON shape can evolve independently of the internal
``WorkflowResult``/``CorrectionResult``/``KGValidationResult``/
``DriftReport``/``RiskAssessment`` dataclasses those agents actually operate on.

Every response schema exposes a ``from_result``/``from_domain``
classmethod that builds it from the corresponding internal dataclass's
own ``to_dict()`` output, so this module never re-implements or drifts
from the serialization logic already defined on those models.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

from app.agents.workflow import get_final_response
from app.models.workflow_models import (
    CorrectionResult,
    DriftReport,
    KGValidationResult,
    RiskAssessment,
    WorkflowResult,
)

__all__ = [
    "WorkflowAnalyzeRequest",
    "ClaimCorrectionSummary",
    "CorrectionResultSummary",
    "EntityValidationSummary",
    "RelationshipValidationSummary",
    "KGValidationResultSummary",
    "DriftedClaimSummary",
    "DriftReportSummary",
    "RiskFactorSummary",
    "RiskAssessmentSummary",
    "WorkflowAnalyzeResponse",
]


class WorkflowAnalyzeRequest(BaseModel):
    """
    Request payload for running the full Person 3 multi-agent workflow
    on a single LLM-generated response.

    Attributes:
        response_text: The raw LLM-generated response text to analyze.
        document_id: If provided, restricts this workflow run's
            supplementary Person 2 evidence lookups (used during
            self-correction) to this specific document only.
        top_k: Maximum number of supplementary evidence passages to
            retrieve per claim needing correction. If omitted, the
            workflow's configured default is used.
    """

    response_text: str = Field(
        ...,
        min_length=1,
        max_length=20_000,
        description="The raw LLM-generated response text to analyze for hallucinations.",
    )
    document_id: Optional[str] = Field(
        default=None,
        description="Optional document filter for supplementary evidence retrieval during self-correction.",
    )
    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=100,
        description="Maximum supplementary evidence passages to retrieve per claim needing correction.",
    )


class ClaimCorrectionSummary(BaseModel):
    """A single per-claim correction action taken by the self-correction agent."""

    claim_id: int
    original_text: str
    corrected_text: Optional[str] = None
    action: str
    reason: str


class CorrectionResultSummary(BaseModel):
    """
    Output of the self-correction agent: whether and how the response
    was rewritten.

    Attributes:
        original_response: The full, unmodified response text that was analyzed.
        corrected_response: The response text after correction (the
            final corrected answer -- see ``WorkflowAnalyzeResponse.corrected_answer``).
        was_corrected: Whether any claim-level correction was applied.
        correction_confidence: Confidence in the overall correction pass, in ``[0, 1]``.
        corrections: Per-claim correction actions taken, in claim order.
        explanation: Human-readable summary of the correction pass.
        corrected_at: ISO-8601 UTC timestamp of when correction was performed.
    """

    original_response: str
    corrected_response: str
    was_corrected: bool
    correction_confidence: float = Field(..., ge=0.0, le=1.0)
    corrections: list[ClaimCorrectionSummary] = Field(default_factory=list)
    explanation: str
    corrected_at: str

    @classmethod
    def from_domain(cls, correction: CorrectionResult) -> "CorrectionResultSummary":
        """Build a response schema instance from an internal :class:`CorrectionResult`."""
        return cls(**correction.to_dict())


class EntityValidationSummary(BaseModel):
    """The result of checking a single entity mention against the knowledge graph."""

    entity_text: str
    exists_in_graph: bool
    label: Optional[str] = None


class RelationshipValidationSummary(BaseModel):
    """The result of checking a single claimed relationship between two entities."""

    source: str
    target: str
    relation: Optional[str] = None
    exists_in_graph: bool
    found_relationships: list[str] = Field(default_factory=list)


class KGValidationResultSummary(BaseModel):
    """
    Output of the knowledge-graph-validation agent.

    Attributes:
        backend_used: Which knowledge graph backend produced this result
            -- ``"neo4j"`` or ``"in_memory_fallback"``.
        is_consistent: Overall consistency verdict.
        consistency_score: Fraction of checked entities/relationships
            successfully validated against the graph, in ``[0, 1]``.
        entities_checked: Per-entity validation results.
        relationships_checked: Per-relationship validation results.
        unvalidated_entities: Entities that could not be found in the graph at all.
        contradicted_relationships: Relationships the response asserted
            that directly conflict with what the graph contains.
        explanation: Human-readable summary of the validation pass.
        validated_at: ISO-8601 UTC timestamp of when validation was performed.
    """

    backend_used: str
    is_consistent: bool
    consistency_score: float = Field(..., ge=0.0, le=1.0)
    entities_checked: list[EntityValidationSummary] = Field(default_factory=list)
    relationships_checked: list[RelationshipValidationSummary] = Field(default_factory=list)
    unvalidated_entities: list[str] = Field(default_factory=list)
    contradicted_relationships: list[str] = Field(default_factory=list)
    explanation: str
    validated_at: str

    @classmethod
    def from_domain(cls, kg_validation: KGValidationResult) -> "KGValidationResultSummary":
        """Build a response schema instance from an internal :class:`KGValidationResult`."""
        return cls(**kg_validation.to_dict())


class DriftedClaimSummary(BaseModel):
    """A single claim identified as having drifted from a prior/baseline knowledge state."""

    claim_id: int
    claim_text: str
    drift_score: float = Field(..., ge=0.0, le=1.0)
    reason: str


class DriftReportSummary(BaseModel):
    """
    Output of the knowledge drift detection agent.

    Attributes:
        has_drift: Whether any meaningful drift was detected.
        drift_severity: Coarse severity bucket for the overall drift
            (``none`` / ``low`` / ``medium`` / ``high`` / ``critical``).
        overall_drift_score: Aggregate drift score across all compared claims, in ``[0, 1]``.
        drifted_claims: The specific claims found to have drifted.
        baseline_source: What the response was compared against, if anything.
        explanation: Human-readable summary of the drift analysis.
        detected_at: ISO-8601 UTC timestamp of when drift detection was performed.
    """

    has_drift: bool
    drift_severity: str
    overall_drift_score: float = Field(..., ge=0.0, le=1.0)
    drifted_claims: list[DriftedClaimSummary] = Field(default_factory=list)
    baseline_source: Optional[str] = None
    explanation: str
    detected_at: str

    @classmethod
    def from_domain(cls, drift_report: DriftReport) -> "DriftReportSummary":
        """Build a response schema instance from an internal :class:`DriftReport`."""
        return cls(**drift_report.to_dict())


class RiskFactorSummary(BaseModel):
    """A single contributing factor to an overall risk assessment."""

    name: str
    weight: float = Field(..., ge=0.0, le=1.0)
    description: str


class RiskAssessmentSummary(BaseModel):
    """
    Output of the risk analysis module.

    Attributes:
        risk_level: Coarse overall risk categorization
            (``low`` / ``medium`` / ``high`` / ``critical``).
        risk_score: Aggregate numeric risk score, in ``[0, 1]``.
        recommendation: A short, human-readable recommended action.
        requires_human_review: Whether this response should be routed
            to human review before use.
        risk_factors: The individual signals combined to produce ``risk_score``.
        explanation: Human-readable summary of the overall risk assessment.
        assessed_at: ISO-8601 UTC timestamp of when the assessment was performed.
    """

    risk_level: str
    risk_score: float = Field(..., ge=0.0, le=1.0)
    recommendation: str
    requires_human_review: bool
    risk_factors: list[RiskFactorSummary] = Field(default_factory=list)
    explanation: str
    assessed_at: str

    @classmethod
    def from_domain(cls, risk_assessment: RiskAssessment) -> "RiskAssessmentSummary":
        """Build a response schema instance from an internal :class:`RiskAssessment`."""
        return cls(**risk_assessment.to_dict())


class WorkflowAnalyzeResponse(BaseModel):
    """
    Final structured response returned by the ``POST /workflow/analyze`` endpoint.

    Attributes:
        original_response: The original, unmodified LLM response text that was analyzed.
        corrected_answer: The final corrected answer -- the single text a
            caller should actually show or act on. Equal to
            ``original_response`` when self-correction found nothing to
            change (or did not run, e.g. no verifiable claims were found).
        final_verdict: The workflow's overall verdict.
        final_verdict_reason: Human-readable explanation of the final verdict.
        hallucination_score: Person 1's aggregate hallucination
            probability for the response, in ``[0, 1]``, if computed.
        confidence_score: Person 1's aggregate confidence/trust score
            for the response, in ``[0, 1]``, if computed.
        total_claims: Total number of claims extracted from the response.
        correction: The self-correction agent's full output, if that stage ran.
        knowledge_graph_validation: The knowledge-graph-validation
            agent's full output, if that stage ran.
        drift_report: The drift detection agent's full output, if that stage ran.
        risk_assessment: The risk analysis module's full output, if that stage ran.
        completed_steps: Names of every workflow step that ran successfully, in execution order.
        errors: Human-readable descriptions of any node-level failures
            that occurred during the run but did not abort the workflow.
        execution_metadata: Free-form bookkeeping accumulated across
            every workflow node (per-stage backends used, timings-relevant
            markers, and other diagnostic detail beyond the structured fields above).
        requires_human_review: Whether the risk assessment (when
            present) flagged this response as needing human review.
        processed_at: ISO-8601 UTC timestamp of when the workflow run completed.
    """

    original_response: str
    corrected_answer: str
    final_verdict: str
    final_verdict_reason: str
    hallucination_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    confidence_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    total_claims: int = Field(..., ge=0)
    correction: Optional[CorrectionResultSummary] = None
    knowledge_graph_validation: Optional[KGValidationResultSummary] = None
    drift_report: Optional[DriftReportSummary] = None
    risk_assessment: Optional[RiskAssessmentSummary] = None
    completed_steps: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    execution_metadata: dict[str, Any] = Field(default_factory=dict)
    requires_human_review: bool
    processed_at: str

    @classmethod
    def from_workflow_result(
        cls,
        result: WorkflowResult,
        hallucination_score: Optional[float] = None,
        confidence_score: Optional[float] = None,
        execution_metadata: Optional[dict[str, Any]] = None,
    ) -> "WorkflowAnalyzeResponse":
        """
        Build the API response from a completed
        :class:`~app.models.workflow_models.WorkflowResult`.

        Args:
            result: The completed workflow result.
            hallucination_score: Person 1's aggregate hallucination
                probability for the response, threaded through
                separately since it is not itself a field on
                ``WorkflowResult`` (it lives upstream, in
                ``WorkflowState``, alongside ``confidence_score``).
            confidence_score: Person 1's aggregate confidence/trust
                score for the response, threaded through the same way.
            execution_metadata: The workflow run's free-form execution
                metadata, threaded through the same way.

        Returns:
            A fully populated :class:`WorkflowAnalyzeResponse`.
        """
        return cls(
            original_response=result.response_text,
            corrected_answer=get_final_response(result),
            final_verdict=result.final_verdict,
            final_verdict_reason=result.final_verdict_reason,
            hallucination_score=hallucination_score,
            confidence_score=confidence_score,
            total_claims=result.total_claims,
            correction=(
                CorrectionResultSummary.from_domain(result.correction)
                if result.correction is not None
                else None
            ),
            knowledge_graph_validation=(
                KGValidationResultSummary.from_domain(result.kg_validation)
                if result.kg_validation is not None
                else None
            ),
            drift_report=(
                DriftReportSummary.from_domain(result.drift_report)
                if result.drift_report is not None
                else None
            ),
            risk_assessment=(
                RiskAssessmentSummary.from_domain(result.risk_assessment)
                if result.risk_assessment is not None
                else None
            ),
            completed_steps=list(result.completed_steps),
            errors=list(result.errors),
            execution_metadata=dict(execution_metadata or {}),
            requires_human_review=result.requires_human_review,
            processed_at=result.processed_at.isoformat(),
        )
