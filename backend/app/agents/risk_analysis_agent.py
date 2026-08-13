"""
Risk Analysis Module
========================

Combines every upstream Person 3 (and reused Person 1) signal -- the
hallucination probability and confidence score from Person 1's
``ConfidenceScorer``, the knowledge-graph-validation result, and the
knowledge drift report -- into a single, actionable
:class:`~app.models.workflow_models.RiskAssessment`.

Scoring model
--------------
Four weighted components are combined into an overall ``risk_score`` in
``[0, 1]`` (higher is riskier):

* ``hallucination_probability``    -- taken directly from Person 1.
* ``confidence_deficit``           -- ``1 - confidence_score`` from Person 1.
* ``knowledge_graph_inconsistency``-- large and floor-clamped when the
  knowledge-graph-validation agent found a corroborated contradiction;
  otherwise a small signal from the fraction of unconfirmed entities alone.
* ``knowledge_drift``              -- the drift report's overall drift
  score when drift was actually detected, else zero.

The weighted score is then bucketed into ``LOW`` / ``MEDIUM`` / ``HIGH``
/ ``CRITICAL`` via configurable thresholds, followed by two narrow,
explicit escalation rules (an extreme hallucination probability, or any
knowledge graph contradiction, each guarantee a floor on the resulting
level) so a single very strong danger signal can never be diluted away
by an otherwise-reassuring weighted average -- the same escalation
philosophy Person 2's ``ResponseAnalyzer`` already applies when
refining its verdict in the presence of contradicted claims.
"""

from __future__ import annotations

import math

from app.agents.exceptions import RiskAnalysisError
from app.core.logging import logger
from app.models.workflow_models import (
    DriftReport,
    KGValidationResult,
    RiskAssessment,
    RiskFactor,
    SeverityLevel,
    WorkflowModelValidationError,
)

__all__ = ["RiskAnalysisAgent"]

#: The four named risk components combined into an overall risk score,
#: in the fixed order they are reported as ``RiskAssessment.risk_factors``.
_RISK_FACTOR_NAMES: tuple[str, ...] = (
    "hallucination_probability",
    "confidence_deficit",
    "knowledge_graph_inconsistency",
    "knowledge_drift",
)


class RiskAnalysisAgent:
    """
    Produces a single, actionable :class:`RiskAssessment` from the
    outputs of Person 1's confidence scoring and Person 3's
    knowledge-graph-validation and drift-detection agents.

    Args:
        factor_weights: Per-component weight, in ``[0, 1]``, used to
            combine the four risk components into an overall
            ``risk_score``. Keys must exactly match
            :data:`_RISK_FACTOR_NAMES` and sum to ``1.0`` (within a small
            floating-point tolerance). Defaults to
            :attr:`DEFAULT_FACTOR_WEIGHTS`. Injected so callers/tests can
            re-tune the scoring model without subclassing.
        severity_thresholds: ``(low_upper, medium_upper, high_upper)``
            cutoffs on the combined ``risk_score`` used to classify
            ``risk_level``. Must be strictly increasing and within
            ``(0, 1]``. Defaults to :attr:`DEFAULT_SEVERITY_THRESHOLDS`.
        recommendations: Per-``SeverityLevel`` human-readable
            recommendation string (must cover ``LOW``/``MEDIUM``/``HIGH``/
            ``CRITICAL``). Defaults to :attr:`DEFAULT_RECOMMENDATIONS`.
        extreme_hallucination_threshold: A ``hallucination_probability``
            at or above this value forces ``risk_level`` to at least
            ``HIGH``, regardless of the weighted score. Defaults to ``0.9``.

    Raises:
        RiskAnalysisError: If any constructor argument is invalid.
    """

    #: Default per-component weight. Hallucination probability is
    #: weighted most heavily since it is Person 1's own direct verdict on
    #: the response; the other three are corroborating/contextual signals.
    DEFAULT_FACTOR_WEIGHTS: dict[str, float] = {
        "hallucination_probability": 0.40,
        "confidence_deficit": 0.25,
        "knowledge_graph_inconsistency": 0.20,
        "knowledge_drift": 0.15,
    }

    #: Default ``(low_upper, medium_upper, high_upper)`` cutoffs on the
    #: combined risk score.
    DEFAULT_SEVERITY_THRESHOLDS: tuple[float, float, float] = (0.25, 0.5, 0.75)

    #: Default human-readable recommendation per severity level.
    DEFAULT_RECOMMENDATIONS: dict[SeverityLevel, str] = {
        SeverityLevel.LOW: "Safe to use as-is.",
        SeverityLevel.MEDIUM: "Review recommended before use in high-stakes contexts.",
        SeverityLevel.HIGH: "Review required before use; the response has significant reliability concerns.",
        SeverityLevel.CRITICAL: (
            "Do not use without human verification; the response has serious, corroborated "
            "reliability issues."
        ),
    }

    def __init__(
        self,
        factor_weights: dict[str, float] | None = None,
        severity_thresholds: tuple[float, float, float] | None = None,
        recommendations: dict[SeverityLevel, str] | None = None,
        extreme_hallucination_threshold: float = 0.9,
    ) -> None:
        weights = factor_weights or dict(self.DEFAULT_FACTOR_WEIGHTS)
        missing_weights = set(_RISK_FACTOR_NAMES) - set(weights)
        if missing_weights:
            raise RiskAnalysisError(f"factor_weights is missing entries for: {sorted(missing_weights)}.")
        extra_weights = set(weights) - set(_RISK_FACTOR_NAMES)
        if extra_weights:
            raise RiskAnalysisError(f"factor_weights has unexpected entries: {sorted(extra_weights)}.")
        for name, weight in weights.items():
            if not 0.0 <= weight <= 1.0:
                raise RiskAnalysisError(f"factor_weights[{name}] must be within [0.0, 1.0], got {weight}.")
        if not math.isclose(sum(weights.values()), 1.0, abs_tol=1e-6):
            raise RiskAnalysisError(f"factor_weights must sum to 1.0, got {sum(weights.values())!r}.")

        thresholds = severity_thresholds or self.DEFAULT_SEVERITY_THRESHOLDS
        low, medium, high = thresholds
        if not (0.0 < low < medium < high <= 1.0):
            raise RiskAnalysisError(
                f"severity_thresholds must be strictly increasing within (0, 1], got {thresholds}."
            )

        recs = recommendations or dict(self.DEFAULT_RECOMMENDATIONS)
        required_levels = {SeverityLevel.LOW, SeverityLevel.MEDIUM, SeverityLevel.HIGH, SeverityLevel.CRITICAL}
        missing_recs = required_levels - set(recs)
        if missing_recs:
            raise RiskAnalysisError(
                f"recommendations is missing entries for: {sorted(level.value for level in missing_recs)}."
            )
        for level, text in recs.items():
            if not text or not text.strip():
                raise RiskAnalysisError(f"recommendations[{getattr(level, 'value', level)}] must not be empty.")

        if not 0.0 <= extreme_hallucination_threshold <= 1.0:
            raise RiskAnalysisError(
                f"extreme_hallucination_threshold must be within [0.0, 1.0], "
                f"got {extreme_hallucination_threshold}."
            )

        self._factor_weights = weights
        self._severity_thresholds = thresholds
        self._recommendations = recs
        self._extreme_hallucination_threshold = extreme_hallucination_threshold

    def analyze(
        self,
        hallucination_score: float,
        confidence_score: float,
        kg_validation: KGValidationResult | None = None,
        drift_report: DriftReport | None = None,
    ) -> RiskAssessment:
        """
        Compute an overall risk assessment for a single analyzed response.

        Args:
            hallucination_score: Person 1's aggregate hallucination
                probability for the response, in ``[0, 1]`` (e.g.
                ``ConfidenceScoreResult.hallucination_probability``).
            confidence_score: Person 1's aggregate confidence/trust score
                for the response, in ``[0, 1]`` (e.g.
                ``ConfidenceScoreResult.confidence_score``).
            kg_validation: The knowledge-graph-validation agent's result,
                if that stage ran. ``None`` contributes no signal (rather
                than being treated as "inconsistent") since the absence
                of a check is not evidence of a problem.
            drift_report: The drift detection agent's result, if that
                stage ran. ``None`` (or ``has_drift=False``) contributes
                no signal.

        Returns:
            A :class:`~app.models.workflow_models.RiskAssessment`
            aggregating all available signals.

        Raises:
            RiskAnalysisError: If ``hallucination_score`` or
                ``confidence_score`` is outside ``[0, 1]``, or an invalid
                ``RiskAssessment`` would otherwise be produced.
        """
        if not 0.0 <= hallucination_score <= 1.0:
            raise RiskAnalysisError(
                f"hallucination_score must be within [0.0, 1.0], got {hallucination_score}."
            )
        if not 0.0 <= confidence_score <= 1.0:
            raise RiskAnalysisError(f"confidence_score must be within [0.0, 1.0], got {confidence_score}.")

        try:
            components = self._compute_components(hallucination_score, confidence_score, kg_validation, drift_report)
            risk_score = self._combine_components(components)
            risk_level = self._classify_risk_level(risk_score, hallucination_score, kg_validation)
            risk_factors = self._build_risk_factors(components)
            requires_human_review = self._requires_human_review(risk_level, kg_validation)
            recommendation = self._recommendations[risk_level]
            explanation = self._build_explanation(risk_level, risk_score, components, kg_validation, drift_report)
        except RiskAnalysisError:
            raise
        except Exception as exc:  # noqa: BLE001 - normalize to domain error
            logger.exception("RiskAnalysisAgent failed to compute a risk assessment.")
            raise RiskAnalysisError("Failed to compute a risk assessment.") from exc

        try:
            result = RiskAssessment(
                risk_level=risk_level,
                risk_score=risk_score,
                recommendation=recommendation,
                requires_human_review=requires_human_review,
                risk_factors=tuple(risk_factors),
                explanation=explanation,
            )
        except WorkflowModelValidationError as exc:
            logger.exception("RiskAnalysisAgent produced an invalid RiskAssessment.")
            raise RiskAnalysisError("Failed to construct a valid RiskAssessment.") from exc

        logger.info(
            "RiskAnalysisAgent complete: level=%s, score=%.3f, requires_review=%s.",
            risk_level.value,
            risk_score,
            requires_human_review,
        )
        return result

    # ------------------------------------------------------------------ #
    # Component computation
    # ------------------------------------------------------------------ #
    @staticmethod
    def _compute_components(
        hallucination_score: float,
        confidence_score: float,
        kg_validation: KGValidationResult | None,
        drift_report: DriftReport | None,
    ) -> dict[str, float]:
        """Compute each of the four named risk components, in ``[0, 1]``."""
        confidence_deficit = round(1.0 - confidence_score, 4)

        if kg_validation is None:
            kg_component = 0.0
        elif not kg_validation.is_consistent:
            # A corroborated contradiction is a strong signal on its own,
            # floor-clamped so a high consistency_score elsewhere in the
            # graph can't dilute away an actual contradiction.
            kg_component = max(0.7, round(1.0 - kg_validation.consistency_score, 4))
        else:
            # Consistent, but not every entity could be confirmed: a much
            # weaker signal, since an unconfirmed entity alone (e.g. on a
            # sparse or freshly-seeded graph) is not evidence of a problem.
            kg_component = round((1.0 - kg_validation.consistency_score) * 0.3, 4)

        if drift_report is None or not drift_report.has_drift:
            drift_component = 0.0
        else:
            drift_component = drift_report.overall_drift_score

        return {
            "hallucination_probability": round(hallucination_score, 4),
            "confidence_deficit": confidence_deficit,
            "knowledge_graph_inconsistency": kg_component,
            "knowledge_drift": drift_component,
        }

    def _combine_components(self, components: dict[str, float]) -> float:
        """Combine the weighted components into a single overall risk score, clamped to ``[0, 1]``."""
        total = sum(self._factor_weights[name] * value for name, value in components.items())
        return round(min(1.0, max(0.0, total)), 4)

    # ------------------------------------------------------------------ #
    # Classification
    # ------------------------------------------------------------------ #
    def _classify_risk_level(
        self,
        risk_score: float,
        hallucination_score: float,
        kg_validation: KGValidationResult | None,
    ) -> SeverityLevel:
        """
        Bucket ``risk_score`` into a :class:`SeverityLevel`, then apply
        two narrow escalation rules so a single very strong danger signal
        cannot be diluted away by the weighted average. Never returns
        ``SeverityLevel.NONE`` -- every analyzed response is assessed as
        at least ``LOW`` risk.
        """
        low_cutoff, medium_cutoff, high_cutoff = self._severity_thresholds
        if risk_score < low_cutoff:
            level = SeverityLevel.LOW
        elif risk_score < medium_cutoff:
            level = SeverityLevel.MEDIUM
        elif risk_score < high_cutoff:
            level = SeverityLevel.HIGH
        else:
            level = SeverityLevel.CRITICAL

        if hallucination_score >= self._extreme_hallucination_threshold and level in (
            SeverityLevel.LOW,
            SeverityLevel.MEDIUM,
        ):
            logger.debug(
                "RiskAnalysisAgent escalating risk level to HIGH: extreme hallucination_score=%.3f.",
                hallucination_score,
            )
            level = SeverityLevel.HIGH

        if kg_validation is not None and not kg_validation.is_consistent and level is SeverityLevel.LOW:
            logger.debug("RiskAnalysisAgent escalating risk level to MEDIUM: knowledge graph inconsistency.")
            level = SeverityLevel.MEDIUM

        return level

    @staticmethod
    def _requires_human_review(risk_level: SeverityLevel, kg_validation: KGValidationResult | None) -> bool:
        """A response requires human review at HIGH/CRITICAL risk, or on any knowledge graph contradiction."""
        if risk_level in (SeverityLevel.HIGH, SeverityLevel.CRITICAL):
            return True
        if kg_validation is not None and not kg_validation.is_consistent:
            return True
        return False

    # ------------------------------------------------------------------ #
    # Reporting
    # ------------------------------------------------------------------ #
    def _build_risk_factors(self, components: dict[str, float]) -> list[RiskFactor]:
        """Build the reported :class:`RiskFactor` list from the computed components, in a fixed order."""
        descriptions = {
            "hallucination_probability": (
                f"Hallucination probability of {components['hallucination_probability']:.2f} "
                "from Person 1's confidence scoring."
            ),
            "confidence_deficit": (
                f"Confidence deficit of {components['confidence_deficit']:.2f} "
                "(1 minus Person 1's aggregate confidence score)."
            ),
            "knowledge_graph_inconsistency": (
                f"Knowledge graph inconsistency contribution of "
                f"{components['knowledge_graph_inconsistency']:.2f}."
            ),
            "knowledge_drift": f"Knowledge drift contribution of {components['knowledge_drift']:.2f}.",
        }
        return [
            RiskFactor(name=name, weight=self._factor_weights[name], description=descriptions[name])
            for name in _RISK_FACTOR_NAMES
        ]

    @staticmethod
    def _build_explanation(
        risk_level: SeverityLevel,
        risk_score: float,
        components: dict[str, float],
        kg_validation: KGValidationResult | None,
        drift_report: DriftReport | None,
    ) -> str:
        """Build a short, human-readable summary of the overall risk assessment."""
        parts = [
            f"Overall risk score {risk_score:.2f} classified as '{risk_level.value}' "
            f"(hallucination probability {components['hallucination_probability']:.2f}, "
            f"confidence deficit {components['confidence_deficit']:.2f})"
        ]
        if kg_validation is not None:
            consistency_word = "consistent" if kg_validation.is_consistent else "inconsistent"
            parts.append(
                f"knowledge graph {consistency_word} (consistency score {kg_validation.consistency_score:.2f})"
            )
        if drift_report is not None and drift_report.has_drift:
            parts.append(f"knowledge drift detected (severity '{drift_report.drift_severity.value}')")
        return "; ".join(parts) + "."
