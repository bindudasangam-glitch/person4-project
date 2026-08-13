"""
Confidence Scoring Service
============================

Aggregates the per-claim verdicts produced by
:class:`app.services.hallucination_detector.HallucinationDetector` into the
response-level metrics required by the system:

* **Trust Score** — weighted proportion of claims that are supported by
  evidence.
* **Hallucination Probability** — weighted proportion of claim "risk"
  (contradicted claims count fully, unsupported/insufficient-evidence claims
  count partially).
* **Reliability Score** — how much of the verdict is actually grounded in
  retrieved evidence, as opposed to defaulting to "insufficient evidence"
  because nothing relevant was found.
* **Confidence Score** — meta-confidence in the analysis pipeline itself,
  combining claim-extraction quality with evidence groundedness.

Weighting
---------
Each claim contributes to the aggregate scores proportionally to its
``extraction_confidence`` (produced by ``ClaimExtractor``): a claim the
extractor was unsure about (short sentence, no entities) should not sway the
overall verdict as much as a long, entity-rich, cleanly extracted claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from app.core.logging import logger
from app.models.claim_model import ClaimModel, VerificationStatus

__all__ = [
    "ConfidenceScoringError",
    "RiskLevel",
    "ClaimScore",
    "ConfidenceScoreResult",
    "ConfidenceScorer",
]


class ConfidenceScoringError(Exception):
    """Raised when confidence scoring cannot be computed for the given input."""


class RiskLevel(str, Enum):
    """Coarse, human-readable summary of the hallucination risk of a response."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True, slots=True)
class ClaimScore:
    """Per-claim contribution to the aggregate confidence metrics."""

    claim_id: int
    verification_status: VerificationStatus
    support_value: float
    weight: float


@dataclass(frozen=True, slots=True)
class ConfidenceScoreResult:
    """Aggregate scoring result for an entire analyzed response."""

    trust_score: float
    reliability_score: float
    hallucination_probability: float
    confidence_score: float
    risk_level: RiskLevel
    total_claims: int
    supported_claims: int
    contradicted_claims: int
    insufficient_evidence_claims: int
    per_claim_scores: tuple[ClaimScore, ...]


class ConfidenceScorer:
    """
    Computes Trust Score, Reliability Score, Hallucination Probability and
    an overall Confidence Score from a batch of verified claims.

    Args:
        min_claim_weight: Floor applied to each claim's ``extraction_confidence``
            so that a single low-confidence claim cannot be given effectively
            zero influence (and thus silently ignored) in the aggregate.
        risk_thresholds: ``(low_upper_bound, medium_upper_bound)`` cutoffs on
            ``hallucination_probability`` used to derive :class:`RiskLevel`.
    """

    _SUPPORT_VALUES: dict[VerificationStatus, float] = {
        VerificationStatus.SUPPORTED: 1.0,
        VerificationStatus.INSUFFICIENT_EVIDENCE: 0.5,
        VerificationStatus.UNVERIFIED: 0.5,
        VerificationStatus.CONTRADICTED: 0.0,
    }

    def __init__(
        self,
        min_claim_weight: float = 0.1,
        risk_thresholds: tuple[float, float] = (0.3, 0.6),
    ) -> None:
        if not 0.0 < min_claim_weight <= 1.0:
            raise ConfidenceScoringError(
                f"min_claim_weight must be in (0, 1], got {min_claim_weight}."
            )

        low, medium = risk_thresholds
        if not 0.0 <= low < medium <= 1.0:
            raise ConfidenceScoringError(
                f"risk_thresholds must satisfy 0 <= low < medium <= 1, got {risk_thresholds}."
            )

        self._min_claim_weight = min_claim_weight
        self._risk_thresholds = risk_thresholds

    def score(self, claims: list[ClaimModel]) -> ConfidenceScoreResult:
        """
        Compute aggregate confidence metrics for a batch of verified claims.

        Args:
            claims: Claims that have already been passed through
                ``HallucinationDetector.detect`` (i.e. carry a meaningful
                ``verification_status``).

        Returns:
            A :class:`ConfidenceScoreResult` summarizing the response.

        Raises:
            ConfidenceScoringError: If ``claims`` is empty or the total claim
                weight is degenerate (zero).
        """
        if not claims:
            raise ConfidenceScoringError("Cannot score an empty claim list.")

        claim_scores = tuple(self._score_claim(claim) for claim in claims)
        total_weight = sum(cs.weight for cs in claim_scores)

        if total_weight <= 0.0:
            raise ConfidenceScoringError("Total claim weight is zero; cannot compute scores.")

        trust_score = round(
            sum(cs.support_value * cs.weight for cs in claim_scores) / total_weight, 4
        )
        hallucination_probability = round(1.0 - trust_score, 4)
        reliability_score = round(self._compute_reliability(claims, claim_scores, total_weight), 4)

        avg_extraction_confidence = round(
            sum(claim.extraction_confidence for claim in claims) / len(claims), 4
        )
        confidence_score = round(
            (0.5 * avg_extraction_confidence) + (0.5 * reliability_score), 4
        )

        supported = sum(
            1 for c in claims if c.verification_status is VerificationStatus.SUPPORTED
        )
        contradicted = sum(
            1 for c in claims if c.verification_status is VerificationStatus.CONTRADICTED
        )
        insufficient = len(claims) - supported - contradicted

        result = ConfidenceScoreResult(
            trust_score=trust_score,
            reliability_score=reliability_score,
            hallucination_probability=hallucination_probability,
            confidence_score=confidence_score,
            risk_level=self._risk_level(hallucination_probability),
            total_claims=len(claims),
            supported_claims=supported,
            contradicted_claims=contradicted,
            insufficient_evidence_claims=insufficient,
            per_claim_scores=claim_scores,
        )

        logger.info(
            "Confidence scoring complete: trust=%.4f hallucination_prob=%.4f "
            "reliability=%.4f confidence=%.4f risk=%s (n=%d claims).",
            result.trust_score,
            result.hallucination_probability,
            result.reliability_score,
            result.confidence_score,
            result.risk_level.value,
            result.total_claims,
        )
        return result

    def _score_claim(self, claim: ClaimModel) -> ClaimScore:
        """Compute the weighted support contribution of a single claim."""
        support_value = self._SUPPORT_VALUES.get(claim.verification_status, 0.5)
        weight = max(claim.extraction_confidence, self._min_claim_weight)

        return ClaimScore(
            claim_id=claim.id,
            verification_status=claim.verification_status,
            support_value=support_value,
            weight=weight,
        )

    @staticmethod
    def _compute_reliability(
        claims: list[ClaimModel],
        claim_scores: tuple[ClaimScore, ...],
        total_weight: float,
    ) -> float:
        """
        Fraction of total claim weight backed by at least one retrieved
        evidence snippet, i.e. how "grounded" the verdict is rather than
        defaulting to uncertainty for lack of retrieval coverage.
        """
        evidenced_weight = sum(
            score.weight
            for claim, score in zip(claims, claim_scores)
            if claim.evidence
        )
        return evidenced_weight / total_weight if total_weight else 0.0

    def _risk_level(self, hallucination_probability: float) -> RiskLevel:
        """Map a numeric hallucination probability onto a coarse risk category."""
        low_upper, medium_upper = self._risk_thresholds

        if hallucination_probability < low_upper:
            return RiskLevel.LOW
        if hallucination_probability < medium_upper:
            return RiskLevel.MEDIUM
        return RiskLevel.HIGH