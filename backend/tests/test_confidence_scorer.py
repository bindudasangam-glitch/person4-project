"""
Test suite for :mod:`app.services.confidence_scorer`.

Covers:
    * Normal cases   — all-supported, all-contradicted, and mixed-verdict
                        claim batches, and correct risk-level derivation.
    * Edge cases      — single claim, low-confidence claim weight flooring,
                        reliability score reflecting evidence coverage.
    * Invalid inputs  — empty claim list, invalid constructor arguments.

No spaCy dependency — ``ConfidenceScorer`` operates purely on already
verified ``ClaimModel`` instances, so these tests are fast and fully offline.
"""

from __future__ import annotations

import pytest

from app.models.claim_model import ClaimModel, VerificationStatus
from app.services.confidence_scorer import (
    ConfidenceScorer,
    ConfidenceScoringError,
    RiskLevel,
)


def _verified_claim(
    claim_id: int,
    status: VerificationStatus,
    extraction_confidence: float = 1.0,
    with_evidence: bool = True,
) -> ClaimModel:
    """Build a ClaimModel already at a given verification status, as if post-detection."""
    claim = ClaimModel(
        id=claim_id,
        text=f"Claim number {claim_id}.",
        extraction_confidence=extraction_confidence,
    )
    claim.mark_verified(status, evidence=("supporting snippet",) if with_evidence else ())
    return claim


class TestConfidenceScorerNormalCases:
    """Straightforward batches with a clear overall verdict."""

    def test_all_supported_claims_yield_high_trust(self) -> None:
        scorer = ConfidenceScorer()
        claims = [
            _verified_claim(1, VerificationStatus.SUPPORTED),
            _verified_claim(2, VerificationStatus.SUPPORTED),
        ]

        result = scorer.score(claims)

        assert result.trust_score == pytest.approx(1.0)
        assert result.hallucination_probability == pytest.approx(0.0)
        assert result.risk_level is RiskLevel.LOW
        assert result.supported_claims == 2
        assert result.contradicted_claims == 0

    def test_all_contradicted_claims_yield_high_risk(self) -> None:
        scorer = ConfidenceScorer()
        claims = [
            _verified_claim(1, VerificationStatus.CONTRADICTED),
            _verified_claim(2, VerificationStatus.CONTRADICTED),
        ]

        result = scorer.score(claims)

        assert result.trust_score == pytest.approx(0.0)
        assert result.hallucination_probability == pytest.approx(1.0)
        assert result.risk_level is RiskLevel.HIGH

    def test_mixed_verdicts_yield_intermediate_scores(self) -> None:
        scorer = ConfidenceScorer()
        claims = [
            _verified_claim(1, VerificationStatus.SUPPORTED),
            _verified_claim(2, VerificationStatus.CONTRADICTED),
        ]

        result = scorer.score(claims)

        assert 0.0 < result.trust_score < 1.0
        assert result.total_claims == 2
        assert len(result.per_claim_scores) == 2

    def test_reliability_score_reflects_evidence_coverage(self) -> None:
        scorer = ConfidenceScorer()
        claims = [
            _verified_claim(1, VerificationStatus.SUPPORTED, with_evidence=True),
            _verified_claim(2, VerificationStatus.INSUFFICIENT_EVIDENCE, with_evidence=False),
        ]

        result = scorer.score(claims)

        assert 0.0 < result.reliability_score < 1.0


class TestConfidenceScorerEdgeCases:
    """Boundary conditions around claim weighting and batch size."""

    def test_single_claim_batch(self) -> None:
        scorer = ConfidenceScorer()
        claims = [_verified_claim(1, VerificationStatus.SUPPORTED)]

        result = scorer.score(claims)

        assert result.total_claims == 1
        assert result.trust_score == pytest.approx(1.0)

    def test_low_extraction_confidence_is_floored_not_zeroed(self) -> None:
        scorer = ConfidenceScorer(min_claim_weight=0.2)
        claims = [_verified_claim(1, VerificationStatus.SUPPORTED, extraction_confidence=0.01)]

        result = scorer.score(claims)

        assert result.per_claim_scores[0].weight == pytest.approx(0.2)

    def test_insufficient_evidence_claims_score_as_uncertain(self) -> None:
        scorer = ConfidenceScorer()
        claims = [_verified_claim(1, VerificationStatus.INSUFFICIENT_EVIDENCE)]

        result = scorer.score(claims)

        assert result.trust_score == pytest.approx(0.5)
        assert result.risk_level is RiskLevel.MEDIUM

    def test_risk_thresholds_are_configurable(self) -> None:
        scorer = ConfidenceScorer(risk_thresholds=(0.1, 0.2))
        claims = [_verified_claim(1, VerificationStatus.INSUFFICIENT_EVIDENCE)]

        result = scorer.score(claims)

        # hallucination_probability of 0.5 exceeds the tightened 0.2 medium bound.
        assert result.risk_level is RiskLevel.HIGH


class TestConfidenceScorerInvalidInputs:
    """Invalid constructor arguments and empty batches."""

    def test_empty_claim_list_raises(self) -> None:
        scorer = ConfidenceScorer()

        with pytest.raises(ConfidenceScoringError):
            scorer.score([])

    @pytest.mark.parametrize("bad_weight", [0.0, -0.1, 1.5])
    def test_invalid_min_claim_weight_raises(self, bad_weight: float) -> None:
        with pytest.raises(ConfidenceScoringError):
            ConfidenceScorer(min_claim_weight=bad_weight)

    @pytest.mark.parametrize(
        "bad_thresholds",
        [(0.6, 0.3), (0.5, 0.5), (-0.1, 0.5), (0.5, 1.2)],
    )
    def test_invalid_risk_thresholds_raise(self, bad_thresholds: tuple[float, float]) -> None:
        with pytest.raises(ConfidenceScoringError):
            ConfidenceScorer(risk_thresholds=bad_thresholds)