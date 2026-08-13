"""
Test suite for :mod:`app.services.response_analyzer`.

Covers:
    * Normal cases   — full pipeline success with RELIABLE/QUESTIONABLE/
                        UNRELIABLE verdict derivation.
    * Edge cases      — no verifiable claims found, structured JSON output.
    * Invalid inputs  — None/whitespace-only text, and error propagation
                        from each pipeline stage (extraction, detection,
                        scoring) into ``ResponseAnalysisError``.

``ResponseAnalyzer`` depends only on the public interfaces of its three
collaborators (constructor injection), so this suite uses lightweight fakes
instead of the real spaCy-backed services — fast, deterministic, fully offline.
"""

from __future__ import annotations

from typing import Callable

import pytest

from app.models.claim_model import ClaimModel, VerificationStatus
from app.services.claim_extractor import ClaimExtractionError, ClaimType, ExtractedClaim
from app.services.confidence_scorer import (
    ClaimScore,
    ConfidenceScoreResult,
    ConfidenceScoringError,
    RiskLevel,
)
from app.services.hallucination_detector import (
    ClaimDetectionOutcome,
    EvidencePassage,
    HallucinationDetectionError,
)
from app.services.response_analyzer import (
    ResponseAnalysisError,
    ResponseAnalyzer,
    Verdict,
)


class FakeClaimExtractor:
    """Test double returning a pre-configured list of ExtractedClaim."""

    def __init__(
        self,
        extracted_claims: list[ExtractedClaim] | None = None,
        error: Exception | None = None,
    ) -> None:
        self._extracted_claims = extracted_claims if extracted_claims is not None else []
        self._error = error
        self.call_count = 0

    def extract(self, text: str) -> list[ExtractedClaim]:
        self.call_count += 1
        if self._error:
            raise self._error
        return self._extracted_claims


class FakeHallucinationDetector:
    """Test double that marks every claim with a fixed verification status."""

    def __init__(
        self,
        status: VerificationStatus = VerificationStatus.SUPPORTED,
        error: Exception | None = None,
    ) -> None:
        self._status = status
        self._error = error
        self.call_count = 0

    def detect(self, claims: list[ClaimModel]) -> list[ClaimDetectionOutcome]:
        self.call_count += 1
        if self._error:
            raise self._error

        outcomes: list[ClaimDetectionOutcome] = []
        for claim in claims:
            # Mirror the real HallucinationDetector.detect() contract: the
            # evidence passed to mark_verified() and the evidence attached
            # to the returned ClaimDetectionOutcome must be the same
            # passages, since ResponseAnalyzer now serializes API evidence
            # from the outcome (matched by claim_id), not from
            # ClaimModel.evidence.
            evidence_passages = (
                EvidencePassage(
                    text="fake evidence",
                    source="fake_source.txt",
                    relevance_score=1.0,
                ),
            )
            claim.mark_verified(
                self._status,
                evidence=tuple(p.text for p in evidence_passages),
            )
            outcomes.append(
                ClaimDetectionOutcome(
                    claim_id=claim.id,
                    verification_status=self._status,
                    support_score=1.0 if self._status is VerificationStatus.SUPPORTED else 0.0,
                    entity_agreement=1.0,
                    negation_mismatch=self._status is VerificationStatus.CONTRADICTED,
                    entity_disagreement=False,
                    evidence=evidence_passages,
                )
            )
        return outcomes


class FakeConfidenceScorer:
    """Test double returning either a fixed factory-built result or a default one."""

    def __init__(
        self,
        result_factory: Callable[[list[ClaimModel]], ConfidenceScoreResult] | None = None,
        error: Exception | None = None,
    ) -> None:
        self._result_factory = result_factory
        self._error = error
        self.call_count = 0

    def score(self, claims: list[ClaimModel]) -> ConfidenceScoreResult:
        self.call_count += 1
        if self._error:
            raise self._error
        if self._result_factory:
            return self._result_factory(claims)
        return _default_result(claims)


def _default_result(
    claims: list[ClaimModel], risk_level: RiskLevel = RiskLevel.LOW
) -> ConfidenceScoreResult:
    supported = sum(1 for c in claims if c.verification_status is VerificationStatus.SUPPORTED)
    contradicted = sum(1 for c in claims if c.verification_status is VerificationStatus.CONTRADICTED)
    return ConfidenceScoreResult(
        trust_score=1.0 if contradicted == 0 else 0.0,
        reliability_score=1.0,
        hallucination_probability=0.0 if contradicted == 0 else 1.0,
        confidence_score=1.0,
        risk_level=risk_level,
        total_claims=len(claims),
        supported_claims=supported,
        contradicted_claims=contradicted,
        insufficient_evidence_claims=len(claims) - supported - contradicted,
        per_claim_scores=tuple(
            ClaimScore(
                claim_id=c.id,
                verification_status=c.verification_status,
                support_value=1.0,
                weight=1.0,
            )
            for c in claims
        ),
    )


def _make_extracted_claim(claim_id: int, text: str) -> ExtractedClaim:
    return ExtractedClaim(
        claim=ClaimModel(id=claim_id, text=text),
        claim_type=ClaimType.FACTUAL,
        entities=(),
        extraction_confidence=1.0,
    )


class TestResponseAnalyzerNormalCases:
    """Full pipeline runs producing each verdict category."""

    def test_all_supported_claims_yield_reliable_verdict(self) -> None:
        analyzer = ResponseAnalyzer(
            claim_extractor=FakeClaimExtractor([_make_extracted_claim(1, "Paris is the capital of France.")]),
            hallucination_detector=FakeHallucinationDetector(VerificationStatus.SUPPORTED),
            confidence_scorer=FakeConfidenceScorer(),
        )

        analysis = analyzer.analyze("Paris is the capital of France.")

        assert analysis.verdict is Verdict.RELIABLE
        assert analysis.confidence is not None
        assert analysis.confidence.trust_score == pytest.approx(1.0)

    def test_contradicted_claim_with_moderate_risk_yields_questionable(self) -> None:
        analyzer = ResponseAnalyzer(
            claim_extractor=FakeClaimExtractor([_make_extracted_claim(1, "The event happened in 1990.")]),
            hallucination_detector=FakeHallucinationDetector(VerificationStatus.CONTRADICTED),
            confidence_scorer=FakeConfidenceScorer(
                result_factory=lambda claims: _default_result(claims, risk_level=RiskLevel.MEDIUM)
            ),
        )

        analysis = analyzer.analyze("The event happened in 1990.")

        assert analysis.verdict is Verdict.QUESTIONABLE
        assert "contradict" in analysis.verdict_reason.lower()

    def test_contradicted_claim_with_high_risk_stays_unreliable(self) -> None:
        analyzer = ResponseAnalyzer(
            claim_extractor=FakeClaimExtractor([_make_extracted_claim(1, "The event happened in 1990.")]),
            hallucination_detector=FakeHallucinationDetector(VerificationStatus.CONTRADICTED),
            confidence_scorer=FakeConfidenceScorer(
                result_factory=lambda claims: _default_result(claims, risk_level=RiskLevel.HIGH)
            ),
        )

        analysis = analyzer.analyze("The event happened in 1990.")

        assert analysis.verdict is Verdict.UNRELIABLE


class TestResponseAnalyzerEdgeCases:
    """No-claims path and structured output shape."""

    def test_no_claims_found_skips_detection_and_scoring(self) -> None:
        extractor = FakeClaimExtractor([])
        detector = FakeHallucinationDetector()
        scorer = FakeConfidenceScorer()
        analyzer = ResponseAnalyzer(
            claim_extractor=extractor,
            hallucination_detector=detector,
            confidence_scorer=scorer,
        )

        analysis = analyzer.analyze("Hello! Thanks for your help.")

        assert analysis.verdict is Verdict.NO_VERIFIABLE_CLAIMS
        assert analysis.confidence is None
        assert detector.call_count == 0
        assert scorer.call_count == 0

    def test_to_dict_contains_expected_top_level_keys(self) -> None:
        analyzer = ResponseAnalyzer(
            claim_extractor=FakeClaimExtractor([_make_extracted_claim(1, "Water is H2O.")]),
            hallucination_detector=FakeHallucinationDetector(VerificationStatus.SUPPORTED),
            confidence_scorer=FakeConfidenceScorer(),
        )

        analysis = analyzer.analyze("Water is H2O.")
        payload = analysis.to_dict()

        assert set(payload.keys()) == {
            "verdict",
            "verdict_reason",
            "analyzed_at",
            "scores",
            "claim_summary",
            "claims",
        }
        assert payload["claim_summary"]["total_claims"] == 1


class TestResponseAnalyzerInvalidInputs:
    """Empty input and error propagation from each pipeline stage."""

    def test_none_input_raises(self) -> None:
        analyzer = ResponseAnalyzer(
            claim_extractor=FakeClaimExtractor(),
            hallucination_detector=FakeHallucinationDetector(),
            confidence_scorer=FakeConfidenceScorer(),
        )

        with pytest.raises(ResponseAnalysisError):
            analyzer.analyze(None)  # type: ignore[arg-type]

    def test_whitespace_only_input_raises(self) -> None:
        analyzer = ResponseAnalyzer(
            claim_extractor=FakeClaimExtractor(),
            hallucination_detector=FakeHallucinationDetector(),
            confidence_scorer=FakeConfidenceScorer(),
        )

        with pytest.raises(ResponseAnalysisError):
            analyzer.analyze("   ")

    def test_extraction_failure_is_wrapped(self) -> None:
        analyzer = ResponseAnalyzer(
            claim_extractor=FakeClaimExtractor(error=ClaimExtractionError("boom")),
            hallucination_detector=FakeHallucinationDetector(),
            confidence_scorer=FakeConfidenceScorer(),
        )

        with pytest.raises(ResponseAnalysisError):
            analyzer.analyze("Some response text.")

    def test_detection_failure_is_wrapped(self) -> None:
        analyzer = ResponseAnalyzer(
            claim_extractor=FakeClaimExtractor([_make_extracted_claim(1, "Some claim.")]),
            hallucination_detector=FakeHallucinationDetector(error=HallucinationDetectionError("boom")),
            confidence_scorer=FakeConfidenceScorer(),
        )

        with pytest.raises(ResponseAnalysisError):
            analyzer.analyze("Some response text.")

    def test_scoring_failure_is_wrapped(self) -> None:
        analyzer = ResponseAnalyzer(
            claim_extractor=FakeClaimExtractor([_make_extracted_claim(1, "Some claim.")]),
            hallucination_detector=FakeHallucinationDetector(),
            confidence_scorer=FakeConfidenceScorer(error=ConfidenceScoringError("boom")),
        )

        with pytest.raises(ResponseAnalysisError):
            analyzer.analyze("Some response text.")


class TestResponseAnalyzerEvidenceSerialization:
    """
    Regression coverage for the source-attribution integration fix:

    * API evidence must be built from ``detection_outcomes`` (matched by
      ``claim_id``), carrying text + source + similarity_score.
    * ``_apply_detection_outcomes`` must match by ``claim_id``, not by the
      previously-broken text matching (which could never find a match
      because ``ClaimDetectionOutcome`` carries no claim text).
    * ``ClaimModel.source`` must be backfilled from the best-ranked
      evidence passage.
    """

    def test_to_dict_evidence_carries_source_and_similarity_score(self) -> None:
        analyzer = ResponseAnalyzer(
            claim_extractor=FakeClaimExtractor(
                [_make_extracted_claim(1, "The capital of India is New Delhi.")]
            ),
            hallucination_detector=FakeHallucinationDetector(VerificationStatus.SUPPORTED),
            confidence_scorer=FakeConfidenceScorer(),
        )

        analysis = analyzer.analyze("The capital of India is New Delhi.")
        payload = analysis.to_dict()

        claim_payload = payload["claims"][0]
        assert claim_payload["evidence"] == [
            {
                "text": "fake evidence",
                "source": "fake_source.txt",
                "similarity_score": 1.0,
            }
        ]
        # claim-level `source` is backfilled from the best evidence passage
        # instead of always being null.
        assert claim_payload["source"] == "fake_source.txt"

    def test_claim_id_matching_survives_out_of_order_outcomes(self) -> None:
        """
        ``_apply_detection_outcomes`` must key strictly on claim_id, so it
        still matches correctly even if outcomes were ever produced in a
        different order than the claims (the old text-matching approach
        could not do this reliably at all).
        """
        claim_a = ClaimModel(id=1, text="Claim A text.", extraction_confidence=1.0)
        claim_b = ClaimModel(id=2, text="Claim B text.", extraction_confidence=1.0)

        outcome_b = ClaimDetectionOutcome(
            claim_id=2,
            verification_status=VerificationStatus.CONTRADICTED,
            support_score=0.9,
            entity_agreement=0.5,
            negation_mismatch=False,
            entity_disagreement=True,
            evidence=(EvidencePassage(text="evidence b", source="doc_b.txt", relevance_score=0.9),),
        )
        outcome_a = ClaimDetectionOutcome(
            claim_id=1,
            verification_status=VerificationStatus.SUPPORTED,
            support_score=1.0,
            entity_agreement=1.0,
            negation_mismatch=False,
            entity_disagreement=False,
            evidence=(EvidencePassage(text="evidence a", source="doc_a.txt", relevance_score=1.0),),
        )

        updated = ResponseAnalyzer._apply_detection_outcomes(
            (claim_a, claim_b),
            [outcome_b, outcome_a],  # deliberately out of order
        )

        updated_a = next(c for c in updated if c.id == 1)
        updated_b = next(c for c in updated if c.id == 2)

        assert updated_a.verification_status is VerificationStatus.SUPPORTED
        assert updated_a.source == "doc_a.txt"

        assert updated_b.verification_status is VerificationStatus.CONTRADICTED
        assert updated_b.source == "doc_b.txt"

    def test_claim_without_matching_outcome_is_left_unchanged(self) -> None:
        claim = ClaimModel(id=99, text="Unmatched claim.", extraction_confidence=1.0)
        claim.mark_verified(VerificationStatus.UNVERIFIED)

        outcome_for_other_claim = ClaimDetectionOutcome(
            claim_id=1,
            verification_status=VerificationStatus.SUPPORTED,
            support_score=1.0,
            entity_agreement=1.0,
            negation_mismatch=False,
            entity_disagreement=False,
            evidence=(),
        )

        updated = ResponseAnalyzer._apply_detection_outcomes(
            (claim,),
            [outcome_for_other_claim],
        )

        assert updated[0].verification_status is VerificationStatus.UNVERIFIED