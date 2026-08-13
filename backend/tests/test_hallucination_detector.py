"""
Test suite for :mod:`app.services.hallucination_detector`.

Covers:
    * Normal cases   — claims fully supported by relevant evidence, and
                        claims that directly contradict evidence.
    * Edge cases      — no matching evidence at all, multiple candidate
                        passages, empty evidence source.
    * Invalid inputs  — empty claim list, invalid constructor thresholds.

Uses :class:`LexicalOverlapEvidenceSource` seeded with an in-memory corpus
rather than any external retrieval system, so these tests run fully offline
and deterministically. Requires the ``en_core_web_sm`` spaCy model
(``python -m spacy download en_core_web_sm``).
"""

from __future__ import annotations

import pytest

from app.models.claim_model import ClaimModel, Entity, VerificationStatus
from app.services.hallucination_detector import (
    HallucinationDetectionError,
    HallucinationDetector,
    LexicalOverlapEvidenceSource,
)


def _make_claim(claim_id: int, text: str) -> ClaimModel:
    """Construct a minimal, valid ClaimModel for detector-level testing."""
    return ClaimModel(id=claim_id, text=text, extraction_confidence=1.0)


def _make_claim_with_entities(
    claim_id: int,
    text: str,
    entities: list[tuple[str, str]],
) -> ClaimModel:
    """
    Construct a ClaimModel with explicit (text, spaCy label) entities.

    Bypasses the extractor's own NER so entity-disagreement tests are
    deterministic regardless of spaCy version/model drift, while still
    exercising the real HallucinationDetector (which still runs its own
    NER over the *evidence* text internally).
    """
    claim = _make_claim(claim_id, text)
    built_entities = []
    cursor = 0
    for entity_text, label in entities:
        start = text.find(entity_text, cursor)
        assert start != -1, f"{entity_text!r} not found in {text!r}"
        end = start + len(entity_text)
        built_entities.append(Entity(text=entity_text, label=label, start_char=start, end_char=end))
        cursor = end
    claim.entities = tuple(built_entities)
    return claim


class TestHallucinationDetectorNormalCases:
    """Claims with clearly supporting or clearly contradicting evidence."""

    def test_claim_supported_by_matching_evidence(self) -> None:
        corpus = ["The Eiffel Tower is located in Paris, France, and was completed in 1889."]
        detector = HallucinationDetector(
            evidence_source=LexicalOverlapEvidenceSource(corpus=corpus),
            support_threshold=0.15,
        )
        claim = _make_claim(1, "The Eiffel Tower is located in Paris, France.")

        outcomes = detector.detect([claim])

        assert len(outcomes) == 1
        assert claim.verification_status is VerificationStatus.SUPPORTED
        assert claim.verified is True
        assert claim.evidence

    def test_claim_contradicted_by_negated_evidence(self) -> None:
        corpus = ["The Eiffel Tower was not completed in 1889 according to updated records."]
        detector = HallucinationDetector(
            evidence_source=LexicalOverlapEvidenceSource(corpus=corpus),
            support_threshold=0.15,
            contradiction_threshold=0.1,
        )
        claim = _make_claim(1, "The Eiffel Tower was completed in 1889.")

        outcomes = detector.detect([claim])

        assert outcomes[0].negation_mismatch is True
        assert claim.verification_status is VerificationStatus.CONTRADICTED
        assert claim.verified is False

    def test_multiple_claims_processed_in_order(self) -> None:
        corpus = [
            "The Amazon River flows through Brazil.",
            "Mount Everest is located in the Himalayas.",
        ]
        detector = HallucinationDetector(
            evidence_source=LexicalOverlapEvidenceSource(corpus=corpus),
            support_threshold=0.1,
        )
        claims = [
            _make_claim(1, "The Amazon River flows through Brazil."),
            _make_claim(2, "Mount Everest is located in the Himalayas."),
        ]

        outcomes = detector.detect(claims)

        assert [o.claim_id for o in outcomes] == [1, 2]


class TestHallucinationDetectorEntityDisagreement:
    """
    Value-substitution hallucinations: claim and evidence are lexically/
    topically near-identical but assert a different named entity for the
    same factual slot (e.g. capital-of-India). Explicit negation words are
    absent, so only the entity-disagreement heuristic can catch these.
    """

    def test_conflicting_entity_value_is_contradicted(self) -> None:
        corpus = ["The capital of India is New Delhi."]
        detector = HallucinationDetector(
            evidence_source=LexicalOverlapEvidenceSource(corpus=corpus),
            support_threshold=0.35,
            contradiction_threshold=0.2,
        )
        claim = _make_claim_with_entities(
            1,
            "The capital of India is Mumbai.",
            entities=[("India", "GPE"), ("Mumbai", "GPE")],
        )

        outcomes = detector.detect([claim])

        assert outcomes[0].entity_disagreement is True
        assert claim.verification_status is VerificationStatus.CONTRADICTED
        assert claim.verified is False

    def test_matching_entity_value_remains_supported(self) -> None:
        """Guards against over-correcting: the true claim must not regress."""
        corpus = ["The capital of India is New Delhi."]
        detector = HallucinationDetector(
            evidence_source=LexicalOverlapEvidenceSource(corpus=corpus),
            support_threshold=0.35,
            contradiction_threshold=0.2,
        )
        claim = _make_claim_with_entities(
            1,
            "The capital of India is New Delhi.",
            entities=[("India", "GPE"), ("New Delhi", "GPE")],
        )

        outcomes = detector.detect([claim])

        assert outcomes[0].entity_disagreement is False
        assert claim.verification_status is VerificationStatus.SUPPORTED
        assert claim.verified is True

    def test_entity_type_absent_from_evidence_is_not_a_disagreement(self) -> None:
        """
        If the evidence doesn't mention any entity of that label at all,
        that's insufficient evidence, not a disagreement -- the heuristic
        must not fire on topics the evidence simply doesn't cover.
        """
        corpus = ["Python was created by Guido van Rossum and first released in 1991."]
        detector = HallucinationDetector(
            evidence_source=LexicalOverlapEvidenceSource(corpus=corpus),
            support_threshold=0.9,
        )
        claim = _make_claim_with_entities(
            1,
            "The largest planet in the solar system is Jupiter.",
            entities=[("Jupiter", "LOC")],
        )

        outcomes = detector.detect([claim])

        assert outcomes[0].entity_disagreement is False
        assert claim.verification_status is not VerificationStatus.CONTRADICTED


class TestHallucinationDetectorEdgeCases:
    """Boundary conditions around evidence availability."""

    def test_no_matching_evidence_yields_insufficient_evidence(self) -> None:
        corpus = ["Bananas are a good source of potassium."]
        detector = HallucinationDetector(evidence_source=LexicalOverlapEvidenceSource(corpus=corpus))
        claim = _make_claim(1, "Quantum computers use qubits instead of classical bits.")

        detector.detect([claim])

        assert claim.verification_status is VerificationStatus.INSUFFICIENT_EVIDENCE

    def test_empty_evidence_corpus_yields_insufficient_evidence(self) -> None:
        detector = HallucinationDetector(evidence_source=LexicalOverlapEvidenceSource(corpus=[]))
        claim = _make_claim(1, "The Great Barrier Reef is off the coast of Australia.")

        detector.detect([claim])

        assert claim.verification_status is VerificationStatus.INSUFFICIENT_EVIDENCE
        assert claim.evidence == ()

    def test_low_relevance_evidence_does_not_falsely_support(self) -> None:
        corpus = ["The stock market closed higher on Tuesday amid strong earnings."]
        detector = HallucinationDetector(
            evidence_source=LexicalOverlapEvidenceSource(corpus=corpus),
            support_threshold=0.5,
        )
        claim = _make_claim(1, "Octopuses have three hearts and blue blood.")

        detector.detect([claim])

        assert claim.verification_status is not VerificationStatus.SUPPORTED


class TestHallucinationDetectorInvalidInputs:
    """Invalid constructor arguments and empty batches."""

    def test_empty_claim_list_raises(self) -> None:
        detector = HallucinationDetector()

        with pytest.raises(HallucinationDetectionError):
            detector.detect([])

    @pytest.mark.parametrize("bad_threshold", [-0.1, 1.1])
    def test_invalid_support_threshold_raises(self, bad_threshold: float) -> None:
        with pytest.raises(HallucinationDetectionError):
            HallucinationDetector(support_threshold=bad_threshold)

    @pytest.mark.parametrize("bad_threshold", [-0.5, 2.0])
    def test_invalid_contradiction_threshold_raises(self, bad_threshold: float) -> None:
        with pytest.raises(HallucinationDetectionError):
            HallucinationDetector(contradiction_threshold=bad_threshold)


class TestLexicalOverlapEvidenceSource:
    """Direct tests of the default evidence retrieval implementation."""

    def test_retrieve_ranks_more_relevant_passage_first(self) -> None:
        source = LexicalOverlapEvidenceSource(
            corpus=[
                "Cats are small domesticated carnivorous mammals.",
                "The moon orbits the Earth roughly every 27 days.",
            ]
        )

        results = source.retrieve("Cats are domesticated mammals.", top_k=2)

        assert results
        assert "Cats" in results[0].text

    def test_retrieve_on_empty_corpus_returns_empty_list(self) -> None:
        source = LexicalOverlapEvidenceSource()

        assert source.retrieve("Anything at all.") == []

    def test_add_documents_extends_corpus(self) -> None:
        source = LexicalOverlapEvidenceSource()
        source.add_documents(["The Sahara desert covers much of North Africa."])

        results = source.retrieve("The Sahara desert covers North Africa.")

        assert results