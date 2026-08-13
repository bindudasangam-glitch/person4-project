"""
Test suite for :mod:`app.services.claim_extractor`.

Covers:
    * Normal cases   — well-formed multi-sentence factual text.
    * Edge cases      — short sentences, entity-free sentences, questions,
                        greetings, opinions, and singleton input.
    * Invalid inputs  — ``None``, empty string, whitespace-only string.

Requires the ``en_core_web_sm`` spaCy model to be installed
(``python -m spacy download en_core_web_sm``), as declared in
``backend/requirements.txt``. The extractor is expensive to instantiate
(model load), so a single session-scoped fixture is shared across all tests.
"""

from __future__ import annotations

import pytest

from app.services.claim_extractor import (
    ClaimExtractionError,
    ClaimExtractor,
    ClaimType,
    ExtractedClaim,
)


@pytest.fixture(scope="session")
def extractor() -> ClaimExtractor:
    """Provide a single shared ClaimExtractor instance for the test session."""
    return ClaimExtractor()


class TestClaimExtractorNormalCases:
    """Well-formed, multi-sentence factual input."""

    def test_extracts_all_factual_sentences(self, extractor: ClaimExtractor) -> None:
        text = (
            "Python was created by Guido van Rossum in 1991. "
            "ChatGPT was developed by OpenAI. "
            "The Eiffel Tower is located in Paris."
        )

        claims = extractor.extract(text)

        assert len(claims) == 3
        assert all(isinstance(claim, ExtractedClaim) for claim in claims)

    def test_claim_ids_are_sequential_in_document_order(self, extractor: ClaimExtractor) -> None:
        text = "The Nile is the longest river in Africa. Mount Everest is the tallest mountain on Earth."

        claims = extractor.extract(text)

        assert [claim.id for claim in claims] == list(range(1, len(claims) + 1))

    def test_claim_text_matches_source_sentence(self, extractor: ClaimExtractor) -> None:
        text = "The Great Wall of China was built over many centuries."

        claims = extractor.extract(text)

        assert len(claims) == 1
        assert claims[0].text == text

    def test_temporal_claim_is_classified_as_temporal(self, extractor: ClaimExtractor) -> None:
        text = "World War II ended in 1945."

        claims = extractor.extract(text)

        assert len(claims) == 1
        assert claims[0].claim_type is ClaimType.TEMPORAL

    def test_entity_centric_claim_carries_named_entities(self, extractor: ClaimExtractor) -> None:
        text = "Marie Curie was born in Warsaw."

        claims = extractor.extract(text)

        assert len(claims) == 1
        assert len(claims[0].entities) >= 1
        entity_texts = {entity.text for entity in claims[0].entities}
        assert "Marie Curie" in entity_texts or "Warsaw" in entity_texts

    def test_numeric_claim_is_classified_as_numeric(self, extractor: ClaimExtractor) -> None:
        text = "The company reported a profit of 50 million dollars."

        claims = extractor.extract(text)

        assert len(claims) == 1
        assert claims[0].claim_type is ClaimType.NUMERIC


class TestClaimExtractorEdgeCases:
    """Boundary conditions and sentences that should be filtered out."""

    def test_single_sentence_input(self, extractor: ClaimExtractor) -> None:
        claims = extractor.extract("The Amazon rainforest spans multiple countries.")

        assert len(claims) == 1

    def test_questions_are_filtered_out(self, extractor: ClaimExtractor) -> None:
        text = "What year was Python released? Python was released in 1991."

        claims = extractor.extract(text)

        assert len(claims) == 1
        assert not claims[0].text.endswith("?")

    def test_greetings_are_filtered_out(self, extractor: ClaimExtractor) -> None:
        text = "Hello there! The Sahara is the largest hot desert in the world."

        claims = extractor.extract(text)

        assert len(claims) == 1
        assert "hello" not in claims[0].text.lower()

    def test_opinions_are_filtered_out(self, extractor: ClaimExtractor) -> None:
        text = "I think Python is the best language. Python was first released in 1991."

        claims = extractor.extract(text)

        assert all("i think" not in claim.text.lower() for claim in claims)

    def test_very_short_fragments_are_filtered_out(self, extractor: ClaimExtractor) -> None:
        text = "Yes. The Pacific Ocean is the largest ocean on Earth."

        claims = extractor.extract(text)

        assert all(len(claim.text.split()) >= 3 for claim in claims)

    def test_entity_free_factual_sentence_still_extracted(self, extractor: ClaimExtractor) -> None:
        text = "Water boils when it reaches its boiling point."

        claims = extractor.extract(text)

        assert len(claims) == 1
        assert claims[0].claim_type in {ClaimType.FACTUAL, ClaimType.UNVERIFIABLE}

    def test_extraction_confidence_is_within_valid_bounds(self, extractor: ClaimExtractor) -> None:
        text = "The Berlin Wall fell in 1989, marking a major turning point in European history."

        claims = extractor.extract(text)

        assert len(claims) == 1
        assert 0.0 <= claims[0].extraction_confidence <= 1.0


class TestClaimExtractorInvalidInputs:
    """Inputs that should raise :class:`ClaimExtractionError`."""

    def test_none_input_raises(self, extractor: ClaimExtractor) -> None:
        with pytest.raises(ClaimExtractionError):
            extractor.extract(None)  # type: ignore[arg-type]

    def test_empty_string_raises(self, extractor: ClaimExtractor) -> None:
        with pytest.raises(ClaimExtractionError):
            extractor.extract("")

    def test_whitespace_only_string_raises(self, extractor: ClaimExtractor) -> None:
        with pytest.raises(ClaimExtractionError):
            extractor.extract("   \n\t  ")

    @pytest.mark.parametrize("non_claim_text", ["Hi!", "Thanks.", "Please continue."])
    def test_pure_non_claim_text_yields_no_claims(
        self, extractor: ClaimExtractor, non_claim_text: str
    ) -> None:
        claims = extractor.extract(non_claim_text)

        assert claims == []