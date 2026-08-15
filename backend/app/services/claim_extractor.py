"""
Claim Extraction Service
========================

Extracts discrete, verifiable factual claims from an LLM-generated response.

Pipeline
--------
1. Text normalization via TextCleaner.
2. Sentence segmentation via the shared spaCy pipeline.
3. Named Entity Recognition (NER) per sentence.
4. Claim detection using linguistic heuristics.
5. Claim classification.
6. Claim object construction.

IMPORTANT
---------
The spaCy model is NOT loaded separately by this service.

ClaimExtractor uses the shared spaCy pipeline from:

    app.utils.nlp_pipeline.get_spacy_pipeline

This prevents ClaimExtractor and HallucinationDetector from loading
two separate copies of en_core_web_sm into memory.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Final

from spacy.language import Language
from spacy.tokens import Span

from app.core.logging import logger
from app.models.claim_model import ClaimModel
from app.utils.nlp_pipeline import get_spacy_pipeline
from app.utils.text_cleaner import TextCleaner


__all__ = [
    "ClaimType",
    "ExtractedEntity",
    "ExtractedClaim",
    "ClaimExtractionError",
    "ClaimExtractor",
]


# ============================================================================
# EXCEPTIONS
# ============================================================================


class ClaimExtractionError(Exception):
    """Raised when claim extraction cannot be completed."""


# ============================================================================
# CLAIM TYPE
# ============================================================================


class ClaimType(str, Enum):
    """Coarse-grained classification of a factual claim."""

    NUMERIC = "numeric"
    TEMPORAL = "temporal"
    ENTITY_CENTRIC = "entity_centric"
    FACTUAL = "factual"
    OPINION = "opinion"
    UNVERIFIABLE = "unverifiable"


# ============================================================================
# EXTRACTED ENTITY
# ============================================================================


@dataclass(frozen=True, slots=True)
class ExtractedEntity:
    """A named entity recognized within a claim."""

    text: str
    label: str
    start_char: int
    end_char: int


# ============================================================================
# EXTRACTED CLAIM
# ============================================================================


@dataclass(frozen=True, slots=True)
class ExtractedClaim:
    """
    Rich, NLP-annotated representation of a single extracted claim.

    Wraps the lightweight persistence model with additional metadata
    required by downstream hallucination detection and confidence scoring.
    """

    claim: ClaimModel
    claim_type: ClaimType
    entities: tuple[ExtractedEntity, ...] = field(
        default_factory=tuple
    )
    extraction_confidence: float = 1.0

    @property
    def id(self) -> int:
        """Return the claim identifier."""

        return self.claim.id

    @property
    def text(self) -> str:
        """Return the claim text."""

        return self.claim.text


# ============================================================================
# CLAIM EXTRACTOR
# ============================================================================


class ClaimExtractor:
    """
    Extracts and classifies factual claims from LLM-generated text.

    The shared spaCy pipeline is loaded lazily through
    app.utils.nlp_pipeline.get_spacy_pipeline().
    """

    _MIN_CLAIM_TOKEN_LENGTH: Final[int] = 3

    # ------------------------------------------------------------------------
    # Opinion cues
    # ------------------------------------------------------------------------

    _OPINION_CUES: Final[tuple[str, ...]] = (
        "i think",
        "i believe",
        "in my opinion",
        "it seems",
        "arguably",
        "i feel",
        "probably",
        "perhaps",
        "maybe",
        "it is possible that",
    )

    # ------------------------------------------------------------------------
    # Non-claim patterns
    # ------------------------------------------------------------------------

    _NON_CLAIM_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
        re.compile(
            r"^(please|note that|remember|let me|i'll|i will)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"^(hi|hello|hey|thanks|thank you)\b",
            re.IGNORECASE,
        ),
    )

    # ------------------------------------------------------------------------
    # Shared spaCy pipeline
    # ------------------------------------------------------------------------

    _nlp: Language | None = None

    # ------------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------------

    def __init__(self) -> None:
        """
        Initialize the extractor.

        The shared spaCy model is obtained lazily.
        """

        self._ensure_pipeline_loaded()

    # ------------------------------------------------------------------------
    # SPA CY PIPELINE
    # ------------------------------------------------------------------------

    @classmethod
    def _ensure_pipeline_loaded(cls) -> None:
        """
        Obtain the shared spaCy pipeline.

        The actual spaCy model is owned by
        app.utils.nlp_pipeline so ClaimExtractor and
        HallucinationDetector share one model instance.
        """

        if cls._nlp is not None:
            return

        try:
            logger.info(
                "Initializing ClaimExtractor with shared spaCy pipeline."
            )

            cls._nlp = get_spacy_pipeline()

            logger.info(
                "ClaimExtractor connected to shared spaCy pipeline."
            )

        except Exception as exc:
            logger.exception(
                "Failed to load shared spaCy pipeline."
            )

            raise ClaimExtractionError(
                "Required spaCy pipeline could not be loaded."
            ) from exc

    # ------------------------------------------------------------------------
    # EXTRACTION
    # ------------------------------------------------------------------------

    def extract(
        self,
        text: str,
    ) -> list[ExtractedClaim]:
        """
        Extract and classify factual claims from text.

        Returns claims in document order while excluding sentences
        considered non-factual.
        """

        if text is None or not text.strip():
            logger.warning(
                "ClaimExtractor.extract called with empty input."
            )

            raise ClaimExtractionError(
                "Cannot extract claims from empty text."
            )

        cleaned_text = TextCleaner.clean(
            text
        )

        self._ensure_pipeline_loaded()

        if self._nlp is None:
            raise ClaimExtractionError(
                "spaCy pipeline is not available."
            )

        try:
            doc = self._nlp(
                cleaned_text
            )

        except Exception as exc:
            logger.exception(
                "spaCy pipeline failed while processing input."
            )

            raise ClaimExtractionError(
                "Failed to process text with NLP pipeline."
            ) from exc

        claims: list[ExtractedClaim] = []

        claim_id = 1

        for sentence in doc.sents:

            normalized = sentence.text.strip()

            if not self._is_candidate_claim(
                sentence,
                normalized,
            ):
                continue

            entities = self._extract_entities(
                sentence
            )

            claim_type = self._classify_claim(
                sentence,
                normalized,
                entities,
            )

            confidence = (
                self._estimate_extraction_confidence(
                    sentence,
                    entities,
                )
            )

            claims.append(
                ExtractedClaim(
                    claim=ClaimModel(
                        id=claim_id,
                        text=normalized,
                    ),
                    claim_type=claim_type,
                    entities=tuple(
                        entities
                    ),
                    extraction_confidence=confidence,
                )
            )

            claim_id += 1

        logger.info(
            "Extracted %d claim(s) from %d sentence(s).",
            len(claims),
            sum(
                1
                for _ in doc.sents
            ),
        )

        return claims

    # ------------------------------------------------------------------------
    # CANDIDATE CLAIM DETECTION
    # ------------------------------------------------------------------------

    def _is_candidate_claim(
        self,
        sentence: Span,
        normalized_text: str,
    ) -> bool:
        """
        Determine whether a sentence contains independently
        checkable content.
        """

        if len(sentence) < self._MIN_CLAIM_TOKEN_LENGTH:
            return False

        if normalized_text.endswith("?"):
            return False

        lowered = normalized_text.lower()

        if lowered.startswith(
            self._OPINION_CUES
        ):
            return False

        if any(
            pattern.match(normalized_text)
            for pattern in self._NON_CLAIM_PATTERNS
        ):
            return False

        has_verb = any(
            token.pos_ in (
                "VERB",
                "AUX",
            )
            for token in sentence
        )

        if not has_verb:
            return False

        return True

    # ------------------------------------------------------------------------
    # ENTITY EXTRACTION
    # ------------------------------------------------------------------------

    @staticmethod
    def _extract_entities(
        sentence: Span,
    ) -> list[ExtractedEntity]:
        """Extract named entities from a spaCy sentence span."""

        return [
            ExtractedEntity(
                text=ent.text,
                label=ent.label_,
                start_char=ent.start_char,
                end_char=ent.end_char,
            )
            for ent in sentence.ents
        ]

    # ------------------------------------------------------------------------
    # CLAIM CLASSIFICATION
    # ------------------------------------------------------------------------

    @staticmethod
    def _classify_claim(
        sentence: Span,
        normalized_text: str,
        entities: list[ExtractedEntity],
    ) -> ClaimType:
        """Assign a coarse ClaimType to a candidate claim."""

        lowered = normalized_text.lower()

        if (
            lowered.startswith(
                ClaimExtractor._OPINION_CUES
            )
            or " i think " in f" {lowered} "
        ):
            return ClaimType.OPINION

        entity_labels = {
            entity.label
            for entity in entities
        }

        if entity_labels & {
            "DATE",
            "TIME",
        }:
            return ClaimType.TEMPORAL

        if entity_labels & {
            "CARDINAL",
            "QUANTITY",
            "PERCENT",
            "MONEY",
            "ORDINAL",
        }:
            return ClaimType.NUMERIC

        if entity_labels & {
            "PERSON",
            "ORG",
            "GPE",
            "PRODUCT",
            "EVENT",
            "NORP",
            "FAC",
        }:
            return ClaimType.ENTITY_CENTRIC

        if entities:
            return ClaimType.FACTUAL

        return ClaimType.UNVERIFIABLE

    # ------------------------------------------------------------------------
    # EXTRACTION CONFIDENCE
    # ------------------------------------------------------------------------

    @staticmethod
    def _estimate_extraction_confidence(
        sentence: Span,
        entities: list[ExtractedEntity],
    ) -> float:
        """
        Estimate confidence that a sentence is a well-formed,
        independently checkable claim.
        """

        token_count = len(sentence)

        length_score = min(
            token_count / 12.0,
            1.0,
        )

        entity_score = min(
            len(entities) / 3.0,
            1.0,
        )

        confidence = round(
            (0.6 * length_score)
            + (0.4 * entity_score),
            4,
        )

        return max(
            confidence,
            0.1,
        )