"""
Claim Extraction Service
=========================

Extracts discrete, verifiable factual claims from an LLM-generated response.

Pipeline
--------
1. Text normalization (delegated to :class:`app.utils.text_cleaner.TextCleaner`).
2. Sentence segmentation via spaCy's statistical sentence boundary detector.
3. Named Entity Recognition (NER) per sentence.
4. Claim detection — filtering out non-factual sentences (questions, greetings,
   hedged/opinion statements, instructions) using linguistic heuristics built
   on top of spaCy's dependency parse and POS tags.
5. Claim classification — tagging each surviving sentence with a
   :class:`ClaimType` (FACTUAL, NUMERIC, TEMPORAL, ENTITY_CENTRIC, OPINION,
   UNVERIFIABLE) based on entity composition and linguistic cues.
6. Claim object construction — each claim is returned as an immutable,
   serializable :class:`ExtractedClaim`, which wraps the persistence-layer
   :class:`app.models.claim_model.ClaimModel` together with the richer
   NLP metadata (entities, claim type, extraction confidence) required by
   downstream services (hallucination detection, confidence scoring).

Design notes
------------
* The spaCy ``Language`` pipeline is expensive to load, so it is loaded lazily
  and cached at the class level (shared across all instances/requests within
  a worker process) rather than re-loaded per call.
* All public methods are defensive: they validate input, never raise raw
  library exceptions to callers, and log at appropriate levels.
* The class is stateless w.r.t. any single extraction call, making it safe to
  reuse across concurrent requests (FastAPI runs sync def endpoints in a
  thread pool; the shared spaCy pipeline is thread-safe for inference).
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Final

import spacy
from spacy.language import Language
from spacy.tokens import Doc, Span

from app.core.logging import logger
from app.models.claim_model import ClaimModel
from app.utils.text_cleaner import TextCleaner

__all__ = [
    "ClaimType",
    "ExtractedEntity",
    "ExtractedClaim",
    "ClaimExtractionError",
    "ClaimExtractor",
]


class ClaimExtractionError(Exception):
    """Raised when claim extraction cannot be completed for the given input."""


class ClaimType(str, Enum):
    """Coarse-grained classification of a factual claim."""

    NUMERIC = "numeric"
    TEMPORAL = "temporal"
    ENTITY_CENTRIC = "entity_centric"
    FACTUAL = "factual"
    OPINION = "opinion"
    UNVERIFIABLE = "unverifiable"


@dataclass(frozen=True, slots=True)
class ExtractedEntity:
    """A named entity recognized within a claim."""

    text: str
    label: str
    start_char: int
    end_char: int


@dataclass(frozen=True, slots=True)
class ExtractedClaim:
    """
    Rich, NLP-annotated representation of a single extracted claim.

    Wraps the lightweight persistence model (:class:`ClaimModel`) with the
    additional metadata required by downstream hallucination detection and
    confidence scoring services.
    """

    claim: ClaimModel
    claim_type: ClaimType
    entities: tuple[ExtractedEntity, ...] = field(default_factory=tuple)
    extraction_confidence: float = 1.0

    @property
    def id(self) -> int:
        return self.claim.id

    @property
    def text(self) -> str:
        return self.claim.text


class ClaimExtractor:
    """
    Extracts and classifies factual claims from LLM-generated text.

    The spaCy pipeline is loaded once per process (lazy singleton) since
    model loading is a relatively expensive I/O + deserialization operation.
    """

    _SPACY_MODEL_NAME: Final[str] = "en_core_web_sm"
    _MIN_CLAIM_TOKEN_LENGTH: Final[int] = 3

    # Sentences dominated by these leading cues are treated as opinion /
    # hedged statements rather than checkable factual claims.
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

    # Sentences matching these patterns carry no independently verifiable
    # factual content (greetings, instructions, meta-commentary).
    _NON_CLAIM_PATTERNS: Final[tuple[re.Pattern, ...]] = (
        re.compile(r"^(please|note that|remember|let me|i'll|i will)\b", re.IGNORECASE),
        re.compile(r"^(hi|hello|hey|thanks|thank you)\b", re.IGNORECASE),
    )

    _nlp: Language | None = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        self._ensure_pipeline_loaded()

    @classmethod
    def _ensure_pipeline_loaded(cls) -> None:
        """
        Lazily load and cache the spaCy pipeline shared across instances.

        Thread-safe: guarded by a class-level lock so concurrent request
        handlers do not race to load the model multiple times.
        """
        if cls._nlp is not None:
            return

        with cls._lock:
            if cls._nlp is not None:
                return

            try:
                logger.info(
                    "Loading spaCy pipeline '%s' for claim extraction.",
                    cls._SPACY_MODEL_NAME,
                )
                cls._nlp = spacy.load(cls._SPACY_MODEL_NAME)
            except OSError as exc:
                logger.error(
                    "spaCy model '%s' is not installed. Install it via "
                    "'python -m spacy download %s'.",
                    cls._SPACY_MODEL_NAME,
                    cls._SPACY_MODEL_NAME,
                )
                raise ClaimExtractionError(
                    f"Required spaCy model '{cls._SPACY_MODEL_NAME}' is not "
                    "available. See logs for installation instructions."
                ) from exc

    def extract(self, text: str) -> list[ExtractedClaim]:
        """
        Extract and classify factual claims from ``text``.

        Args:
            text: Raw LLM response text to analyze.

        Returns:
            A list of :class:`ExtractedClaim`, in document order, excluding
            sentences deemed non-factual (questions, greetings, pure
            instructions).

        Raises:
            ClaimExtractionError: If ``text`` is empty/whitespace-only, or
                if the underlying NLP pipeline fails unexpectedly.
        """
        if text is None or not text.strip():
            logger.warning("ClaimExtractor.extract called with empty input.")
            raise ClaimExtractionError("Cannot extract claims from empty text.")

        cleaned_text = TextCleaner.clean(text)

        try:
            doc = self._nlp(cleaned_text)  # type: ignore[misc]
        except Exception as exc:  # noqa: BLE001 - re-raised as domain error
            logger.exception("spaCy pipeline failed while processing input.")
            raise ClaimExtractionError("Failed to process text with NLP pipeline.") from exc

        claims: list[ExtractedClaim] = []
        claim_id = 1

        for sentence in doc.sents:
            normalized = sentence.text.strip()

            if not self._is_candidate_claim(sentence, normalized):
                continue

            entities = self._extract_entities(sentence)
            claim_type = self._classify_claim(sentence, normalized, entities)
            confidence = self._estimate_extraction_confidence(sentence, entities)

            claims.append(
                ExtractedClaim(
                    claim=ClaimModel(id=claim_id, text=normalized),
                    claim_type=claim_type,
                    entities=tuple(entities),
                    extraction_confidence=confidence,
                )
            )
            claim_id += 1

        logger.info(
            "Extracted %d claim(s) from %d sentence(s).",
            len(claims),
            sum(1 for _ in doc.sents),
        )
        return claims

    def _is_candidate_claim(self, sentence: Span, normalized_text: str) -> bool:
        """Determine whether a sentence contains independently checkable content."""
        if len(sentence) < self._MIN_CLAIM_TOKEN_LENGTH:
            return False

        if normalized_text.endswith("?"):
            return False

        lowered = normalized_text.lower()

        if lowered.startswith(self._OPINION_CUES):
            return False

        if any(pattern.match(normalized_text) for pattern in self._NON_CLAIM_PATTERNS):
            return False

        has_verb = any(token.pos_ in ("VERB", "AUX") for token in sentence)
        if not has_verb:
            return False

        return True

    @staticmethod
    def _extract_entities(sentence: Span) -> list[ExtractedEntity]:
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

    @staticmethod
    def _classify_claim(
        sentence: Span,
        normalized_text: str,
        entities: list[ExtractedEntity],
    ) -> ClaimType:
        """Assign a coarse :class:`ClaimType` to a candidate claim."""
        lowered = normalized_text.lower()

        if lowered.startswith(ClaimExtractor._OPINION_CUES) or " i think " in f" {lowered} ":
            return ClaimType.OPINION

        entity_labels = {entity.label for entity in entities}

        if entity_labels & {"DATE", "TIME"}:
            return ClaimType.TEMPORAL

        if entity_labels & {"CARDINAL", "QUANTITY", "PERCENT", "MONEY", "ORDINAL"}:
            return ClaimType.NUMERIC

        if entity_labels & {"PERSON", "ORG", "GPE", "PRODUCT", "EVENT", "NORP", "FAC"}:
            return ClaimType.ENTITY_CENTRIC

        if entities:
            return ClaimType.FACTUAL

        # No entities and no numeric/temporal grounding: hard to independently
        # verify against external evidence sources.
        return ClaimType.UNVERIFIABLE

    @staticmethod
    def _estimate_extraction_confidence(
        sentence: Span,
        entities: list[ExtractedEntity],
    ) -> float:
        """
        Heuristic confidence that this sentence is a well-formed, checkable claim.

        Longer sentences with grounded entities score higher; very short or
        entity-free sentences score lower, signaling downstream services to
        weight them less heavily.
        """
        token_count = len(sentence)
        length_score = min(token_count / 12.0, 1.0)
        entity_score = min(len(entities) / 3.0, 1.0)

        confidence = round((0.6 * length_score) + (0.4 * entity_score), 4)
        return max(confidence, 0.1)