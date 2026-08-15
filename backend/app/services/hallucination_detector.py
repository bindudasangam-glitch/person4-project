"""
Hallucination Detection Service
================================

Compares extracted claims against retrieved evidence and classifies
each claim as supported, contradicted, or insufficient evidence.

The spaCy pipeline is shared with ClaimExtractor through:

    app.utils.nlp_pipeline.get_spacy_pipeline

This prevents multiple copies of en_core_web_sm from being loaded
inside the same Render worker process.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from app.core.logging import logger
from app.models.claim_model import (
    ClaimModel,
    VerificationStatus,
)
from app.utils.nlp_pipeline import get_spacy_pipeline


__all__ = [
    "EvidencePassage",
    "EvidenceSource",
    "LexicalOverlapEvidenceSource",
    "HallucinationDetectionError",
    "ClaimDetectionOutcome",
    "HallucinationDetector",
]


# ============================================================================
# EXCEPTIONS
# ============================================================================


class HallucinationDetectionError(Exception):
    """Raised when hallucination detection cannot be completed."""


# ============================================================================
# TOKENIZATION
# ============================================================================


_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "of",
        "in",
        "on",
        "at",
        "to",
        "for",
        "and",
        "or",
        "but",
        "with",
        "by",
        "as",
        "that",
        "this",
        "it",
        "its",
        "from",
        "has",
        "have",
        "had",
        "not",
        "no",
    }
)


_NEGATION_CUES: frozenset[str] = frozenset(
    {
        "not",
        "never",
        "no",
        "n't",
        "cannot",
        "isn't",
        "wasn't",
        "weren't",
        "didn't",
    }
)


_WORD_PATTERN = re.compile(
    r"[A-Za-z0-9]+"
)


# ============================================================================
# EVIDENCE
# ============================================================================


@dataclass(frozen=True, slots=True)
class EvidencePassage:
    """A single retrieved piece of evidence."""

    text: str
    source: str
    relevance_score: float = 0.0


@runtime_checkable
class EvidenceSource(Protocol):
    """Abstraction over any system capable of retrieving evidence."""

    def retrieve(
        self,
        claim_text: str,
        top_k: int = 3,
    ) -> list[EvidencePassage]:
        """Return relevant evidence passages."""
        ...


# ============================================================================
# LEXICAL EVIDENCE SOURCE
# ============================================================================


class LexicalOverlapEvidenceSource:
    """
    Dependency-free evidence source using lexical Jaccard similarity.

    This remains available as a lightweight fallback/test implementation.
    """

    def __init__(
        self,
        corpus: list[str] | None = None,
    ) -> None:
        self._corpus: list[str] = (
            list(corpus)
            if corpus
            else []
        )

    def add_documents(
        self,
        documents: list[str],
    ) -> None:
        """Add non-empty documents to the in-memory corpus."""

        self._corpus.extend(
            document
            for document in documents
            if document and document.strip()
        )

    def retrieve(
        self,
        claim_text: str,
        top_k: int = 3,
    ) -> list[EvidencePassage]:
        """Retrieve the highest lexical-overlap passages."""

        if not self._corpus:
            return []

        claim_tokens = self._tokenize(
            claim_text
        )

        if not claim_tokens:
            return []

        scored: list[EvidencePassage] = []

        for index, passage in enumerate(
            self._corpus
        ):
            passage_tokens = self._tokenize(
                passage
            )

            score = self._jaccard(
                claim_tokens,
                passage_tokens,
            )

            if score > 0.0:
                scored.append(
                    EvidencePassage(
                        text=passage,
                        source=f"corpus_doc_{index}",
                        relevance_score=score,
                    )
                )

        scored.sort(
            key=lambda item: item.relevance_score,
            reverse=True,
        )

        return scored[:top_k]

    @staticmethod
    def _tokenize(
        text: str,
    ) -> frozenset[str]:
        words = (
            word.lower()
            for word in _WORD_PATTERN.findall(text)
        )

        return frozenset(
            word
            for word in words
            if word not in _STOPWORDS
        )

    @staticmethod
    def _jaccard(
        a: frozenset[str],
        b: frozenset[str],
    ) -> float:
        if not a or not b:
            return 0.0

        intersection = len(a & b)
        union = len(a | b)

        return (
            intersection / union
            if union
            else 0.0
        )


# ============================================================================
# DETECTION RESULT
# ============================================================================


@dataclass(frozen=True, slots=True)
class ClaimDetectionOutcome:
    """Explainable result for one claim."""

    claim_id: int
    verification_status: VerificationStatus
    support_score: float
    entity_agreement: float
    negation_mismatch: bool
    entity_disagreement: bool
    evidence: tuple[EvidencePassage, ...]


# ============================================================================
# HALLUCINATION DETECTOR
# ============================================================================


class HallucinationDetector:
    """
    Detect unsupported, fabricated, or contradicted claims.

    The detector uses the shared spaCy pipeline rather than loading
    another copy of en_core_web_sm.
    """

    def __init__(
        self,
        evidence_source: EvidenceSource | None = None,
        support_threshold: float = 0.35,
        contradiction_threshold: float = 0.2,
        top_k_evidence: int = 3,
    ) -> None:

        if not 0.0 <= support_threshold <= 1.0:
            raise HallucinationDetectionError(
                "support_threshold must be in [0, 1], "
                f"got {support_threshold}."
            )

        if not 0.0 <= contradiction_threshold <= 1.0:
            raise HallucinationDetectionError(
                "contradiction_threshold must be in [0, 1], "
                f"got {contradiction_threshold}."
            )

        if top_k_evidence < 1:
            raise HallucinationDetectionError(
                "top_k_evidence must be at least 1."
            )

        self._evidence_source = (
            evidence_source
            or LexicalOverlapEvidenceSource()
        )

        self._support_threshold = (
            support_threshold
        )

        self._contradiction_threshold = (
            contradiction_threshold
        )

        self._top_k_evidence = (
            top_k_evidence
        )

        # IMPORTANT:
        # Do not load spaCy separately here.
        # ClaimExtractor and HallucinationDetector share one pipeline.
        self._nlp = get_spacy_pipeline()

    # =========================================================================
    # DETECTION
    # =========================================================================

    def detect(
        self,
        claims: list[ClaimModel],
    ) -> list[ClaimDetectionOutcome]:
        """
        Run hallucination detection over a batch of claims.
        """

        if not claims:
            raise HallucinationDetectionError(
                "Cannot run detection on an empty claim list."
            )

        outcomes: list[
            ClaimDetectionOutcome
        ] = []

        for claim in claims:

            try:
                outcome = self._detect_single(
                    claim
                )

            except HallucinationDetectionError:
                raise

            except Exception as exc:

                logger.exception(
                    "Unexpected failure detecting claim %d.",
                    claim.id,
                )

                raise HallucinationDetectionError(
                    f"Detection failed for claim {claim.id}."
                ) from exc

            claim.mark_verified(
                outcome.verification_status,
                evidence=tuple(
                    passage.text
                    for passage in outcome.evidence
                ),
            )

            outcomes.append(
                outcome
            )

        supported = sum(
            1
            for outcome in outcomes
            if outcome.verification_status
            is VerificationStatus.SUPPORTED
        )

        contradicted = sum(
            1
            for outcome in outcomes
            if outcome.verification_status
            is VerificationStatus.CONTRADICTED
        )

        insufficient = (
            len(outcomes)
            - supported
            - contradicted
        )

        logger.info(
            "Hallucination detection complete: "
            "%d claim(s) — %d supported, "
            "%d contradicted, %d insufficient evidence.",
            len(outcomes),
            supported,
            contradicted,
            insufficient,
        )

        return outcomes

    # =========================================================================
    # SINGLE CLAIM
    # =========================================================================

    def _detect_single(
        self,
        claim: ClaimModel,
    ) -> ClaimDetectionOutcome:
        """Evaluate one claim against retrieved evidence."""

        passages = self._evidence_source.retrieve(
            claim.text,
            top_k=self._top_k_evidence,
        )

        if not passages:
            return ClaimDetectionOutcome(
                claim_id=claim.id,
                verification_status=(
                    VerificationStatus.INSUFFICIENT_EVIDENCE
                ),
                support_score=0.0,
                entity_agreement=0.0,
                negation_mismatch=False,
                entity_disagreement=False,
                evidence=(),
            )

        best_passage = passages[0]

        support_score = (
            best_passage.relevance_score
        )

        entity_agreement = (
            self._entity_agreement(
                claim,
                best_passage.text,
            )
        )

        negation_mismatch = (
            self._has_negation_mismatch(
                claim.text,
                best_passage.text,
            )
        )

        entity_disagreement = (
            self._has_entity_disagreement(
                claim,
                best_passage.text,
            )
        )

        combined_score = round(
            (0.6 * support_score)
            + (0.4 * entity_agreement),
            4,
        )

        if (
            (
                negation_mismatch
                or entity_disagreement
            )
            and support_score
            >= self._contradiction_threshold
        ):
            verification_status = (
                VerificationStatus.CONTRADICTED
            )

        elif (
            combined_score
            >= self._support_threshold
        ):
            verification_status = (
                VerificationStatus.SUPPORTED
            )

        else:
            verification_status = (
                VerificationStatus.INSUFFICIENT_EVIDENCE
            )

        return ClaimDetectionOutcome(
            claim_id=claim.id,
            verification_status=verification_status,
            support_score=combined_score,
            entity_agreement=entity_agreement,
            negation_mismatch=negation_mismatch,
            entity_disagreement=entity_disagreement,
            evidence=tuple(passages),
        )

    # =========================================================================
    # ENTITY AGREEMENT
    # =========================================================================

    def _entity_agreement(
        self,
        claim: ClaimModel,
        evidence_text: str,
    ) -> float:
        """Calculate the fraction of claim entities found in evidence."""

        if not claim.entities:
            return 0.0

        evidence_lower = (
            evidence_text.lower()
        )

        matches = sum(
            1
            for entity in claim.entities
            if entity.text.lower()
            in evidence_lower
        )

        return matches / len(
            claim.entities
        )

    # =========================================================================
    # NEGATION
    # =========================================================================

    def _has_negation_mismatch(
        self,
        claim_text: str,
        evidence_text: str,
    ) -> bool:
        """
        Detect whether claim and evidence disagree on polarity.
        """

        claim_negated = (
            self._contains_negation(
                claim_text
            )
        )

        evidence_negated = (
            self._contains_negation(
                evidence_text
            )
        )

        return (
            claim_negated
            != evidence_negated
        )

    def _contains_negation(
        self,
        text: str,
    ) -> bool:
        """Return True when text contains an explicit negation."""

        doc = self._nlp(
            text
        )

        for token in doc:

            if (
                token.dep_ == "neg"
                or token.lower_
                in _NEGATION_CUES
            ):
                return True

        return False

    # =========================================================================
    # ENTITY DISAGREEMENT
    # =========================================================================

    def _has_entity_disagreement(
        self,
        claim: ClaimModel,
        evidence_text: str,
    ) -> bool:
        """
        Detect disagreement caused by a substituted named entity.

        Example:

            Claim:
                The capital of India is Mumbai.

            Evidence:
                The capital of India is New Delhi.

        The claim entity is absent while another entity of the same
        spaCy label is present.
        """

        if not claim.entities:
            return False

        evidence_doc = self._nlp(
            evidence_text
        )

        evidence_entities_by_label: dict[
            str,
            set[str],
        ] = {}

        for entity in evidence_doc.ents:

            evidence_entities_by_label.setdefault(
                entity.label_,
                set(),
            ).add(
                entity.text.strip().lower()
            )

        for entity in claim.entities:

            evidence_values = (
                evidence_entities_by_label.get(
                    entity.label
                )
            )

            if not evidence_values:
                continue

            if (
                entity.text.strip().lower()
                in evidence_values
            ):
                continue

            return True

        return False