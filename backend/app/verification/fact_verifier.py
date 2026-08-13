"""
Fact verification service.

Compares a factual claim against a bundle of retrieved evidence and
produces a VerificationResult describing whether the evidence supports,
contradicts, or is insufficient to judge the claim.

The verifier uses three lightweight signals:

1. Semantic support score
   - Cosine similarity between claim and evidence embeddings.

2. Negation-aware contradiction score
   - Detects polarity conflicts such as:
       "India is not in Asia."
       "India is in Asia."

3. Explicit factual-value conflict detection
   - Detects conflicts such as:
       "The capital of India is Mumbai."
       "New Delhi is the capital of India."

The third signal is important because semantic similarity alone cannot
distinguish a correct factual statement from a highly similar but
factually conflicting statement.
"""

from __future__ import annotations

import logging
import math
import re
from typing import Optional

from app.core.config import Settings, get_settings
from app.core.exceptions import InvalidClaimError
from app.embeddings.embedding_service import EmbeddingService
from app.models.evidence import Evidence, EvidenceBundle
from app.models.verification import (
    Claim,
    EvidenceAssessment,
    VerificationResult,
    VerificationStatus,
)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Negation detection
# ---------------------------------------------------------------------------

_NEGATION_PATTERN = re.compile(
    r"\b("
    r"not|no|never|none|cannot|can't|won't|"
    r"doesn't|didn't|isn't|aren't|wasn't|weren't|"
    r"hasn't|haven't|hadn't|without"
    r")\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Explicit factual statement patterns
# ---------------------------------------------------------------------------

# Example:
#
#   "The capital of India is Mumbai."
#
# Produces:
#
#   predicate = capital
#   context   = india
#   value     = mumbai
#
# This is deliberately conservative. We only use this rule when the
# sentence clearly has the form:
#
#   [predicate] of [context] is [value]
#
_FACT_FORWARD_PATTERN = re.compile(
    r"""
    \b
    (?:the\s+)?
    (?P<predicate>[a-z][a-z\s-]{1,40}?)
    \s+of\s+
    (?P<context>[a-z][a-z0-9\s-]{1,60}?)
    \s+is\s+
    (?P<value>[a-z][a-z0-9\s-]{1,80}?)
    (?=[.!?,;:]|$)
    """,
    re.IGNORECASE | re.VERBOSE,
)


# Example:
#
#   "New Delhi is the capital of India."
#
# Produces the same canonical representation as:
#
#   "The capital of India is New Delhi."
#
_FACT_REVERSE_PATTERN = re.compile(
    r"""
    \b
    (?P<value>[a-z][a-z0-9\s-]{1,80}?)
    \s+is\s+
    (?:the\s+)?
    (?P<predicate>[a-z][a-z\s-]{1,40}?)
    \s+of\s+
    (?P<context>[a-z][a-z0-9\s-]{1,60}?)
    (?=[.!?,;:]|$)
    """,
    re.IGNORECASE | re.VERBOSE,
)


# Simple "X is Y" pattern.
#
# Used as a fallback for statements such as:
#
#   "Paris is the capital of France."
#
_SIMPLE_IS_PATTERN = re.compile(
    r"""
    \b
    (?P<subject>[a-z][a-z0-9\s-]{1,80}?)
    \s+is\s+
    (?P<object>[a-z][a-z0-9\s-]{1,100}?)
    (?=[.!?,;:]|$)
    """,
    re.IGNORECASE | re.VERBOSE,
)


# A conservative threshold for explicit factual conflict detection.
#
# We do NOT want a merely topical sentence to become a contradiction.
# The semantic similarity must therefore already be reasonably strong.
_FACT_CONFLICT_MIN_SIMILARITY = 0.55


class FactVerifier:
    """
    Verifies factual claims against retrieved evidence using semantic
    similarity, negation-aware contradiction detection, and explicit
    factual-value conflict detection.
    """

    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        """
        Args:
            embedding_service:
                Service used to embed claim and evidence text.

            settings:
                Application settings providing verification thresholds.
        """
        self._settings = settings or get_settings()

        self._embedding_service = (
            embedding_service
            or EmbeddingService(self._settings)
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def verify(
        self,
        claim: Claim,
        evidence_bundle: EvidenceBundle,
    ) -> VerificationResult:
        """
        Verify a claim against retrieved evidence.

        Returns:
            VerificationResult containing:
                - status
                - confidence
                - explanation
                - per-evidence assessments
        """

        if not claim.text or not claim.text.strip():
            raise InvalidClaimError(
                "claim text must not be empty or whitespace-only."
            )

        if evidence_bundle.is_empty:
            logger.info(
                "No evidence retrieved for claim '%s'; marking UNVERIFIED.",
                claim.claim_id,
            )

            return VerificationResult.unverified(claim)

        assessments = [
            self._assess_evidence(claim, evidence)
            for evidence in evidence_bundle.results
        ]

        status, confidence, explanation = self._decide_status(
            assessments
        )

        result = VerificationResult(
            claim=claim,
            status=status,
            confidence=confidence,
            assessments=assessments,
            explanation=explanation,
        )

        logger.info(
            "Verified claim '%s': status=%s confidence=%.3f "
            "(%d evidence assessed).",
            claim.claim_id,
            status.value,
            confidence,
            len(assessments),
        )

        return result

    # ------------------------------------------------------------------
    # Evidence assessment
    # ------------------------------------------------------------------

    def _assess_evidence(
        self,
        claim: Claim,
        evidence: Evidence,
    ) -> EvidenceAssessment:
        """
        Compute support and contradiction scores for one evidence chunk.

        Contradiction can now be detected through:

        1. Negation/polarity mismatch.
        2. Explicit factual-value conflict.

        Example:

            Claim:
                The capital of India is Mumbai.

            Evidence:
                New Delhi is the capital of India.

        The semantic similarity is high, but the extracted factual value
        differs. Therefore the evidence is treated as contradictory.
        """

        semantic_similarity = self._semantic_similarity(
            claim.text,
            evidence.text,
        )

        polarity_mismatch = self._has_polarity_mismatch(
            claim.text,
            evidence.text,
        )

        factual_conflict = self._has_factual_conflict(
            claim.text,
            evidence.text,
            semantic_similarity,
        )

        # --------------------------------------------------------------
        # Contradiction has priority over support.
        # --------------------------------------------------------------

        if polarity_mismatch or factual_conflict:
            support_score = semantic_similarity * 0.10
            contradiction_score = semantic_similarity

            logger.debug(
                "Contradiction detected. claim=%r evidence=%r "
                "polarity_mismatch=%s factual_conflict=%s "
                "similarity=%.4f",
                claim.text,
                evidence.text,
                polarity_mismatch,
                factual_conflict,
                semantic_similarity,
            )

        else:
            support_score = semantic_similarity
            contradiction_score = 0.0

        return EvidenceAssessment(
            evidence=evidence,
            support_score=round(support_score, 6),
            contradiction_score=round(contradiction_score, 6),
        )

    # ------------------------------------------------------------------
    # Semantic similarity
    # ------------------------------------------------------------------

    def _semantic_similarity(
        self,
        claim_text: str,
        evidence_text: str,
    ) -> float:
        """
        Compute cosine similarity between claim and evidence embeddings.

        Returns:
            Similarity score in [0, 1].
        """

        claim_vector, evidence_vector = (
            self._embedding_service.embed_texts(
                [claim_text, evidence_text]
            )
        )

        return self._cosine_similarity(
            claim_vector,
            evidence_vector,
        )

    @staticmethod
    def _cosine_similarity(
        vector_a: list[float],
        vector_b: list[float],
    ) -> float:
        """
        Compute cosine similarity between two vectors.

        The result is clamped to [0, 1].
        """

        dot_product = sum(
            a * b
            for a, b in zip(vector_a, vector_b)
        )

        magnitude_a = math.sqrt(
            sum(a * a for a in vector_a)
        )

        magnitude_b = math.sqrt(
            sum(b * b for b in vector_b)
        )

        if magnitude_a == 0.0 or magnitude_b == 0.0:
            return 0.0

        similarity = dot_product / (
            magnitude_a * magnitude_b
        )

        return max(
            0.0,
            min(1.0, similarity),
        )

    # ------------------------------------------------------------------
    # Negation detection
    # ------------------------------------------------------------------

    def _has_polarity_mismatch(
        self,
        claim_text: str,
        evidence_text: str,
    ) -> bool:
        """
        Detect simple negation-polarity mismatch.

        Example:

            Claim:
                The Earth is not flat.

            Evidence:
                The Earth is flat.

        Returns True when exactly one sentence contains a negation marker.
        """

        claim_has_negation = bool(
            _NEGATION_PATTERN.search(claim_text)
        )

        evidence_has_negation = bool(
            _NEGATION_PATTERN.search(evidence_text)
        )

        return claim_has_negation != evidence_has_negation

    # ------------------------------------------------------------------
    # Factual conflict detection
    # ------------------------------------------------------------------

    def _has_factual_conflict(
        self,
        claim_text: str,
        evidence_text: str,
        semantic_similarity: float,
    ) -> bool:
        """
        Detect explicit factual-value conflicts.

        This fixes the important case where two statements are highly
        semantically similar but contain different factual values.

        Example:

            Claim:
                The capital of India is Mumbai.

            Evidence:
                New Delhi is the capital of India.

        Both statements are about the capital of India and therefore have
        high semantic similarity, but their factual values are different.

        We only activate this detector when semantic similarity is above
        _FACT_CONFLICT_MIN_SIMILARITY to avoid treating unrelated topics
        as contradictions.
        """

        if semantic_similarity < _FACT_CONFLICT_MIN_SIMILARITY:
            return False

        claim_facts = self._extract_factual_slots(
            claim_text
        )

        evidence_facts = self._extract_factual_slots(
            evidence_text
        )

        if not claim_facts or not evidence_facts:
            return False

        for claim_key, claim_value in claim_facts:
            for evidence_key, evidence_value in evidence_facts:

                # Same factual relation/context.
                if claim_key != evidence_key:
                    continue

                # Same value means the evidence supports the claim.
                if self._normalize_phrase(
                    claim_value
                ) == self._normalize_phrase(
                    evidence_value
                ):
                    continue

                # Different values for the same fact are contradictory.
                logger.debug(
                    "Explicit factual conflict detected: "
                    "key=%r claim_value=%r evidence_value=%r",
                    claim_key,
                    claim_value,
                    evidence_value,
                )

                return True

        return False

    # ------------------------------------------------------------------
    # Fact extraction
    # ------------------------------------------------------------------

    def _extract_factual_slots(
        self,
        text: str,
    ) -> list[tuple[tuple[str, str], str]]:
        """
        Extract conservative factual slots.

        The canonical representation is:

            ((predicate, context), value)

        Example:

            "The capital of India is Mumbai."

        becomes approximately:

            (("capital", "india"), "mumbai")

        And:

            "New Delhi is the capital of India."

        becomes:

            (("capital", "india"), "new delhi")

        This makes the two statements directly comparable.
        """

        normalized_text = self._clean_sentence(text)

        facts: list[tuple[tuple[str, str], str]] = []

        # --------------------------------------------------------------
        # Pattern 1:
        #
        #   The capital of India is Mumbai.
        # --------------------------------------------------------------

        for match in _FACT_FORWARD_PATTERN.finditer(
            normalized_text
        ):
            predicate = self._normalize_predicate(
                match.group("predicate")
            )

            context = self._normalize_phrase(
                match.group("context")
            )

            value = self._normalize_phrase(
                match.group("value")
            )

            if (
                predicate
                and context
                and value
            ):
                facts.append(
                    (
                        (predicate, context),
                        value,
                    )
                )

        # --------------------------------------------------------------
        # Pattern 2:
        #
        #   New Delhi is the capital of India.
        # --------------------------------------------------------------

        for match in _FACT_REVERSE_PATTERN.finditer(
            normalized_text
        ):
            predicate = self._normalize_predicate(
                match.group("predicate")
            )

            context = self._normalize_phrase(
                match.group("context")
            )

            value = self._normalize_phrase(
                match.group("value")
            )

            if (
                predicate
                and context
                and value
            ):
                facts.append(
                    (
                        (predicate, context),
                        value,
                    )
                )

        # Remove duplicates while preserving order.
        unique_facts: list[
            tuple[tuple[str, str], str]
        ] = []

        seen: set[
            tuple[tuple[str, str], str]
        ] = set()

        for fact in facts:
            if fact not in seen:
                seen.add(fact)
                unique_facts.append(fact)

        return unique_facts

    # ------------------------------------------------------------------
    # Text normalization
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_sentence(text: str) -> str:
        """
        Normalize text enough for conservative fact extraction.
        """

        text = text.strip()

        # Replace repeated whitespace.
        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        # Remove quotation marks around sentences.
        text = text.strip(
            "\"'` "
        )

        return text

    @staticmethod
    def _normalize_phrase(
        text: str,
    ) -> str:
        """
        Normalize a phrase for comparison.
        """

        value = text.lower().strip()

        # Remove punctuation.
        value = re.sub(
            r"[^a-z0-9\s-]",
            " ",
            value,
        )

        # Remove common leading articles.
        value = re.sub(
            r"^(the|a|an)\s+",
            "",
            value,
        )

        # Collapse whitespace.
        value = re.sub(
            r"\s+",
            " ",
            value,
        ).strip()

        return value

    @classmethod
    def _normalize_predicate(
        cls,
        text: str,
    ) -> str:
        """
        Normalize a factual predicate.

        Treats common equivalent expressions such as:

            capital
            capital city

        as the same predicate.
        """

        predicate = cls._normalize_phrase(text)

        # "capital city" and "capital" represent the same relation.
        predicate = re.sub(
            r"\bcapital\s+city\b",
            "capital",
            predicate,
        )

        # Normalize a few common equivalent forms.
        replacements = {
            "located in": "location",
            "situated in": "location",
            "headquartered in": "headquarters",
        }

        predicate = replacements.get(
            predicate,
            predicate,
        )

        return predicate

    # ------------------------------------------------------------------
    # Status decision
    # ------------------------------------------------------------------

    def _decide_status(
        self,
        assessments: list[EvidenceAssessment],
    ) -> tuple[
        VerificationStatus,
        float,
        str,
    ]:
        """
        Aggregate evidence assessments into one verification result.

        Decision priority:

        1. Strong contradiction -> CONTRADICTED
        2. Strong support -> SUPPORTED
        3. Otherwise -> INSUFFICIENT_EVIDENCE

        Contradiction deliberately has priority because a conflicting
        high-quality passage must not be hidden by a high similarity score.
        """

        max_support = max(
            (
                assessment.support_score
                for assessment in assessments
            ),
            default=0.0,
        )

        max_contradiction = max(
            (
                assessment.contradiction_score
                for assessment in assessments
            ),
            default=0.0,
        )

        contradiction_threshold = (
            self._settings
            .verification_contradiction_threshold
        )

        support_threshold = (
            self._settings
            .verification_support_threshold
        )

        # --------------------------------------------------------------
        # Contradiction first
        # --------------------------------------------------------------

        if (
            max_contradiction
            >= contradiction_threshold
        ):
            return (
                VerificationStatus.CONTRADICTED,
                max_contradiction,
                (
                    "Retrieved evidence contradicts the claim. "
                    "The evidence is semantically related but contains "
                    "a conflicting factual value or polarity."
                ),
            )

        # --------------------------------------------------------------
        # Support
        # --------------------------------------------------------------

        if (
            max_support
            >= support_threshold
        ):
            return (
                VerificationStatus.SUPPORTED,
                max_support,
                (
                    "Retrieved evidence semantically supports "
                    "the claim."
                ),
            )

        # --------------------------------------------------------------
        # Insufficient evidence
        # --------------------------------------------------------------

        best_effort_confidence = max(
            max_support,
            max_contradiction,
        )

        return (
            VerificationStatus.INSUFFICIENT_EVIDENCE,
            best_effort_confidence,
            (
                "Relevant evidence was retrieved, but it does not "
                "clearly support or contradict the claim above the "
                "configured confidence thresholds."
            ),
        )