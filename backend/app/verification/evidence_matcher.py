"""
Claim-to-evidence relevance matching.

Provides a lightweight, reusable way to rank a bundle of retrieved
evidence by how relevant each piece is to a given claim, independent of
:class:`app.verification.fact_verifier.FactVerifier`'s support/
contradiction judgment. This is useful as a preprocessing step -- e.g.
selecting only the most relevant evidence to pass on for full
verification, or surfacing the single best-matching passage for
citation purposes -- without needing to run the full verification
pipeline.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

from app.core.config import Settings, get_settings
from app.core.exceptions import InvalidClaimError
from app.embeddings.embedding_service import EmbeddingService
from app.models.evidence import Evidence, EvidenceBundle
from app.models.verification import Claim

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvidenceMatch:
    """
    A single piece of evidence paired with its relevance score to a claim.

    Attributes:
        evidence: The matched evidence.
        relevance_score: Cosine similarity between the claim and this
            evidence's text, in ``[0, 1]``.
    """

    evidence: Evidence
    relevance_score: float


class EvidenceMatcher:
    """Ranks retrieved evidence by semantic relevance to a claim."""

    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        """
        Args:
            embedding_service: Service used to embed claim and evidence
                text for similarity comparison. Defaults to a new
                :class:`EmbeddingService`.
            settings: Application settings. Defaults to the cached
                process-wide settings instance.
        """
        self._settings = settings or get_settings()
        self._embedding_service = embedding_service or EmbeddingService(self._settings)

    def match(self, claim: Claim, evidence_bundle: EvidenceBundle) -> list[EvidenceMatch]:
        """
        Rank a bundle of evidence by relevance to a claim.

        Args:
            claim: The claim to match evidence against.
            evidence_bundle: Evidence to rank.

        Returns:
            A list of :class:`EvidenceMatch` instances, ordered by
            descending relevance score. Returns an empty list if the
            evidence bundle is empty.

        Raises:
            InvalidClaimError: If the claim's text is empty or whitespace-only.
        """
        if not claim.text or not claim.text.strip():
            raise InvalidClaimError("claim text must not be empty or whitespace-only.")

        if evidence_bundle.is_empty:
            return []

        evidence_texts = [evidence.text for evidence in evidence_bundle.results]
        claim_vector, *evidence_vectors = self._embedding_service.embed_texts(
            [claim.text, *evidence_texts]
        )

        matches = [
            EvidenceMatch(
                evidence=evidence,
                relevance_score=self._cosine_similarity(claim_vector, evidence_vector),
            )
            for evidence, evidence_vector in zip(evidence_bundle.results, evidence_vectors)
        ]
        matches.sort(key=lambda match: match.relevance_score, reverse=True)

        logger.debug(
            "Matched %d evidence result(s) for claim '%s'.", len(matches), claim.claim_id
        )
        return matches

    def best_match(self, claim: Claim, evidence_bundle: EvidenceBundle) -> Optional[EvidenceMatch]:
        """
        Return the single most relevant piece of evidence for a claim.

        Args:
            claim: The claim to match evidence against.
            evidence_bundle: Evidence to search.

        Returns:
            The highest-scoring :class:`EvidenceMatch`, or ``None`` if
            the evidence bundle is empty.

        Raises:
            InvalidClaimError: If the claim's text is empty or whitespace-only.
        """
        matches = self.match(claim, evidence_bundle)
        return matches[0] if matches else None

    @staticmethod
    def _cosine_similarity(vector_a: list[float], vector_b: list[float]) -> float:
        """
        Compute cosine similarity between two equal-length vectors,
        clamped to ``[0, 1]``.

        Args:
            vector_a: First embedding vector.
            vector_b: Second embedding vector.

        Returns:
            Cosine similarity in ``[0, 1]``. Returns 0.0 if either
            vector has zero magnitude.
        """
        dot_product = sum(a * b for a, b in zip(vector_a, vector_b))
        magnitude_a = math.sqrt(sum(a * a for a in vector_a))
        magnitude_b = math.sqrt(sum(b * b for b in vector_b))

        if magnitude_a == 0.0 or magnitude_b == 0.0:
            return 0.0

        similarity = dot_product / (magnitude_a * magnitude_b)
        return max(0.0, min(1.0, similarity))