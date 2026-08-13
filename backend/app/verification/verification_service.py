"""
Top-level verification service.

Wraps :class:`app.verification.claim_verifier.ClaimVerifier` with
response-level aggregation, turning a set of per-claim verification
results into a single summary (counts by status, average confidence).
This is intended as the primary integration point for an external
consumer -- such as a hallucination detection engine -- that has
already extracted candidate claims from an LLM response and wants a
single, holistic verification report rather than having to aggregate
individual results itself.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from app.core.config import Settings, get_settings
from app.models.verification import VerificationResult, VerificationStatus
from app.verification.claim_verifier import ClaimVerifier

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VerificationSummary:
    """
    An aggregated view over the verification results for a set of claims.

    Attributes:
        total: Total number of claims verified.
        supported: Number of claims with status ``SUPPORTED``.
        contradicted: Number of claims with status ``CONTRADICTED``.
        insufficient_evidence: Number of claims with status ``INSUFFICIENT_EVIDENCE``.
        unverified: Number of claims with status ``UNVERIFIED``.
        average_confidence: Mean confidence score across all results.
        results: The individual per-claim verification results, in
            submission order.
    """

    total: int
    supported: int
    contradicted: int
    insufficient_evidence: int
    unverified: int
    average_confidence: float
    results: list[VerificationResult] = field(default_factory=list)

    @property
    def has_contradictions(self) -> bool:
        """True if at least one claim was contradicted by the evidence."""
        return self.contradicted > 0

    @property
    def support_rate(self) -> float:
        """The fraction of claims that were supported, in ``[0, 1]``. Returns 0.0 if there are no claims."""
        return self.supported / self.total if self.total else 0.0


class VerificationService:
    """
    Verifies a batch of claims and produces a response-level summary
    report.
    """

    def __init__(
        self,
        claim_verifier: Optional[ClaimVerifier] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        """
        Args:
            claim_verifier: Component used to verify each individual
                claim. Defaults to a new :class:`ClaimVerifier`.
            settings: Application settings. Defaults to the cached
                process-wide settings instance.
        """
        self._settings = settings or get_settings()
        self._claim_verifier = claim_verifier or ClaimVerifier(settings=self._settings)

    def verify_response(
        self,
        claim_texts: list[str],
        source_response_id: Optional[str] = None,
        top_k: Optional[int] = None,
        document_id: Optional[str] = None,
    ) -> VerificationSummary:
        """
        Verify all claims extracted from a single response and
        summarize the results.

        Args:
            claim_texts: The claim texts to verify.
            source_response_id: Identifier of the source LLM response
                these claims were extracted from, if known.
            top_k: Maximum number of evidence chunks to retrieve per
                claim. Defaults to the configured application retrieval
                top-k.
            document_id: Optional document filter for retrieval.

        Returns:
            A :class:`VerificationSummary` aggregating the outcome
            across all submitted claims. If ``claim_texts`` is empty,
            all counts are zero and ``results`` is an empty list.

        Raises:
            InvalidClaimError: If any claim text is empty or whitespace-only.
            CollectionNotFoundError: If no documents have been indexed yet.
            EmbeddingGenerationError: If a claim cannot be embedded.
        """
        if not claim_texts:
            logger.debug("No claims provided to verification service; returning empty summary.")
            return VerificationSummary(
                total=0,
                supported=0,
                contradicted=0,
                insufficient_evidence=0,
                unverified=0,
                average_confidence=0.0,
                results=[],
            )

        results = self._claim_verifier.verify_claims_batch(
            claim_texts,
            source_response_id=source_response_id,
            top_k=top_k,
            document_id=document_id,
        )
        summary = self._summarize(results)

        logger.info(
            "Verification summary: total=%d supported=%d contradicted=%d "
            "insufficient_evidence=%d unverified=%d avg_confidence=%.3f.",
            summary.total,
            summary.supported,
            summary.contradicted,
            summary.insufficient_evidence,
            summary.unverified,
            summary.average_confidence,
        )
        return summary

    @staticmethod
    def _summarize(results: list[VerificationResult]) -> VerificationSummary:
        """
        Aggregate a list of per-claim verification results into a
        summary.

        Args:
            results: Per-claim verification results.

        Returns:
            A :class:`VerificationSummary` describing the aggregate outcome.
        """
        status_counts = {status: 0 for status in VerificationStatus}
        for result in results:
            status_counts[result.status] += 1

        average_confidence = (
            sum(result.confidence for result in results) / len(results) if results else 0.0
        )

        return VerificationSummary(
            total=len(results),
            supported=status_counts[VerificationStatus.SUPPORTED],
            contradicted=status_counts[VerificationStatus.CONTRADICTED],
            insufficient_evidence=status_counts[VerificationStatus.INSUFFICIENT_EVIDENCE],
            unverified=status_counts[VerificationStatus.UNVERIFIED],
            average_confidence=round(average_confidence, 6),
            results=results,
        )