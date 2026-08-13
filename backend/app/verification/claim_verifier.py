"""
Claim verification orchestration.

Composes :class:`app.retrieval.retriever.Retriever` and
:class:`app.verification.fact_verifier.FactVerifier` to verify a claim
end-to-end: retrieve relevant evidence for the claim, then judge
whether that evidence supports, contradicts, or is insufficient to
assess it. This factors out the "retrieve, then verify" sequence
currently duplicated inline across the single- and batch-claim
verification API routes into one reusable, independently testable
service.
"""

from __future__ import annotations

import logging
from typing import Optional

from app.core.config import Settings, get_settings
from app.models.verification import Claim, VerificationResult
from app.retrieval.retriever import Retriever
from app.verification.fact_verifier import FactVerifier

logger = logging.getLogger(__name__)


class ClaimVerifier:
    """Orchestrates evidence retrieval and fact verification for one or more claims."""

    def __init__(
        self,
        retriever: Optional[Retriever] = None,
        fact_verifier: Optional[FactVerifier] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        """
        Args:
            retriever: Component used to retrieve evidence for a claim.
                Defaults to a new :class:`Retriever`.
            fact_verifier: Component used to judge a claim against its
                retrieved evidence. Defaults to a new :class:`FactVerifier`.
            settings: Application settings. Defaults to the cached
                process-wide settings instance.
        """
        self._settings = settings or get_settings()
        self._retriever = retriever or Retriever(settings=self._settings)
        self._fact_verifier = fact_verifier or FactVerifier(settings=self._settings)

    def verify_claim_text(
        self,
        claim_text: str,
        source_response_id: Optional[str] = None,
        top_k: Optional[int] = None,
        document_id: Optional[str] = None,
    ) -> VerificationResult:
        """
        Verify a single claim, given as raw text, end-to-end.

        Args:
            claim_text: The claim text to verify.
            source_response_id: Identifier of the source LLM response
                this claim was extracted from, if known.
            top_k: Maximum number of evidence chunks to retrieve.
                Defaults to the configured application retrieval top-k.
            document_id: Optional document filter for retrieval.

        Returns:
            A :class:`VerificationResult` describing the outcome.

        Raises:
            InvalidClaimError: If ``claim_text`` is empty or whitespace-only.
            CollectionNotFoundError: If no documents have been indexed yet.
            EmbeddingGenerationError: If the claim cannot be embedded.
        """
        claim = Claim(text=claim_text, source_response_id=source_response_id)
        return self.verify_claim(claim, top_k=top_k, document_id=document_id)

    def verify_claim(
        self,
        claim: Claim,
        top_k: Optional[int] = None,
        document_id: Optional[str] = None,
    ) -> VerificationResult:
        """
        Verify an already-constructed claim end-to-end.

        Args:
            claim: The claim to verify.
            top_k: Maximum number of evidence chunks to retrieve.
                Defaults to the configured application retrieval top-k.
            document_id: Optional document filter for retrieval.

        Returns:
            A :class:`VerificationResult` describing the outcome.

        Raises:
            InvalidClaimError: If the claim's text is empty or whitespace-only.
            CollectionNotFoundError: If no documents have been indexed yet.
            EmbeddingGenerationError: If the claim cannot be embedded.
        """
        evidence_bundle = self._retriever.retrieve(
            query=claim.text, top_k=top_k, document_id=document_id
        )
        result = self._fact_verifier.verify(claim, evidence_bundle)

        logger.debug(
            "Claim verifier resolved claim '%s' to status '%s'.", claim.claim_id, result.status.value
        )
        return result

    def verify_claims_batch(
        self,
        claim_texts: list[str],
        source_response_id: Optional[str] = None,
        top_k: Optional[int] = None,
        document_id: Optional[str] = None,
    ) -> list[VerificationResult]:
        """
        Verify multiple claims, each independently retrieving its own
        evidence.

        Args:
            claim_texts: The claim texts to verify.
            source_response_id: Identifier of the source LLM response
                these claims were extracted from, if known. Applied to
                every claim in the batch.
            top_k: Maximum number of evidence chunks to retrieve per
                claim. Defaults to the configured application retrieval
                top-k.
            document_id: Optional document filter for retrieval, applied
                to every claim in the batch.

        Returns:
            A list of :class:`VerificationResult` instances, one per
            claim, in the same order as ``claim_texts``. Returns an
            empty list if ``claim_texts`` is empty.

        Raises:
            InvalidClaimError: If any claim text is empty or whitespace-only.
            CollectionNotFoundError: If no documents have been indexed yet.
            EmbeddingGenerationError: If a claim cannot be embedded.
        """
        results = [
            self.verify_claim_text(
                claim_text,
                source_response_id=source_response_id,
                top_k=top_k,
                document_id=document_id,
            )
            for claim_text in claim_texts
        ]

        logger.info("Verified %d claim(s) in batch.", len(results))
        return results