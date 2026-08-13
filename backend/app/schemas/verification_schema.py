"""
Verification API schemas.

Defines the request and response models for the `/verification` router
(`app/api/routes/verification.py`). Kept separate from the internal
`app.models.verification.VerificationResult` domain model so the API's
public shape can evolve independently, while still surfacing the
per-evidence support/contradiction reasoning behind each verdict rather
than a single opaque score.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from app.models.claim_model import VerificationStatus
from app.models.verification import VerificationResult

__all__ = [
    "BatchClaimVerificationRequest",
    "BatchClaimVerificationResponse",
    "ClaimVerificationRequest",
    "ClaimVerificationResponse",
    "EvidenceAssessmentSummary",
]


class ClaimVerificationRequest(BaseModel):
    """
    Request payload for verifying a single claim.

    Attributes:
        claim: The claim text to verify.
        source_response_id: Identifier of the source LLM response this
            claim was extracted from, if known.
        top_k: Maximum number of evidence chunks to retrieve for this
            claim. If omitted, the server-side configured default is used.
        document_id: If provided, restricts evidence retrieval to this
            specific document only.
    """

    claim: str = Field(..., min_length=1, max_length=5000)
    source_response_id: Optional[str] = None
    top_k: Optional[int] = Field(default=None, ge=1, le=100)
    document_id: Optional[str] = None


class EvidenceAssessmentSummary(BaseModel):
    """
    A flattened summary of a single evidence assessment, for API transparency.

    Attributes:
        chunk_id: Identifier of the assessed evidence chunk.
        text: The evidence chunk's text content.
        source_name: Human-readable filename/source of the evidence's document.
        support_score: Degree to which this evidence supports the claim, in ``[0, 1]``.
        contradiction_score: Degree to which this evidence contradicts the claim, in ``[0, 1]``.
    """

    chunk_id: str
    text: str
    source_name: str
    support_score: float = Field(..., ge=0.0, le=1.0)
    contradiction_score: float = Field(..., ge=0.0, le=1.0)


class ClaimVerificationResponse(BaseModel):
    """
    The verification outcome for a single claim.

    Attributes:
        claim_id: Unique identifier of the verified claim.
        claim_text: The claim's text content.
        status: The overall verification outcome.
        confidence: Confidence in ``status``, in ``[0, 1]``.
        explanation: A short, human-readable explanation of how
            ``status`` was reached.
        evidence: Per-evidence support/contradiction assessments that
            informed this result.
    """

    claim_id: str
    claim_text: str
    status: VerificationStatus
    confidence: float = Field(..., ge=0.0, le=1.0)
    explanation: str
    evidence: list[EvidenceAssessmentSummary] = Field(default_factory=list)

    @classmethod
    def from_result(cls, result: VerificationResult) -> "ClaimVerificationResponse":
        """
        Build a response from an internal :class:`VerificationResult`.

        Args:
            result: The verification result to convert.

        Returns:
            A populated :class:`ClaimVerificationResponse`.
        """
        evidence_summaries = [
            EvidenceAssessmentSummary(
                chunk_id=assessment.evidence.chunk_id,
                text=assessment.evidence.text,
                source_name=assessment.evidence.attribution.source_name,
                support_score=assessment.support_score,
                contradiction_score=assessment.contradiction_score,
            )
            for assessment in result.assessments
        ]
        return cls(
            claim_id=result.claim.claim_id,
            claim_text=result.claim.text,
            status=result.status,
            confidence=result.confidence,
            explanation=result.explanation,
            evidence=evidence_summaries,
        )


class BatchClaimVerificationRequest(BaseModel):
    """
    Request payload for verifying multiple claims in a single call.

    Attributes:
        claims: The claim texts to verify.
        source_response_id: Identifier of the source LLM response these
            claims were extracted from, if known. Applied to every
            claim in the batch.
        top_k: Maximum number of evidence chunks to retrieve per claim.
            If omitted, the server-side configured default is used.
        document_id: If provided, restricts evidence retrieval to this
            specific document only, for every claim in the batch.
    """

    claims: list[str] = Field(..., min_length=1, max_length=50)
    source_response_id: Optional[str] = None
    top_k: Optional[int] = Field(default=None, ge=1, le=100)
    document_id: Optional[str] = None


class BatchClaimVerificationResponse(BaseModel):
    """
    The verification outcomes for a batch of claims.

    Attributes:
        total: Number of claims verified.
        results: The per-claim verification results, in the same order
            as the request's ``claims``.
    """

    total: int = Field(..., ge=0)
    results: list[ClaimVerificationResponse] = Field(default_factory=list)