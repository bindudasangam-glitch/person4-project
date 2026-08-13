"""
Verification domain model.

Defines the models that flow through the fact-verification stage:
:class:`Claim` (a single factual assertion to be checked),
:class:`EvidenceAssessment` (one piece of evidence's support/
contradiction scores with respect to a claim), and
:class:`VerificationResult` (the aggregate verdict for a claim).

Deliberately reuses :class:`app.models.claim_model.VerificationStatus`
(Person 1's enum) rather than redefining an equivalent-but-incompatible
one: Person 1's enum already has exactly the four members Person 2's
verification pipeline needs (``UNVERIFIED``, ``SUPPORTED``,
``CONTRADICTED``, ``INSUFFICIENT_EVIDENCE``). This module only *reads*
that enum -- it never imports or depends on anything from
``app.services``, so no circular import is introduced.
"""

from __future__ import annotations

import uuid
from typing import Optional

from pydantic import BaseModel, Field

from app.models.claim_model import VerificationStatus
from app.models.evidence import Evidence

__all__ = ["Claim", "EvidenceAssessment", "VerificationResult", "VerificationStatus"]


class Claim(BaseModel):
    """
    A single factual assertion submitted for verification against
    retrieved evidence.

    Attributes:
        claim_id: Unique identifier for this claim, generated
            automatically if not provided.
        text: The claim's text content.
        source_response_id: Identifier of the source LLM response this
            claim was extracted from, if known.
    """

    claim_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str
    source_response_id: Optional[str] = None


class EvidenceAssessment(BaseModel):
    """
    The support/contradiction scores computed for a single piece of
    evidence with respect to a claim.

    Attributes:
        evidence: The evidence this assessment refers to.
        support_score: Degree to which this evidence supports the
            claim, in ``[0, 1]``.
        contradiction_score: Degree to which this evidence contradicts
            the claim, in ``[0, 1]``.
    """

    evidence: Evidence
    support_score: float = Field(..., ge=0.0, le=1.0)
    contradiction_score: float = Field(..., ge=0.0, le=1.0)


class VerificationResult(BaseModel):
    """
    The aggregate verification verdict for a single claim.

    Attributes:
        claim: The claim this result pertains to.
        status: The overall verification outcome.
        confidence: Confidence in ``status``, in ``[0, 1]``.
        assessments: Per-evidence support/contradiction assessments
            that informed this result. Empty if no evidence was retrieved.
        explanation: A short, human-readable explanation of how
            ``status`` was reached.
    """

    claim: Claim
    status: VerificationStatus
    confidence: float = Field(..., ge=0.0, le=1.0)
    assessments: list[EvidenceAssessment] = Field(default_factory=list)
    explanation: str = ""

    @classmethod
    def unverified(cls, claim: Claim) -> "VerificationResult":
        """
        Build a result for a claim that could not be checked at all
        (e.g. no evidence was retrieved for it).

        Args:
            claim: The claim that could not be verified.

        Returns:
            A :class:`VerificationResult` with status ``UNVERIFIED``,
            zero confidence, and no assessments.
        """
        return cls(
            claim=claim,
            status=VerificationStatus.UNVERIFIED,
            confidence=0.0,
            assessments=[],
            explanation="No evidence was retrieved for this claim.",
        )