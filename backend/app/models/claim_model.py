"""
Claim Domain Model
===================

Defines the core domain entity for a single factual claim extracted from an
LLM response, along with the small value objects it is composed of
(:class:`Entity`) and the enumerations that describe its nature
(:class:`ClaimType`) and lifecycle (:class:`VerificationStatus`).

This model is the single source of truth consumed by every downstream
service in the pipeline:

* ``ClaimExtractor``          -> constructs ``ClaimModel`` instances.
* ``HallucinationDetector``   -> reads ``text`` / ``entities`` / ``claim_type``
                                  and writes ``verification_status`` + ``evidence``.
* ``ConfidenceScorer``        -> reads ``extraction_confidence`` and
                                  ``verification_status`` to compute trust scores.
* ``ResponseAnalyzer``        -> aggregates many ``ClaimModel`` instances into
                                  the final structured JSON verdict.

Design notes
------------
* The model is intentionally framework-agnostic (a plain ``dataclass``, not a
  Pydantic ``BaseModel``) so it can be used freely inside services without
  incurring validation overhead on every mutation; API-boundary validation is
  the responsibility of ``app.schemas.claim.Claim``.
* Mutation is restricted to a small, explicit set of methods
  (``mark_verified`` / ``add_evidence``) rather than free attribute
  assignment, keeping claim lifecycle transitions auditable and centralized
  (Single Responsibility / encapsulation).
* All construction is validated in ``__post_init__`` so an invalid
  ``ClaimModel`` can never exist in memory — callers get a fast, explicit
  failure instead of silent corruption propagating downstream.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from app.core.logging import logger

__all__ = [
    "ClaimType",
    "VerificationStatus",
    "Entity",
    "ClaimModel",
    "ClaimValidationError",
]


class ClaimValidationError(ValueError):
    """Raised when a ``ClaimModel`` is constructed or mutated with invalid data."""


class ClaimType(str, Enum):
    """Coarse-grained linguistic/semantic category of a claim."""

    NUMERIC = "numeric"
    TEMPORAL = "temporal"
    ENTITY_CENTRIC = "entity_centric"
    FACTUAL = "factual"
    OPINION = "opinion"
    UNVERIFIABLE = "unverifiable"


class VerificationStatus(str, Enum):
    """Lifecycle state of a claim as it moves through hallucination detection."""

    UNVERIFIED = "unverified"
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


@dataclass(frozen=True, slots=True)
class Entity:
    """An immutable named entity recognized within a claim's text span."""

    text: str
    label: str
    start_char: int
    end_char: int

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ClaimValidationError("Entity.text must not be empty.")
        if self.start_char < 0 or self.end_char <= self.start_char:
            raise ClaimValidationError(
                f"Entity char span is invalid: start={self.start_char}, "
                f"end={self.end_char}."
            )


@dataclass(slots=True)
class ClaimModel:
    """
    Represents a single factual claim extracted from an LLM response.

    Attributes:
        id: 1-indexed position of the claim within the source response.
        text: The normalized claim text.
        claim_type: Linguistic/semantic classification of the claim.
        entities: Named entities recognized within the claim text.
        extraction_confidence: Confidence (0.0-1.0) that this span is a
            well-formed, independently checkable claim.
        verification_status: Current stage in the verification lifecycle.
        evidence: Snippets of supporting/contradicting evidence attached by
            the hallucination detector.
        source: Optional identifier of the originating document/response.
        verified: Backward-compatible convenience flag, kept in sync with
            ``verification_status`` (``True`` iff status is ``SUPPORTED``).
        created_at: UTC timestamp of claim construction.
    """

    id: int
    text: str
    claim_type: ClaimType = ClaimType.UNVERIFIABLE
    entities: tuple[Entity, ...] = field(default_factory=tuple)
    extraction_confidence: float = 1.0
    verification_status: VerificationStatus = VerificationStatus.UNVERIFIED
    evidence: tuple[str, ...] = field(default_factory=tuple)
    source: str | None = None
    verified: bool = False
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Enforce structural invariants for this claim."""
        if self.id <= 0:
            raise ClaimValidationError(f"Claim id must be positive, got {self.id}.")

        if not self.text or not self.text.strip():
            raise ClaimValidationError("Claim text must not be empty.")

        if not 0.0 <= self.extraction_confidence <= 1.0:
            raise ClaimValidationError(
                "extraction_confidence must be within [0.0, 1.0], got "
                f"{self.extraction_confidence}."
            )

    def mark_verified(
        self,
        status: VerificationStatus,
        evidence: tuple[str, ...] = (),
    ) -> None:
        """
        Transition this claim to a new verification status.

        Args:
            status: The resulting verification status.
            evidence: Evidence snippets supporting the status transition.

        Raises:
            ClaimValidationError: If ``status`` is not a valid
                :class:`VerificationStatus` member.
        """
        if not isinstance(status, VerificationStatus):
            raise ClaimValidationError(
                f"Invalid verification status: {status!r}."
            )

        self.verification_status = status
        self.verified = status is VerificationStatus.SUPPORTED

        if evidence:
            self.evidence = self.evidence + tuple(evidence)

        logger.debug(
            "Claim %d transitioned to verification_status=%s (verified=%s).",
            self.id,
            self.verification_status.value,
            self.verified,
        )

    def add_evidence(self, snippet: str) -> None:
        """Append a single evidence snippet to this claim without changing status."""
        if not snippet or not snippet.strip():
            logger.warning("Ignored empty evidence snippet for claim %d.", self.id)
            return

        self.evidence = self.evidence + (snippet.strip(),)

    def entity_labels(self) -> frozenset[str]:
        """Return the distinct spaCy entity labels present in this claim."""
        return frozenset(entity.label for entity in self.entities)

    def to_dict(self) -> dict[str, Any]:
        """Serialize this claim to a plain, JSON-compatible dictionary."""
        return {
            "id": self.id,
            "text": self.text,
            "claim_type": self.claim_type.value,
            "entities": [
                {
                    "text": entity.text,
                    "label": entity.label,
                    "start_char": entity.start_char,
                    "end_char": entity.end_char,
                }
                for entity in self.entities
            ],
            "extraction_confidence": self.extraction_confidence,
            "verification_status": self.verification_status.value,
            "evidence": list(self.evidence),
            "source": self.source,
            "verified": self.verified,
            "created_at": self.created_at.isoformat(),
        }