"""
Hallucination Detection Service
=================================

Compares each extracted :class:`app.models.claim_model.ClaimModel` against
retrieved evidence passages and classifies it as supported, contradicted, or
lacking sufficient evidence.

Architecture
------------
Evidence retrieval is abstracted behind the :class:`EvidenceSource`
protocol (dependency inversion / Open-Closed principle). This module ships a
self-contained :class:`LexicalOverlapEvidenceSource` default implementation
that requires no external services, so the detector is fully testable and
usable offline. A production deployment can swap in an embedding-based
retriever (e.g. a LangChain ``VectorStore`` wrapper) without touching
:class:`HallucinationDetector` itself, since it only depends on the
``EvidenceSource`` protocol.

Detection heuristics
---------------------
For each claim, the detector:

1. Retrieves the top-k most relevant evidence passages from the configured
   ``EvidenceSource``.
2. Computes a *lexical support score* (token-overlap / cosine-like Jaccard
   similarity over content words) between the claim and each passage.
3. Computes an *entity agreement score* — the fraction of the claim's named
   entities that also appear in the evidence.
4. Detects *negation mismatches* — cases where the claim and its best
   matching evidence discuss the same entities/topic but disagree on
   polarity (e.g. claim says "X was founded in 1998", evidence says
   "X was not founded until 2001") using spaCy dependency-parsed negation
   cues.
5. Combines these signals into a final :class:`VerificationStatus` and
   attaches the supporting/contradicting evidence text to the claim.

This is a transparent, explainable rule-based baseline suitable for a final
year project; the ``EvidenceSource`` seam is where a learned/embedding-based
retriever would be substituted in a larger production system.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import spacy
from spacy.language import Language

from app.core.logging import logger
from app.models.claim_model import ClaimModel, VerificationStatus

__all__ = [
    "EvidencePassage",
    "EvidenceSource",
    "LexicalOverlapEvidenceSource",
    "HallucinationDetectionError",
    "ClaimDetectionOutcome",
    "HallucinationDetector",
]


class HallucinationDetectionError(Exception):
    """Raised when hallucination detection cannot be completed."""


_STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "of", "in", "on", "at", "to", "for", "and", "or", "but", "with",
        "by", "as", "that", "this", "it", "its", "from", "has", "have",
        "had", "not", "no",
    }
)

_NEGATION_CUES: frozenset[str] = frozenset(
    {"not", "never", "no", "n't", "cannot", "isn't", "wasn't", "weren't", "didn't"}
)

_WORD_PATTERN = re.compile(r"[A-Za-z0-9]+")


@dataclass(frozen=True, slots=True)
class EvidencePassage:
    """A single retrieved piece of evidence considered for a claim."""

    text: str
    source: str
    relevance_score: float = 0.0


@runtime_checkable
class EvidenceSource(Protocol):
    """Abstraction over any system capable of retrieving evidence for a claim."""

    def retrieve(self, claim_text: str, top_k: int = 3) -> list[EvidencePassage]:
        """Return up to ``top_k`` evidence passages relevant to ``claim_text``."""
        ...


class LexicalOverlapEvidenceSource:
    """
    Default, dependency-free :class:`EvidenceSource` implementation.

    Ranks a fixed in-memory corpus of reference passages against a claim
    using content-word (stopword-filtered) Jaccard similarity. Intended as a
    transparent baseline and as a drop-in test double; swap for a vector-store
    backed retriever in production for semantic (not just lexical) recall.
    """

    def __init__(self, corpus: list[str] | None = None) -> None:
        self._corpus: list[str] = list(corpus) if corpus else []

    def add_documents(self, documents: list[str]) -> None:
        """Extend the in-memory evidence corpus with additional documents."""
        self._corpus.extend(doc for doc in documents if doc and doc.strip())

    def retrieve(self, claim_text: str, top_k: int = 3) -> list[EvidencePassage]:
        if not self._corpus:
            return []

        claim_tokens = self._tokenize(claim_text)
        if not claim_tokens:
            return []

        scored: list[EvidencePassage] = []
        for index, passage in enumerate(self._corpus):
            passage_tokens = self._tokenize(passage)
            score = self._jaccard(claim_tokens, passage_tokens)
            if score > 0.0:
                scored.append(
                    EvidencePassage(
                        text=passage,
                        source=f"corpus_doc_{index}",
                        relevance_score=score,
                    )
                )

        scored.sort(key=lambda p: p.relevance_score, reverse=True)
        return scored[:top_k]

    @staticmethod
    def _tokenize(text: str) -> frozenset[str]:
        words = (w.lower() for w in _WORD_PATTERN.findall(text))
        return frozenset(w for w in words if w not in _STOPWORDS)

    @staticmethod
    def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
        if not a or not b:
            return 0.0
        intersection = len(a & b)
        union = len(a | b)
        return intersection / union if union else 0.0


@dataclass(frozen=True, slots=True)
class ClaimDetectionOutcome:
    """Explainable result of running detection against a single claim."""

    claim_id: int
    verification_status: VerificationStatus
    support_score: float
    entity_agreement: float
    negation_mismatch: bool
    entity_disagreement: bool
    evidence: tuple[EvidencePassage, ...]


class HallucinationDetector:
    """
    Detects unsupported, fabricated, or contradicted claims within LLM output.

    Args:
        evidence_source: Any object implementing the :class:`EvidenceSource`
            protocol. Defaults to an empty :class:`LexicalOverlapEvidenceSource`
            (all claims will be marked ``INSUFFICIENT_EVIDENCE`` unless a
            corpus is supplied).
        support_threshold: Minimum combined score to classify a claim as
            ``SUPPORTED``.
        contradiction_threshold: Minimum topical relevance required before a
            negation mismatch is trusted as a genuine ``CONTRADICTED`` signal
            (guards against negation false positives on unrelated passages).
        top_k_evidence: Number of evidence passages retrieved per claim.
    """

    _SPACY_MODEL_NAME = "en_core_web_sm"
    _nlp: Language | None = None

    def __init__(
        self,
        evidence_source: EvidenceSource | None = None,
        support_threshold: float = 0.35,
        contradiction_threshold: float = 0.2,
        top_k_evidence: int = 3,
    ) -> None:
        if not 0.0 <= support_threshold <= 1.0:
            raise HallucinationDetectionError(
                f"support_threshold must be in [0, 1], got {support_threshold}."
            )
        if not 0.0 <= contradiction_threshold <= 1.0:
            raise HallucinationDetectionError(
                f"contradiction_threshold must be in [0, 1], got "
                f"{contradiction_threshold}."
            )

        self._evidence_source: EvidenceSource = evidence_source or LexicalOverlapEvidenceSource()
        self._support_threshold = support_threshold
        self._contradiction_threshold = contradiction_threshold
        self._top_k_evidence = top_k_evidence
        self._ensure_pipeline_loaded()

    @classmethod
    def _ensure_pipeline_loaded(cls) -> None:
        if cls._nlp is not None:
            return
        try:
            logger.info("Loading spaCy pipeline '%s' for hallucination detection.", cls._SPACY_MODEL_NAME)
            cls._nlp = spacy.load(cls._SPACY_MODEL_NAME)
        except OSError as exc:
            logger.error(
                "spaCy model '%s' is not installed. Install it via "
                "'python -m spacy download %s'.",
                cls._SPACY_MODEL_NAME,
                cls._SPACY_MODEL_NAME,
            )
            raise HallucinationDetectionError(
                f"Required spaCy model '{cls._SPACY_MODEL_NAME}' is not available."
            ) from exc

    def detect(self, claims: list[ClaimModel]) -> list[ClaimDetectionOutcome]:
        """
        Run hallucination detection over a batch of claims, mutating each
        claim's ``verification_status`` / ``evidence`` in place.

        Args:
            claims: Claims previously produced by ``ClaimExtractor``.

        Returns:
            One :class:`ClaimDetectionOutcome` per input claim, in order,
            explaining how the verdict was reached.

        Raises:
            HallucinationDetectionError: If ``claims`` is empty or detection
                fails unexpectedly for the batch.
        """
        if not claims:
            raise HallucinationDetectionError("Cannot run detection on an empty claim list.")

        outcomes: list[ClaimDetectionOutcome] = []

        for claim in claims:
            try:
                outcome = self._detect_single(claim)
            except HallucinationDetectionError:
                raise
            except Exception as exc:  # noqa: BLE001 - normalize to domain error
                logger.exception("Unexpected failure detecting claim %d.", claim.id)
                raise HallucinationDetectionError(
                    f"Detection failed for claim {claim.id}."
                ) from exc

            claim.mark_verified(
                outcome.verification_status,
                evidence=tuple(p.text for p in outcome.evidence),
            )
            outcomes.append(outcome)

        supported = sum(1 for o in outcomes if o.verification_status is VerificationStatus.SUPPORTED)
        contradicted = sum(1 for o in outcomes if o.verification_status is VerificationStatus.CONTRADICTED)
        logger.info(
            "Hallucination detection complete: %d claim(s) — %d supported, "
            "%d contradicted, %d insufficient evidence.",
            len(outcomes),
            supported,
            contradicted,
            len(outcomes) - supported - contradicted,
        )
        return outcomes

    def _detect_single(self, claim: ClaimModel) -> ClaimDetectionOutcome:
        """Evaluate a single claim against retrieved evidence."""
        passages = self._evidence_source.retrieve(claim.text, top_k=self._top_k_evidence)

        if not passages:
            return ClaimDetectionOutcome(
                claim_id=claim.id,
                verification_status=VerificationStatus.INSUFFICIENT_EVIDENCE,
                support_score=0.0,
                entity_agreement=0.0,
                negation_mismatch=False,
                entity_disagreement=False,
                evidence=(),
            )

        best_passage = passages[0]
        support_score = best_passage.relevance_score
        entity_agreement = self._entity_agreement(claim, best_passage.text)
        negation_mismatch = self._has_negation_mismatch(claim.text, best_passage.text)
        entity_disagreement = self._has_entity_disagreement(claim, best_passage.text)

        combined_score = round((0.6 * support_score) + (0.4 * entity_agreement), 4)

        if (
            (negation_mismatch or entity_disagreement)
            and support_score >= self._contradiction_threshold
        ):
            status = VerificationStatus.CONTRADICTED
        elif combined_score >= self._support_threshold:
            status = VerificationStatus.SUPPORTED
        else:
            status = VerificationStatus.INSUFFICIENT_EVIDENCE

        return ClaimDetectionOutcome(
            claim_id=claim.id,
            verification_status=status,
            support_score=combined_score,
            entity_agreement=entity_agreement,
            negation_mismatch=negation_mismatch,
            entity_disagreement=entity_disagreement,
            evidence=tuple(passages),
        )

    def _entity_agreement(self, claim: ClaimModel, evidence_text: str) -> float:
        """Fraction of the claim's named entities that also appear in the evidence."""
        if not claim.entities:
            return 0.0

        evidence_lower = evidence_text.lower()
        matches = sum(1 for entity in claim.entities if entity.text.lower() in evidence_lower)
        return matches / len(claim.entities)

    def _has_negation_mismatch(self, claim_text: str, evidence_text: str) -> bool:
        """
        Detect whether the claim and its best evidence disagree on polarity.

        Uses spaCy's dependency parse to find explicit negation particles
        (``neg`` dependency relation) attached to the main predicate of each
        text, then checks whether exactly one of the two carries a negation
        while discussing overlapping content.
        """
        claim_negated = self._contains_negation(claim_text)
        evidence_negated = self._contains_negation(evidence_text)
        return claim_negated != evidence_negated

    def _contains_negation(self, text: str) -> bool:
        doc = self._nlp(text)  # type: ignore[misc]
        for token in doc:
            if token.dep_ == "neg" or token.lower_ in _NEGATION_CUES:
                return True
        return False

    def _has_entity_disagreement(self, claim: ClaimModel, evidence_text: str) -> bool:
        """
        Detect claim/evidence disagreement caused by a substituted named
        entity for the same factual slot, rather than explicit negation.

        Example: claim "The capital of India is Mumbai." against evidence
        "The capital of India is New Delhi." — both sentences are lexically
        near-identical and topically about the same thing (so support_score
        and entity_agreement alone can look high enough to pass as
        SUPPORTED), but the claim's asserted entity ("Mumbai") never
        appears in the evidence, while the evidence instead names a
        *different* entity of the same spaCy label ("New Delhi", also a
        GPE). That pattern — claim entity absent, but a same-label entity
        present in its place — is a strong signal of a fabricated/incorrect
        value rather than genuinely missing evidence.

        This is intentionally conservative: if the evidence doesn't mention
        any entity of that label at all, this returns False (that's just
        "insufficient evidence", not a disagreement).
        """
        if not claim.entities:
            return False

        evidence_doc = self._nlp(evidence_text)  # type: ignore[misc]

        evidence_entities_by_label: dict[str, set[str]] = {}
        for ent in evidence_doc.ents:
            evidence_entities_by_label.setdefault(ent.label_, set()).add(
                ent.text.strip().lower()
            )

        for entity in claim.entities:
            evidence_values = evidence_entities_by_label.get(entity.label)

            if not evidence_values:
                # Evidence doesn't mention this entity type at all -- no
                # basis for a disagreement signal, only for insufficiency.
                continue

            if entity.text.strip().lower() in evidence_values:
                # The claim's entity is actually present in the evidence.
                continue

            # Claim asserts an entity of this label that the evidence does
            # not contain, while the evidence names a different entity of
            # the same label -- treat as a conflicting value.
            return True

        return False