"""
Response Analyzer Service
=========================

Top-level orchestrator for the hallucination-detection pipeline.

Pipeline:

    LLM response
        -> Claim Extraction
        -> Hallucination Detection
        -> RAG Evidence Retrieval
        -> Confidence Scoring
        -> Final Verdict

The ResponseAnalyzer is responsible for connecting the claim extractor,
hallucination detector, and confidence scorer.

When a Retriever is supplied, the hallucination detector uses the same
RAG/vector-store evidence used by the retrieval API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any

from app.core.logging import logger

from app.models.claim_model import (
    ClaimModel,
    Entity,
    VerificationStatus,
)

from app.services.claim_extractor import (
    ClaimExtractionError,
    ClaimExtractor,
    ExtractedClaim,
)

from app.services.confidence_scorer import (
    ConfidenceScorer,
    ConfidenceScoreResult,
    ConfidenceScoringError,
    RiskLevel,
)

from app.services.hallucination_detector import (
    ClaimDetectionOutcome,
    EvidencePassage,
    HallucinationDetectionError,
    HallucinationDetector,
)


# ---------------------------------------------------------------------------
# TYPE CHECKING ONLY
# ---------------------------------------------------------------------------
#
# IMPORTANT:
# Do NOT import Retriever normally here.
#
# app.retrieval.retriever may depend on other application modules.
# Importing it at runtime from this module can create circular imports.
#
# The actual Retriever object is injected into ResponseAnalyzer by the API
# dependency layer.
# ---------------------------------------------------------------------------

if TYPE_CHECKING:
    from app.retrieval.retriever import Retriever


__all__ = [
    "ResponseAnalysisError",
    "Verdict",
    "ResponseAnalysis",
    "ResponseAnalyzer",
]


# ===========================================================================
# EXCEPTIONS
# ===========================================================================


class ResponseAnalysisError(Exception):
    """
    Raised when the end-to-end response analysis pipeline fails.
    """

    pass


# ===========================================================================
# VERDICT
# ===========================================================================


class Verdict(str, Enum):
    """
    Final human-facing verdict for an analyzed response.
    """

    RELIABLE = "reliable"
    MOSTLY_RELIABLE = "mostly_reliable"
    QUESTIONABLE = "questionable"
    UNRELIABLE = "unreliable"
    NO_VERIFIABLE_CLAIMS = "no_verifiable_claims"


# ===========================================================================
# RESPONSE ANALYSIS RESULT
# ===========================================================================


@dataclass(frozen=True, slots=True)
class ResponseAnalysis:
    """
    Final structured result of analyzing one LLM response.
    """

    response_text: str

    claims: tuple[ClaimModel, ...]

    detection_outcomes: tuple[ClaimDetectionOutcome, ...]

    confidence: ConfidenceScoreResult | None

    verdict: Verdict

    verdict_reason: str

    analyzed_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    # -----------------------------------------------------------------------
    # SERIALIZATION
    # -----------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the analysis result into a JSON-compatible dictionary.

        This is consumed by the FastAPI /analyze endpoint.

        Evidence is deliberately rebuilt here from ``detection_outcomes``
        (matched to each claim by ``claim_id``) rather than read off
        ``ClaimModel.evidence``. ``ClaimModel.evidence`` is a flat
        ``tuple[str, ...]`` (text only) by design -- the richer
        ``EvidencePassage`` objects (text + source + relevance_score)
        produced by the retriever/detector live only on
        ``ClaimDetectionOutcome.evidence``. Serializing from there is what
        lets the API response carry real source attribution and similarity
        scores through to the frontend instead of losing them.
        """

        outcome_by_claim_id = {
            outcome.claim_id: outcome
            for outcome in self.detection_outcomes
        }

        def _serialize_claim(claim: ClaimModel) -> dict[str, Any]:
            data = claim.to_dict()

            outcome = outcome_by_claim_id.get(claim.id)

            if outcome is not None and outcome.evidence:
                data["evidence"] = [
                    {
                        "text": passage.text,
                        "source": passage.source,
                        "similarity_score": passage.relevance_score,
                    }
                    for passage in outcome.evidence
                ]
            else:
                data["evidence"] = []

            return data

        return {
            "verdict": self.verdict.value,

            "verdict_reason": self.verdict_reason,

            "analyzed_at": self.analyzed_at.isoformat(),

            "scores": (
                {
                    "trust_score": self.confidence.trust_score,
                    "reliability_score": self.confidence.reliability_score,
                    "hallucination_probability": (
                        self.confidence.hallucination_probability
                    ),
                    "confidence_score": self.confidence.confidence_score,
                    "risk_level": self.confidence.risk_level.value,
                }
                if self.confidence is not None
                else None
            ),

            "claim_summary": {
                "total_claims": len(self.claims),

                "supported": sum(
                    1
                    for claim in self.claims
                    if claim.verification_status
                    is VerificationStatus.SUPPORTED
                ),

                "contradicted": sum(
                    1
                    for claim in self.claims
                    if claim.verification_status
                    is VerificationStatus.CONTRADICTED
                ),

                "insufficient_evidence": sum(
                    1
                    for claim in self.claims
                    if claim.verification_status
                    is VerificationStatus.INSUFFICIENT_EVIDENCE
                ),
            },

            "claims": [
                _serialize_claim(claim)
                for claim in self.claims
            ],
        }


# ===========================================================================
# RAG EVIDENCE ADAPTER
# ===========================================================================


class _RetrieverEvidenceSource:
    """
    Adapter between the application's Retriever and the
    HallucinationDetector EvidenceSource interface.

    The Retriever returns EvidenceBundle objects.

    HallucinationDetector expects EvidencePassage objects.

    This class performs only that conversion.
    """

    def __init__(self, retriever: Retriever) -> None:
        self._retriever = retriever

    def retrieve(
        self,
        claim_text: str,
        top_k: int = 3,
    ) -> list[EvidencePassage]:
        """
        Retrieve evidence from the application's vector database.

        Args:
            claim_text:
                Claim that needs verification.

            top_k:
                Maximum number of evidence passages.

        Returns:
            EvidencePassage objects for the hallucination detector.
        """

        if not claim_text or not claim_text.strip():
            return []

        try:
            bundle = self._retriever.retrieve(
                query=claim_text.strip(),
                top_k=top_k,
            )

        except Exception:
            logger.exception(
                "RAG evidence retrieval failed for claim: %s",
                claim_text,
            )

            # Do not crash the entire analysis because retrieval failed.
            # The detector can classify the claim as insufficient evidence.
            return []

        if bundle is None:
            return []

        results = getattr(bundle, "results", None)

        if not results:
            return []

        passages: list[EvidencePassage] = []

        for evidence in results:

            text = getattr(evidence, "text", None)

            if not text:
                continue

            attribution = getattr(
                evidence,
                "attribution",
                None,
            )

            source_name = "Unknown source"

            if attribution is not None:
                source_name = getattr(
                    attribution,
                    "source_name",
                    None,
                ) or "Unknown source"

            relevance_score = getattr(
                evidence,
                "similarity_score",
                0.0,
            )

            try:
                relevance_score = float(relevance_score)
            except (TypeError, ValueError):
                relevance_score = 0.0

            relevance_score = max(
                0.0,
                min(
                    1.0,
                    relevance_score,
                ),
            )

            passages.append(
                EvidencePassage(
                    text=str(text),
                    source=str(source_name),
                    relevance_score=relevance_score,
                )
            )

            if len(passages) >= top_k:
                break

        return passages


# ===========================================================================
# RESPONSE ANALYZER
# ===========================================================================


class ResponseAnalyzer:
    """
    Main orchestrator for the hallucination detection pipeline.

    Pipeline:

        response text
            ↓
        ClaimExtractor
            ↓
        HallucinationDetector
            ↓
        ConfidenceScorer
            ↓
        ResponseAnalysis
    """

    # -----------------------------------------------------------------------
    # Risk -> final verdict mapping
    # -----------------------------------------------------------------------

    _RISK_TO_VERDICT: dict[RiskLevel, Verdict] = {
        RiskLevel.LOW: Verdict.RELIABLE,
        RiskLevel.MEDIUM: Verdict.MOSTLY_RELIABLE,
        RiskLevel.HIGH: Verdict.UNRELIABLE,
    }

    # -----------------------------------------------------------------------
    # CONSTRUCTOR
    # -----------------------------------------------------------------------

    def __init__(
        self,
        claim_extractor: ClaimExtractor | None = None,
        hallucination_detector: HallucinationDetector | None = None,
        confidence_scorer: ConfidenceScorer | None = None,
        retriever: Retriever | None = None,
    ) -> None:
        """
        Create a ResponseAnalyzer.

        Dependencies can be injected for testing.

        If a Retriever is supplied and a HallucinationDetector is not
        explicitly supplied, the detector is automatically connected
        to the real RAG retriever.
        """

        self._claim_extractor = (
            claim_extractor
            if claim_extractor is not None
            else ClaimExtractor()
        )

        self._retriever = retriever

        # ---------------------------------------------------------------
        # Hallucination detector
        # ---------------------------------------------------------------

        if hallucination_detector is not None:

            # Use explicitly supplied detector.
            self._hallucination_detector = hallucination_detector

        elif retriever is not None:

            # IMPORTANT:
            # Connect the detector to the application's RAG retriever.
            self._hallucination_detector = HallucinationDetector(
                evidence_source=_RetrieverEvidenceSource(
                    retriever
                )
            )

        else:

            # Fallback detector.
            self._hallucination_detector = HallucinationDetector()

        # ---------------------------------------------------------------
        # Confidence scorer
        # ---------------------------------------------------------------

        self._confidence_scorer = (
            confidence_scorer
            if confidence_scorer is not None
            else ConfidenceScorer()
        )


    # =========================================================================
    # APPLY DETECTION RESULTS TO CLAIM MODELS
    # =========================================================================

    @staticmethod
    def _apply_detection_outcomes(
        claims: tuple[ClaimModel, ...],
        outcomes: list[ClaimDetectionOutcome],
    ) -> tuple[ClaimModel, ...]:
        """
        Ensure each ClaimModel's verification_status / verified / evidence /
        source reflect the HallucinationDetector's outcome for that claim.

        This is important because the confidence scorer reads ClaimModel
        verification_status. If detector results are not copied back,
        a contradicted claim can incorrectly be scored as reliable.

        NOTE: HallucinationDetector.detect() already mutates each claim in
        place via claim.mark_verified(), so under normal operation this is
        a safety-net re-application rather than the primary write path. It
        matches outcomes to claims by ``claim_id`` -- the only identifier
        both ``ClaimModel`` and ``ClaimDetectionOutcome`` actually carry.
        ClaimDetectionOutcome has no claim text on it, so matching by text
        (as a previous version of this method attempted) can never find a
        match and silently does nothing; claim_id matching is correct and
        deterministic instead.
        """

        if not claims or not outcomes:
            return claims

        outcome_by_claim_id: dict[int, ClaimDetectionOutcome] = {
            outcome.claim_id: outcome
            for outcome in outcomes
        }

        updated: list[ClaimModel] = []

        for claim in claims:
            outcome = outcome_by_claim_id.get(claim.id)

            if outcome is None:
                updated.append(claim)
                continue

            claim.verification_status = outcome.verification_status
            claim.verified = (
                outcome.verification_status is VerificationStatus.SUPPORTED
            )
            claim.evidence = tuple(
                passage.text for passage in outcome.evidence
            )

            if outcome.evidence:
                best_passage = max(
                    outcome.evidence,
                    key=lambda passage: passage.relevance_score,
                )
                claim.source = best_passage.source

            updated.append(claim)

        return tuple(updated)

    # =========================================================================
    # MAIN ANALYSIS METHOD
    # =========================================================================

    def analyze(
        self,
        response_text: str,
    ) -> ResponseAnalysis:
        """
        Run the complete hallucination-detection pipeline.

        Steps:

            1. Validate input
            2. Extract factual claims
            3. Hydrate claim models
            4. Retrieve RAG evidence
            5. Detect supported/contradicted/unsupported claims
            6. Calculate confidence
            7. Derive final verdict
            8. Return ResponseAnalysis
        """

        # -------------------------------------------------------------------
        # Validate input
        # -------------------------------------------------------------------

        if response_text is None:
            raise ResponseAnalysisError(
                "Cannot analyze empty response text."
            )

        response_text = response_text.strip()

        if not response_text:
            raise ResponseAnalysisError(
                "Cannot analyze empty response text."
            )

        logger.info(
            "Starting response analysis. text_length=%d",
            len(response_text),
        )

        # -------------------------------------------------------------------
        # STEP 1: CLAIM EXTRACTION
        # -------------------------------------------------------------------

        try:

            extracted_claims = self._claim_extractor.extract(
                response_text
            )

        except ClaimExtractionError as exc:

            logger.exception(
                "Claim extraction failed during response analysis."
            )

            raise ResponseAnalysisError(
                "Claim extraction stage failed."
            ) from exc

        except Exception as exc:

            logger.exception(
                "Unexpected error during claim extraction."
            )

            raise ResponseAnalysisError(
                "Claim extraction stage failed unexpectedly."
            ) from exc

        # -------------------------------------------------------------------
        # No claims
        # -------------------------------------------------------------------

        if not extracted_claims:

            logger.info(
                "No verifiable claims found in response."
            )

            return ResponseAnalysis(
                response_text=response_text,
                claims=(),
                detection_outcomes=(),
                confidence=None,
                verdict=Verdict.NO_VERIFIABLE_CLAIMS,
                verdict_reason=(
                    "No independently checkable factual claims "
                    "were found in the response."
                ),
            )

        logger.info(
            "Claim extraction complete. claims=%d",
            len(extracted_claims),
        )

        # -------------------------------------------------------------------
        # STEP 2: HYDRATE CLAIM MODELS
        # -------------------------------------------------------------------

        try:

            claim_models = tuple(
                self._hydrate_claim_model(
                    extracted_claim
                )
                for extracted_claim in extracted_claims
            )

        except Exception as exc:

            logger.exception(
                "Failed to prepare extracted claims."
            )

            raise ResponseAnalysisError(
                "Claim preparation stage failed."
            ) from exc

        # -------------------------------------------------------------------
        # STEP 3: HALLUCINATION DETECTION + RAG
        # -------------------------------------------------------------------

        try:

            outcomes = self._hallucination_detector.detect(
                list(claim_models)
            )

        except HallucinationDetectionError as exc:

            logger.exception(
                "Hallucination detection failed."
            )

            raise ResponseAnalysisError(
                "Hallucination detection stage failed."
            ) from exc

        except Exception as exc:

            logger.exception(
                "Unexpected error during hallucination detection."
            )

            raise ResponseAnalysisError(
                "Hallucination detection stage failed unexpectedly."
            ) from exc

        logger.info(
            "Hallucination detection complete. outcomes=%d",
            len(outcomes),
        )

        # IMPORTANT:
        # The detector has the RAG-backed verification result.
        # Copy that result into ClaimModel before confidence scoring.
        claim_models = self._apply_detection_outcomes(
            claim_models,
            list(outcomes),
        )

        # -------------------------------------------------------------------
        # STEP 4: CONFIDENCE SCORING
        # -------------------------------------------------------------------

        try:

            confidence = self._confidence_scorer.score(
                list(claim_models)
            )

        except ConfidenceScoringError as exc:

            logger.exception(
                "Confidence scoring failed."
            )

            raise ResponseAnalysisError(
                "Confidence scoring stage failed."
            ) from exc

        except Exception as exc:

            logger.exception(
                "Unexpected error during confidence scoring."
            )

            raise ResponseAnalysisError(
                "Confidence scoring stage failed unexpectedly."
            ) from exc

        # -------------------------------------------------------------------
        # STEP 5: FINAL VERDICT
        # -------------------------------------------------------------------

        verdict, reason = self._derive_verdict(
            confidence
        )

        # -------------------------------------------------------------------
        # STEP 6: FINAL RESULT
        # -------------------------------------------------------------------

        analysis = ResponseAnalysis(
            response_text=response_text,
            claims=claim_models,
            detection_outcomes=tuple(outcomes),
            confidence=confidence,
            verdict=verdict,
            verdict_reason=reason,
        )

        logger.info(
            "Response analysis complete: "
            "verdict=%s claims=%d trust_score=%.4f",
            verdict.value,
            len(claim_models),
            confidence.trust_score,
        )

        return analysis


    # =========================================================================
    # CLAIM HYDRATION
    # =========================================================================

    @staticmethod
    def _hydrate_claim_model(
        extracted: ExtractedClaim,
    ) -> ClaimModel:
        """
        Copy extraction metadata into the underlying ClaimModel.

        The ClaimExtractor provides:

            - claim
            - claim_type
            - entities
            - extraction_confidence

        These are merged into the ClaimModel used by the downstream
        hallucination detector and confidence scorer.
        """

        claim_model = extracted.claim

        claim_model.claim_type = (
            extracted.claim_type
        )

        claim_model.extraction_confidence = (
            extracted.extraction_confidence
        )

        claim_model.entities = tuple(
            Entity(
                text=entity.text,
                label=entity.label,
                start_char=entity.start_char,
                end_char=entity.end_char,
            )
            for entity in extracted.entities
        )

        return claim_model


    # =========================================================================
    # VERDICT DERIVATION
    # =========================================================================

    def _derive_verdict(
        self,
        confidence: ConfidenceScoreResult,
    ) -> tuple[Verdict, str]:
        """
        Convert confidence/risk information into the final verdict.
        """

        base_verdict = self._RISK_TO_VERDICT.get(
            confidence.risk_level,
            Verdict.QUESTIONABLE,
        )

        # -------------------------------------------------------------------
        # Explicit contradictions take priority.
        # -------------------------------------------------------------------

        if confidence.contradicted_claims > 0:
            verdict = Verdict.UNRELIABLE

            reason = (
                f"{confidence.contradicted_claims} of "
                f"{confidence.total_claims} claim(s) contradict "
                "retrieved evidence. "
                f"Trust score {confidence.trust_score:.2f}, "
                f"hallucination probability "
                f"{confidence.hallucination_probability:.2f}."
            )

            return verdict, reason

        # -------------------------------------------------------------------
        # Normal verdict
        # -------------------------------------------------------------------

        reason = (
            f"Trust score {confidence.trust_score:.2f}, "
            f"hallucination probability "
            f"{confidence.hallucination_probability:.2f}, "
            f"reliability "
            f"{confidence.reliability_score:.2f} across "
            f"{confidence.total_claims} claim(s) "
            f"({confidence.supported_claims} supported, "
            f"{confidence.contradicted_claims} contradicted, "
            f"{confidence.insufficient_evidence_claims} "
            "insufficient evidence)."
        )

        return base_verdict, reason