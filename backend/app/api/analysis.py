"""
Analysis API routes.

Exposes the hallucination-detection pipeline as a REST endpoint.

Pipeline:

    LLM response
        -> Claim Extraction
        -> Hallucination Detection
        -> RAG Evidence Retrieval
        -> Confidence Scoring
        -> Final Verdict

The ResponseAnalyzer is created once per worker process and is wired
to the application's real Retriever so hallucination detection uses
the documents indexed in the vector database.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.api.routes.retrieval import get_retriever
from app.core.logging import logger
from app.retrieval.retriever import Retriever
from app.services.response_analyzer import (
    ResponseAnalysisError,
    ResponseAnalyzer,
)


__all__ = ["router"]


# ============================================================================
# ROUTER
# ============================================================================

router = APIRouter(
    tags=["Analysis"],
)


# ============================================================================
# REQUEST / RESPONSE SCHEMAS
# ============================================================================


class AnalyzeRequest(BaseModel):
    """
    Request payload for hallucination analysis.
    """

    response_text: str = Field(
        ...,
        min_length=1,
        max_length=20_000,
        description="Raw LLM-generated response text to analyze.",
    )


class ScoreBreakdown(BaseModel):
    """
    Aggregate confidence metrics for the analyzed response.
    """

    trust_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
    )

    reliability_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
    )

    hallucination_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
    )

    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
    )

    risk_level: str


class ClaimSummary(BaseModel):
    """
    Counts of claims by verification outcome.
    """

    total_claims: int = Field(
        ...,
        ge=0,
    )

    supported: int = Field(
        ...,
        ge=0,
    )

    contradicted: int = Field(
        ...,
        ge=0,
    )

    insufficient_evidence: int = Field(
        ...,
        ge=0,
    )


class EvidenceItem(BaseModel):
    """
    A single piece of retrieved evidence backing (or contradicting) a claim.

    Carries full source attribution and similarity score through to the
    frontend, rather than collapsing evidence down to bare text.
    """

    text: str

    source: str

    similarity_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
    )


class ClaimDetail(BaseModel):
    """
    Full detail of one extracted claim.
    """

    id: int

    text: str

    claim_type: str

    entities: list[dict[str, Any]]

    extraction_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
    )

    verification_status: str

    evidence: list[EvidenceItem]

    source: str | None

    verified: bool

    created_at: str


class AnalyzeResponse(BaseModel):
    """
    Final structured response returned by /analyze.
    """

    verdict: str

    verdict_reason: str

    analyzed_at: str

    scores: ScoreBreakdown | None

    claim_summary: ClaimSummary

    claims: list[ClaimDetail]


# ============================================================================
# RESPONSE ANALYZER DEPENDENCY
# ============================================================================


@lru_cache(maxsize=1)
def _build_response_analyzer() -> ResponseAnalyzer:
    """
    Create the process-wide ResponseAnalyzer singleton.

    The real application Retriever is injected into ResponseAnalyzer.

    This ensures the hallucination detector uses the same indexed
    documents/vector database used by the retrieval API.
    """

    logger.info(
        "Initializing ResponseAnalyzer singleton with RAG Retriever."
    )

    # Get the application's shared Retriever.
    retriever: Retriever = get_retriever()

    # Inject the Retriever into ResponseAnalyzer.
    #
    # ResponseAnalyzer internally creates:
    #
    #     Retriever
    #         ↓
    #     _RetrieverEvidenceSource
    #         ↓
    #     HallucinationDetector
    #         ↓
    #     Evidence-backed claim verification
    #
    return ResponseAnalyzer(
        retriever=retriever,
    )


def get_response_analyzer() -> ResponseAnalyzer:
    """
    FastAPI dependency that returns the shared ResponseAnalyzer.
    """

    return _build_response_analyzer()


# ============================================================================
# ANALYZE ENDPOINT
# ============================================================================


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyze an LLM response for hallucinations",
    response_description=(
        "Structured hallucination analysis with "
        "RAG-backed evidence and per-claim verdicts."
    ),
)
def analyze_response(
    payload: AnalyzeRequest,
    analyzer: ResponseAnalyzer = Depends(
        get_response_analyzer
    ),
) -> AnalyzeResponse:
    """
    Run the complete hallucination-detection pipeline.

    Pipeline:

        1. Extract claims
        2. Retrieve relevant evidence
        3. Detect supported / contradicted / insufficient claims
        4. Calculate confidence scores
        5. Generate final verdict
        6. Return structured JSON
    """

    # ------------------------------------------------------------------------
    # INPUT VALIDATION
    # ------------------------------------------------------------------------

    response_text = payload.response_text.strip()

    if not response_text:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Response text cannot be empty.",
        )

    # ------------------------------------------------------------------------
    # RUN ANALYSIS
    # ------------------------------------------------------------------------

    try:
        analysis = analyzer.analyze(
            response_text
        )

    except ResponseAnalysisError as exc:
        logger.warning(
            "Response analysis rejected input: %s",
            exc,
        )

        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception(
            "Unexpected failure while analyzing response."
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "An unexpected error occurred while "
                "analyzing the response."
            ),
        ) from exc

    # ------------------------------------------------------------------------
    # SERIALIZE DOMAIN RESULT
    # ------------------------------------------------------------------------

    try:
        result = analysis.to_dict()

        return AnalyzeResponse(
            **result,
        )

    except Exception as exc:
        logger.exception(
            "Failed to serialize response analysis."
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "Analysis completed but the result "
                "could not be serialized."
            ),
        ) from exc