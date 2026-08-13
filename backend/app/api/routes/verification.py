"""
Fact verification API routes.

Exposes endpoints that accept one or more factual claims, retrieve
relevant evidence for each, and return structured verification results.

Verification statuses may include:
- SUPPORTED
- CONTRADICTED
- INSUFFICIENT_EVIDENCE
- UNVERIFIED
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.routes.documents import get_embedding_service
from app.api.routes.retrieval import get_retriever
from app.core.exceptions import (
    CollectionNotFoundError,
    RAGFactVerificationError,
)
from app.models.verification import Claim
from app.retrieval.retriever import Retriever
from app.schemas.verification_schema import (
    BatchClaimVerificationRequest,
    BatchClaimVerificationResponse,
    ClaimVerificationRequest,
    ClaimVerificationResponse,
)
from app.verification.fact_verifier import FactVerifier

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/verification",
    tags=["verification"],
)


# ---------------------------------------------------------------------------
# Fact verifier
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_fact_verifier() -> FactVerifier:
    """
    Return the process-wide cached FactVerifier instance.

    The verifier uses the shared embedding service singleton.
    """
    return FactVerifier(
        embedding_service=get_embedding_service()
    )


# ---------------------------------------------------------------------------
# Evidence helper
# ---------------------------------------------------------------------------

def _has_evidence(evidence_bundle: Any) -> bool:
    """
    Check whether a retrieval response actually contains evidence.

    This helper supports the common response formats used by the
    retrieval layer without depending on one particular implementation.
    """

    if evidence_bundle is None:
        return False

    # Pydantic/object-style response.
    for attribute in ("results", "evidence", "items"):
        if hasattr(evidence_bundle, attribute):
            value = getattr(evidence_bundle, attribute)

            if value is not None:
                try:
                    return len(value) > 0
                except TypeError:
                    return bool(value)

    # Dictionary-style response.
    if isinstance(evidence_bundle, dict):
        for key in ("results", "evidence", "items"):
            value = evidence_bundle.get(key)

            if value is not None:
                try:
                    return len(value) > 0
                except TypeError:
                    return bool(value)

    # If the bundle itself is a list/tuple.
    if isinstance(evidence_bundle, (list, tuple)):
        return len(evidence_bundle) > 0

    return False


# ---------------------------------------------------------------------------
# Single claim verification
# ---------------------------------------------------------------------------

def _verify_single_claim(
    claim_text: str,
    source_response_id: str | None,
    top_k: int | None,
    document_id: str | None,
    retriever: Retriever,
    verifier: FactVerifier,
) -> ClaimVerificationResponse:
    """
    Verify a single factual claim.

    Process:
        1. Create a Claim object.
        2. Retrieve relevant evidence.
        3. If a document filter was supplied but produced no evidence,
           retry retrieval without the document filter.
        4. Pass the retrieved evidence to FactVerifier.
        5. Convert the verification result into the API response schema.
    """

    claim = Claim(
        text=claim_text,
        source_response_id=source_response_id,
    )

    # ------------------------------------------------------------------
    # First retrieval attempt
    # ------------------------------------------------------------------

    logger.info(
        "Retrieving evidence for claim. document_id=%s, top_k=%s",
        document_id,
        top_k,
    )

    evidence_bundle = retriever.retrieve(
        query=claim.text,
        top_k=top_k,
        document_id=document_id,
    )

    # ------------------------------------------------------------------
    # Fallback retrieval
    # ------------------------------------------------------------------
    #
    # In our testing:
    #   /retrieval/query
    # successfully returned evidence,
    # while /verification/verify returned no evidence when document_id
    # was supplied.
    #
    # Therefore, if the filtered retrieval returns nothing, retry once
    # without document_id.
    # ------------------------------------------------------------------

    if document_id is not None and not _has_evidence(evidence_bundle):
        logger.warning(
            "No evidence found using document_id=%s. "
            "Retrying retrieval without document filter.",
            document_id,
        )

        evidence_bundle = retriever.retrieve(
            query=claim.text,
            top_k=top_k,
            document_id=None,
        )

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    logger.info(
        "Evidence retrieval completed. has_evidence=%s",
        _has_evidence(evidence_bundle),
    )

    result = verifier.verify(
        claim,
        evidence_bundle,
    )

    return ClaimVerificationResponse.from_result(result)


# ---------------------------------------------------------------------------
# Verify one claim
# ---------------------------------------------------------------------------

@router.post(
    "/verify",
    response_model=ClaimVerificationResponse,
    summary="Verify a single factual claim against retrieved evidence.",
)
async def verify_claim(
    request: ClaimVerificationRequest,
    retriever: Retriever = Depends(get_retriever),
    verifier: FactVerifier = Depends(get_fact_verifier),
) -> ClaimVerificationResponse:
    """
    Verify a single factual claim by retrieving relevant evidence
    and comparing it against the claim.
    """

    try:
        return _verify_single_claim(
            claim_text=request.claim,
            source_response_id=request.source_response_id,
            top_k=request.top_k,
            document_id=request.document_id,
            retriever=retriever,
            verifier=verifier,
        )

    except CollectionNotFoundError as exc:
        logger.warning(
            "Claim verification attempted before any documents "
            "were ingested: %s",
            exc,
        )

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                "No indexed documents found. "
                "Upload documents before verifying claims."
            ),
        ) from exc

    except RAGFactVerificationError as exc:
        logger.warning(
            "Claim verification failed: %s",
            exc,
        )

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception(
            "Unexpected error during claim verification."
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "Unexpected error during claim verification: "
                f"{exc}"
            ),
        ) from exc


# ---------------------------------------------------------------------------
# Verify multiple claims
# ---------------------------------------------------------------------------

@router.post(
    "/verify/batch",
    response_model=BatchClaimVerificationResponse,
    summary=(
        "Verify multiple factual claims against retrieved "
        "evidence in a single call."
    ),
)
async def verify_claims_batch(
    request: BatchClaimVerificationRequest,
    retriever: Retriever = Depends(get_retriever),
    verifier: FactVerifier = Depends(get_fact_verifier),
) -> BatchClaimVerificationResponse:
    """
    Verify multiple factual claims.

    Each claim independently retrieves its own evidence.
    """

    try:
        results = [
            _verify_single_claim(
                claim_text=claim_text,
                source_response_id=request.source_response_id,
                top_k=request.top_k,
                document_id=request.document_id,
                retriever=retriever,
                verifier=verifier,
            )
            for claim_text in request.claims
        ]

        return BatchClaimVerificationResponse(
            total=len(results),
            results=results,
        )

    except CollectionNotFoundError as exc:
        logger.warning(
            "Batch claim verification attempted before any "
            "documents were ingested: %s",
            exc,
        )

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                "No indexed documents found. "
                "Upload documents before verifying claims."
            ),
        ) from exc

    except RAGFactVerificationError as exc:
        logger.warning(
            "Batch claim verification failed: %s",
            exc,
        )

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception(
            "Unexpected error during batch claim verification."
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "Unexpected error during batch claim verification: "
                f"{exc}"
            ),
        ) from exc