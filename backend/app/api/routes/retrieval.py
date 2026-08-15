"""
Evidence retrieval API route.

Heavy retrieval dependencies are imported lazily so FastAPI startup
remains lightweight on memory-constrained environments such as Render.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.routes.documents import (
    get_embedding_service,
    get_vector_store,
)
from app.core.exceptions import (
    CollectionNotFoundError,
    RAGFactVerificationError,
)
from app.schemas.retrieval_schema import (
    RetrievalRequest,
    RetrievalResponse,
)


if TYPE_CHECKING:
    from app.retrieval.retriever import Retriever


logger = logging.getLogger(__name__)


router = APIRouter(
    prefix="/retrieval",
    tags=["retrieval"],
)


# ============================================================================
# LAZY RETRIEVER
# ============================================================================


@lru_cache(maxsize=1)
def get_retriever() -> "Retriever":
    """
    Return the process-wide cached Retriever.

    IMPORTANT:
    The Retriever module is imported only when the retrieval endpoint
    actually needs it.

    This prevents retrieval-related heavy dependencies from being
    loaded during FastAPI startup.
    """

    # Lazy import.
    from app.retrieval.retriever import Retriever

    return Retriever(
        embedding_service=get_embedding_service(),
        vector_store=get_vector_store(),
    )


# ============================================================================
# RETRIEVAL ENDPOINT
# ============================================================================


@router.post(
    "/query",
    response_model=RetrievalResponse,
    summary="Retrieve semantically relevant evidence for a query.",
)
async def retrieve_evidence(
    request: RetrievalRequest,
    retriever: Any = Depends(get_retriever),
) -> RetrievalResponse:
    """
    Retrieve ranked, deduplicated evidence for a natural-language query.

    The Retriever and its heavy dependencies are initialized lazily
    when this endpoint is actually called.
    """

    try:

        bundle = retriever.retrieve(
            query=request.query,
            top_k=request.top_k,
            similarity_threshold=request.similarity_threshold,
            document_id=request.document_id,
        )

        return RetrievalResponse.from_bundle(
            bundle
        )

    except CollectionNotFoundError as exc:

        logger.warning(
            "Retrieval attempted before any documents "
            "were ingested: %s",
            exc,
        )

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                "No indexed documents found. "
                "Upload documents before querying."
            ),
        ) from exc

    except RAGFactVerificationError as exc:

        logger.warning(
            "Retrieval failed for query: %s",
            exc,
        )

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    except Exception as exc:

        logger.exception(
            "Unexpected error during evidence retrieval."
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "Unexpected error during retrieval: "
                f"{exc}"
            ),
        ) from exc