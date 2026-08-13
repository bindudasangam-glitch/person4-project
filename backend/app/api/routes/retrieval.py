"""
Evidence retrieval API route.

Exposes an endpoint that accepts a natural-language query and returns
ranked, deduplicated, threshold-filtered evidence retrieved from the
vector store, with full source attribution.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.routes.documents import get_embedding_service, get_vector_store
from app.core.exceptions import CollectionNotFoundError, RAGFactVerificationError
from app.embeddings.embedding_service import EmbeddingService
from app.retrieval.retriever import Retriever
from app.schemas.retrieval_schema import RetrievalRequest, RetrievalResponse
from app.vectorstore.chroma_client import ChromaVectorStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/retrieval", tags=["retrieval"])


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    """
    Return the process-wide cached :class:`Retriever` instance, built on
    top of the shared embedding service and vector store singletons.
    """
    return Retriever(
        embedding_service=get_embedding_service(),
        vector_store=get_vector_store(),
    )


@router.post(
    "/query",
    response_model=RetrievalResponse,
    summary="Retrieve semantically relevant evidence for a query.",
)
async def retrieve_evidence(
    request: RetrievalRequest,
    retriever: Retriever = Depends(get_retriever),
) -> RetrievalResponse:
    """
    Retrieve ranked, deduplicated evidence for a natural-language query.

    Args:
        request: The retrieval request, containing the query text and
            optional top-k, similarity threshold, and document filter.

    Returns:
        A :class:`RetrievalResponse` containing the retrieved evidence.

    Raises:
        HTTPException: 404 if the target collection does not exist yet
            (i.e. no documents have been ingested); 400 for other
            application-level errors; 500 for unexpected failures.
    """
    try:
        bundle = retriever.retrieve(
            query=request.query,
            top_k=request.top_k,
            similarity_threshold=request.similarity_threshold,
            document_id=request.document_id,
        )
        return RetrievalResponse.from_bundle(bundle)

    except CollectionNotFoundError as exc:
        logger.warning("Retrieval attempted before any documents were ingested: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No indexed documents found. Upload documents before querying.",
        ) from exc
    except RAGFactVerificationError as exc:
        logger.warning("Retrieval failed for query: %s", exc)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - guarantee a clean 500 rather than a raw traceback
        logger.exception("Unexpected error during evidence retrieval.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error during retrieval: {exc}",
        ) from exc