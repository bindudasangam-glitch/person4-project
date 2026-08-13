"""
Multi-Agent Workflow API Routes
====================================

Exposes Person 3's compiled LangGraph multi-agent workflow (claim
extraction/verification reused from Person 1/Person 2, followed by
knowledge-graph validation, knowledge drift detection, risk analysis,
and self-correction) as a REST endpoint.

Wiring
------
This module only *defines* the router. It is activated in
``app/main.py``:

    from app.api.routes.workflow import router as workflow_router
    app.include_router(workflow_router, prefix=settings.API_PREFIX)

Dependency injection
---------------------
:class:`~app.agents.workflow.HallucinationWorkflow` owns several
expensive-to-build collaborators (the spaCy-backed
``ResponseAnalyzer``, the Neo4j/NetworkX-backed
``KnowledgeGraphService``, ...), so a single instance is built once and
cached for the lifetime of the worker process via
:func:`get_hallucination_workflow` (FastAPI ``Depends``), exactly
mirroring the existing pattern already used by
``app.api.analysis.get_response_analyzer`` and
``app.api.routes.retrieval.get_retriever``. The cached instance is
wired with the shared Person 2 ``Retriever`` singleton, so both
Person 1's hallucination detection *and* Person 3's self-correction
supplementary-evidence lookups are backed by real, embedding-based
evidence from ChromaDB rather than an empty in-memory corpus.

Surfacing the full run's state
---------------------------------
:meth:`~app.agents.workflow.HallucinationWorkflow.run` intentionally
returns only the final
:class:`~app.models.workflow_models.WorkflowResult` -- it does not
expose ``hallucination_score``, ``confidence_score``, or
``execution_metadata``, since those live on
:class:`~app.agents.state.WorkflowState` rather than on
``WorkflowResult`` itself. Since this batch may not modify
``app/agents/workflow.py`` to add a new public accessor for them, this
route instead invokes the workflow's already-compiled LangGraph graph
directly (``workflow._compiled_graph``) exactly once per request --
functionally identical to what ``run()`` does internally, just also
retaining the full final state alongside the ``WorkflowResult`` it
produces, so every value the API needs to return is available without
running the graph twice.
"""

from __future__ import annotations

from functools import lru_cache

from fastapi import APIRouter, Depends, HTTPException, status

from app.agents.exceptions import AgentWorkflowError
from app.agents.state import create_initial_state
from app.agents.workflow import HallucinationWorkflow
from app.api.routes.retrieval import get_retriever
from app.core.logging import logger
from app.retrieval.retriever import Retriever
from app.schemas.workflow_schema import WorkflowAnalyzeRequest, WorkflowAnalyzeResponse

__all__ = ["router", "get_hallucination_workflow"]

router = APIRouter(prefix="/workflow", tags=["Workflow"])


@lru_cache(maxsize=1)
def _build_hallucination_workflow(retriever: Retriever) -> HallucinationWorkflow:
    """
    Construct the process-wide ``HallucinationWorkflow`` singleton.

    Cached via ``lru_cache`` so the workflow's LangGraph ``StateGraph``
    is built and compiled, and every underlying agent (and their own
    expensive collaborators, e.g. the spaCy pipelines behind
    ``ResponseAnalyzer``) constructed, exactly once per worker process.

    Args:
        retriever: The shared Person 2 evidence retriever to wire into
            both response analysis and self-correction's supplementary
            evidence lookups.

    Returns:
        The cached ``HallucinationWorkflow`` instance.
    """
    logger.info("Initializing HallucinationWorkflow singleton for API layer.")
    return HallucinationWorkflow(retriever=retriever)


def get_hallucination_workflow(
    retriever: Retriever = Depends(get_retriever),
) -> HallucinationWorkflow:
    """FastAPI dependency that resolves the shared ``HallucinationWorkflow`` instance."""
    return _build_hallucination_workflow(retriever)


@router.post(
    "/analyze",
    response_model=WorkflowAnalyzeResponse,
    status_code=status.HTTP_200_OK,
    summary="Run the full multi-agent hallucination-detection-and-correction workflow",
    response_description=(
        "The complete workflow result: original response, final verdict, hallucination and "
        "confidence scores, verification/evidence results, knowledge graph validation, drift "
        "report, risk assessment, the final corrected answer, and execution metadata."
    ),
)
def analyze_with_workflow(
    payload: WorkflowAnalyzeRequest,
    workflow: HallucinationWorkflow = Depends(get_hallucination_workflow),
) -> WorkflowAnalyzeResponse:
    """
    Run the full Person 3 multi-agent workflow (claim extraction and
    verification reused from Person 1/Person 2, followed by knowledge
    graph validation, knowledge drift detection, risk analysis, and
    self-correction) on a single LLM-generated response.

    Args:
        payload: Request body containing the response text to analyze
            and optional supplementary-evidence scoping (``document_id``, ``top_k``).
        workflow: Injected, process-shared ``HallucinationWorkflow``.

    Returns:
        The complete workflow result, with the final corrected answer
        exposed directly via ``corrected_answer``.

    Raises:
        HTTPException: ``422`` if the input cannot be analyzed (e.g. it
            is empty after validation); ``500`` for unexpected internal failures.
    """
    try:
        initial_state = create_initial_state(
            payload.response_text, document_id=payload.document_id, top_k=payload.top_k
        )
    except ValueError as exc:
        logger.warning("Workflow request rejected: %s", exc)
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc

    try:
        # Invoke the already-compiled graph directly (rather than
        # workflow.run(), which only returns the WorkflowResult) so the
        # full final WorkflowState -- including hallucination_score,
        # confidence_score, and execution_metadata, none of which live
        # on WorkflowResult itself -- is available to build the API
        # response from a single graph execution. See this module's
        # docstring for the full rationale.
        final_state = workflow._compiled_graph.invoke(initial_state)  # noqa: SLF001
    except AgentWorkflowError as exc:
        logger.warning("Workflow execution rejected: %s", exc)
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - guard against unhandled pipeline failures
        logger.exception("Unexpected failure while running the multi-agent workflow.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while running the workflow.",
        ) from exc

    result = final_state.get("workflow_result")
    if result is None:
        logger.error("Workflow completed without producing a WorkflowResult.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The workflow completed without producing a result.",
        )

    return WorkflowAnalyzeResponse.from_workflow_result(
        result,
        hallucination_score=final_state.get("hallucination_score"),
        confidence_score=final_state.get("confidence_score"),
        execution_metadata=_sanitize_execution_metadata(final_state.get("execution_metadata")),
    )


def _sanitize_execution_metadata(execution_metadata: dict | None) -> dict:
    """
    Strip non-JSON-serializable internal objects out of
    ``WorkflowState['execution_metadata']`` before it is returned in the
    API response.

    The ``self_correct`` node stashes the full, internal
    ``CorrectionResult`` dataclass instance under the
    ``"correction_result"`` key (see ``app/agents/workflow.py``) so
    ``_finalize_node`` can attach it to ``WorkflowResult.correction``.
    That raw object is not JSON-serializable and is redundant in the API
    response anyway, since it is already fully exposed via
    ``WorkflowAnalyzeResponse.correction``. Every other key populated by
    the workflow's nodes (step names, backend identifiers, verdict
    strings, booleans, counts) is already a plain, JSON-safe value.

    Args:
        execution_metadata: The raw execution metadata from the final
            ``WorkflowState``, or ``None``.

    Returns:
        A shallow copy of ``execution_metadata`` with ``"correction_result"`` removed.
    """
    if not execution_metadata:
        return {}
    return {key: value for key, value in execution_metadata.items() if key != "correction_result"}
