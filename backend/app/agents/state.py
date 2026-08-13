"""
LangGraph Workflow State
============================

Defines the shared state schema threaded through every node of the
Person 3 LangGraph multi-agent workflow (built in ``app.agents.workflow``,
a later batch): claim extraction and verification reused from Person 1
and Person 2, followed by knowledge-graph validation, knowledge drift
detection, risk analysis, and self-correction.

Design notes
------------
* Modeled as a ``TypedDict`` (LangGraph's standard state representation
  for ``StateGraph``), not a dataclass or Pydantic model. LangGraph
  merges each node's partial return value into this structure by key on
  every superstep, using each key's declared *reducer* -- ``operator.add``
  for ``completed_steps``/``errors`` below, a custom shallow-merge
  reducer for ``execution_metadata``, and "last write wins" for
  everything else. This merge-by-key mechanism is what ``TypedDict`` +
  ``Annotated`` integrates with directly; wrapping the state in a
  dataclass/Pydantic model would require LangGraph-specific adapters and
  is not the library's idiomatic pattern.
* ``total=False`` allows nodes early in the graph to run before every
  key has been populated (e.g. ``knowledge_graph_validation`` does not
  exist yet before the KG-validation node has run).
* The values stored under each key are otherwise the same well-typed
  domain objects used throughout the rest of Person 3's code
  (``app.models.workflow_models``) or reused directly from Person 1 /
  Person 2 (``ClaimModel``, ``VerificationResult``), so node functions
  never need to fight the state schema or re-parse plain dicts.
"""

from __future__ import annotations

import operator
from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Any, TypedDict

from app.models.workflow_models import DriftReport, KGValidationResult, RiskAssessment, WorkflowResult

__all__ = [
    "WorkflowStep",
    "WorkflowState",
    "create_initial_state",
    "record_step",
    "record_error",
    "merge_execution_metadata",
]


class WorkflowStep(str, Enum):
    """
    Canonical names for each node in the LangGraph multi-agent workflow.

    Used both as LangGraph node identifiers (see ``app.agents.workflow``)
    and as the values recorded in ``WorkflowState['execution_metadata']
    ['completed_steps']`` / ``WorkflowResult.completed_steps``, so the
    two always stay in sync.
    """

    EXTRACT_CLAIMS = "extract_claims"
    RETRIEVE_EVIDENCE = "retrieve_evidence"
    VERIFY_CLAIMS = "verify_claims"
    SCORE_CONFIDENCE = "score_confidence"
    VALIDATE_KNOWLEDGE_GRAPH = "validate_knowledge_graph"
    DETECT_DRIFT = "detect_drift"
    ANALYZE_RISK = "analyze_risk"
    SELF_CORRECT = "self_correct"
    FINALIZE = "finalize"


def merge_execution_metadata(current: dict[str, Any] | None, update: dict[str, Any] | None) -> dict[str, Any]:
    """
    LangGraph reducer for the ``execution_metadata`` state channel.

    Plain "last write wins" semantics (LangGraph's default for an
    un-annotated ``dict`` field) would let one node's metadata update
    silently clobber every other node's, since each node only returns
    the keys it itself cares about. This reducer instead performs a
    shallow merge of ``update`` into ``current``, with special handling
    for the two list-valued bookkeeping keys (``completed_steps`` and
    ``errors``), which are *concatenated* rather than overwritten so
    that every node's contribution accumulates across the whole run --
    matching the accumulation semantics already used for the top-level
    ``completed_steps`` / ``errors`` channels below.

    Args:
        current: The state's existing ``execution_metadata`` value
            (``None`` before any node has written to it).
        update: The partial metadata dict returned by the node that just ran.

    Returns:
        A new, merged metadata dictionary. Never mutates ``current`` or
        ``update`` in place.
    """
    merged: dict[str, Any] = dict(current or {})
    for key, value in (update or {}).items():
        if key in ("completed_steps", "errors") and isinstance(value, list):
            merged[key] = list(merged.get(key, [])) + list(value)
        else:
            merged[key] = value
    return merged


class WorkflowState(TypedDict, total=False):
    """
    Shared state schema threaded through every node of the LangGraph
    multi-agent workflow (see ``app.agents.workflow.build_workflow_graph``).

    Attributes:
        original_response: The raw LLM-generated response text being analyzed. Input.
        document_id: Optional Person 2 document filter restricting
            evidence retrieval to a single ingested document. Input.
        top_k: Optional override for the number of evidence passages
            retrieved per claim. Input.
        extracted_claims: Claims extracted from ``original_response`` by
            Person 1's ``ClaimExtractor`` (``ClaimModel`` instances).
        verification_results: Per-claim verification outcomes from
            Person 1's ``HallucinationDetector`` / Person 2's
            ``FactVerifier`` (``VerificationResult`` instances).
        hallucination_score: Aggregate hallucination probability for the
            whole response, in [0, 1], from Person 1's ``ConfidenceScorer``.
        confidence_score: Aggregate trust/confidence score for the whole
            response, in [0, 1], from Person 1's ``ConfidenceScorer``.
        knowledge_graph_validation: Output of the knowledge-graph-validation agent.
        drift_report: Output of the knowledge drift detection agent.
        risk_assessment: Output of the risk analysis module.
        corrected_response: Output text of the self-correction agent
            (equal to ``original_response`` if no correction was needed).
        workflow_result: The final aggregated ``WorkflowResult``, set by
            the terminal ``finalize`` node.
        completed_steps: Names of every node that has completed
            successfully so far, in execution order. Accumulates across
            nodes via the ``operator.add`` reducer (safe under LangGraph
            fan-out/parallel node execution).
        errors: Human-readable descriptions of node-level failures that
            occurred but did not abort the run. Accumulates the same way
            as ``completed_steps``.
        execution_metadata: Free-form bookkeeping populated by
            individual nodes (e.g. which KG backend was used, per-node
            timings, retrieved-evidence counts). Merged across nodes via
            :func:`merge_execution_metadata` rather than overwritten.
        started_at: ISO-8601 UTC timestamp of when the workflow run began.
    """

    # --- Input ---------------------------------------------------------- #
    original_response: str
    document_id: str | None
    top_k: int | None

    # --- Person 1 / Person 2 pipeline outputs, reused as-is -------------- #
    extracted_claims: list[Any]
    verification_results: list[Any]
    hallucination_score: float | None
    confidence_score: float | None

    # --- Person 3 agent outputs ------------------------------------------ #
    knowledge_graph_validation: KGValidationResult | None
    drift_report: DriftReport | None
    risk_assessment: RiskAssessment | None
    corrected_response: str | None

    # --- Final aggregated output ------------------------------------------#
    workflow_result: WorkflowResult | None

    # --- Observability / control flow (accumulate across nodes) --------- #
    completed_steps: Annotated[list[str], operator.add]
    errors: Annotated[list[str], operator.add]
    execution_metadata: Annotated[dict[str, Any], merge_execution_metadata]

    # --- Bookkeeping ------------------------------------------------------#
    started_at: str


def create_initial_state(
    original_response: str,
    document_id: str | None = None,
    top_k: int | None = None,
) -> WorkflowState:
    """
    Build the initial ``WorkflowState`` passed into
    ``CompiledGraph.invoke()`` / ``.stream()`` to start a new workflow run.

    Args:
        original_response: The raw LLM-generated response text to analyze.
        document_id: Optional Person 2 document filter restricting
            evidence retrieval to a single ingested document.
        top_k: Optional override for the number of evidence passages
            retrieved per claim.

    Returns:
        A fully initialized ``WorkflowState`` with every accumulator
        field present as an empty list/dict (so their reducers have a
        valid starting value to merge into) and every other field either
        set from the arguments or explicitly ``None``.

    Raises:
        ValueError: If ``original_response`` is empty or whitespace-only.
    """
    if original_response is None or not original_response.strip():
        raise ValueError("original_response must not be empty.")

    return WorkflowState(
        original_response=original_response,
        document_id=document_id,
        top_k=top_k,
        extracted_claims=[],
        verification_results=[],
        hallucination_score=None,
        confidence_score=None,
        knowledge_graph_validation=None,
        drift_report=None,
        risk_assessment=None,
        corrected_response=None,
        workflow_result=None,
        completed_steps=[],
        errors=[],
        execution_metadata={},
        started_at=datetime.now(timezone.utc).isoformat(),
    )


def record_step(step: WorkflowStep, **metadata: Any) -> dict[str, Any]:
    """
    Build a partial state update marking ``step`` as completed.

    Node functions in ``app.agents.workflow`` return
    ``{**record_step(WorkflowStep.X, ...), other_key: value}`` so the
    ``completed_steps`` accumulator and ``execution_metadata`` stay in
    sync without every node duplicating the same dict-literal boilerplate.

    Args:
        step: The workflow step that just completed successfully.
        **metadata: Additional free-form key/value pairs to merge into
            ``execution_metadata`` alongside the step-completion record
            (e.g. ``record_step(WorkflowStep.VALIDATE_KNOWLEDGE_GRAPH, kg_backend="neo4j")``).

    Returns:
        A partial ``WorkflowState`` update.
    """
    update: dict[str, Any] = {"completed_steps": [step.value]}
    if metadata:
        update["execution_metadata"] = {"completed_steps": [step.value], **metadata}
    else:
        update["execution_metadata"] = {"completed_steps": [step.value]}
    return update


def record_error(step: WorkflowStep, error: Exception) -> dict[str, Any]:
    """
    Build a partial state update recording a node-level failure without
    halting graph execution.

    Args:
        step: The workflow step that failed.
        error: The exception that was raised.

    Returns:
        A partial ``WorkflowState`` update.
    """
    message = f"{step.value}: {error}"
    return {
        "errors": [message],
        "execution_metadata": {"errors": [message]},
    }
