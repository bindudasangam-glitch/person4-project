"""
Multi-Agent LangGraph Workflow
==================================

Wires Person 1's/Person 2's existing analysis pipeline together with
every Person 3 agent (Batches 1-4) into a single, compiled LangGraph
``StateGraph``, threading :class:`~app.agents.state.WorkflowState`
through each node and producing a final
:class:`~app.models.workflow_models.WorkflowResult`.

Graph topology
---------------
::

    START -> analyze_response --+-- (no claims) --------------> finalize -> END
                                 |
                                 +-- (has claims) --> validate_knowledge_graph --+
                                 |                                              |
                                 +-- (has claims) --> detect_drift -------------+--> analyze_risk -> self_correct -> finalize -> END

* ``analyze_response`` reuses Person 1's/Person 2's existing, unmodified
  :class:`~app.services.response_analyzer.ResponseAnalyzer` (claim
  extraction -> hallucination detection -> confidence scoring) as a
  single node, so Person 1's/Person 2's outputs (claims, detection
  outcomes, confidence scores) are preserved and carried through
  ``WorkflowState`` exactly as produced, rather than being
  re-derived or replaced by Person 3 logic.
* A conditional edge routes straight to ``finalize`` when no verifiable
  claims were found (or response analysis itself failed), since there is
  nothing for the knowledge-graph, drift, risk, or correction stages to
  operate on. This is LangGraph's documented mechanism for static,
  data-dependent branching via a routing function returning a list of
  destination node names.
* ``validate_knowledge_graph`` and ``detect_drift`` run as a genuine
  parallel fan-out (both depend only on ``analyze_response``'s output,
  not on each other) and fan back in at ``analyze_risk``, which needs
  both of their results. LangGraph's superstep execution model waits for
  every incoming edge into a node to have fired before running it, so
  ``analyze_risk`` is guaranteed to see both results without any manual
  synchronization here.
* ``self_correct`` always runs whenever claims exist, so the corrected
  answer is genuinely produced by executing the compiled graph, not by
  calling ``SelfCorrectionAgent`` separately outside of it.
* ``finalize`` never raises: even if assembling the final
  ``WorkflowResult`` itself fails unexpectedly, it falls back to a
  minimal, valid result rather than losing the run entirely, so
  :meth:`HallucinationWorkflow.run` always returns a
  :class:`~app.models.workflow_models.WorkflowResult`.

Resilience
-----------
Every Person 3 stage (KG validation, drift detection, risk analysis,
self-correction) degrades gracefully on failure: the corresponding
domain exception from :mod:`app.agents.exceptions` is caught, logged,
and recorded in ``WorkflowState['errors']`` /
``WorkflowState['execution_metadata']`` (and therefore
``WorkflowResult.errors``), and that stage's output is left ``None``
rather than aborting the whole run. Downstream agents (in particular
``RiskAnalysisAgent``) are already designed to treat a missing upstream
signal as "no information" rather than "problem detected". Only a
failure in ``analyze_response`` that is *not* a recognized
``ResponseAnalysisError`` (i.e. a genuinely unexpected bug) is allowed
to abort the run, since without claims there is no meaningful partial
result Person 3's agents could still produce.

Dependency injection
----------------------
Every collaborator -- the ``ResponseAnalyzer``, the Person 2
``Retriever``, and all four Person 3 agents -- is accepted as an
explicit constructor argument. Sensible defaults are constructed when a
given argument is omitted (mirroring the pattern already used by every
Person 1/2/3 service in this codebase, e.g. ``ResponseAnalyzer`` itself
defaulting its own ``hallucination_detector``), but nothing is reached
for implicitly via a hidden module-level global inside this class --
the process-wide cached singleton exposed via :func:`get_workflow` is a
separate, opt-in convenience for callers that don't need custom wiring.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Any

from langgraph.graph import END, START, StateGraph

from app.agents.drift_detection_agent import DriftDetectionAgent
from app.agents.exceptions import (
    DriftDetectionError,
    KnowledgeGraphValidationError,
    RiskAnalysisError,
    SelfCorrectionError,
    WorkflowExecutionError,
)
from app.agents.knowledge_graph_agent import KnowledgeGraphAgent
from app.agents.risk_analysis_agent import RiskAnalysisAgent
from app.agents.self_correction_agent import SelfCorrectionAgent
from app.agents.state import WorkflowState, WorkflowStep, create_initial_state, record_error, record_step
from app.core.logging import logger
from app.models.claim_model import VerificationStatus
from app.models.workflow_models import SeverityLevel, WorkflowModelValidationError, WorkflowResult
from app.services.response_analyzer import ResponseAnalysisError, ResponseAnalyzer

if TYPE_CHECKING:
    from app.models.claim_model import ClaimModel
    from app.models.evidence import EvidenceBundle
    from app.retrieval.retriever import Retriever

__all__ = ["HallucinationWorkflow", "get_workflow", "get_final_response"]


def _record_steps(steps: list[WorkflowStep], **metadata: Any) -> dict[str, Any]:
    """
    Build a partial ``WorkflowState`` update marking every step in
    ``steps`` as completed in one go, mirroring
    :func:`app.agents.state.record_step`'s shape for the case of a
    single node logically completing several ``WorkflowStep`` stages at
    once -- e.g. ``analyze_response`` wraps Person 1's/Person 2's entire
    extract -> retrieve -> verify -> score pipeline in one call.

    Args:
        steps: The workflow steps that just completed successfully.
        **metadata: Additional free-form key/value pairs merged into
            ``execution_metadata`` alongside the step-completion record.

    Returns:
        A partial ``WorkflowState`` update.
    """
    step_values = [step.value for step in steps]
    return {
        "completed_steps": step_values,
        "execution_metadata": {"completed_steps": step_values, **metadata},
    }


def get_final_response(result: WorkflowResult) -> str:
    """
    Return the single, final response text a caller should actually show
    or act on for a completed workflow run.

    ``WorkflowResult`` itself has no top-level "corrected response"
    field (that value lives on the nested
    ``correction.corrected_response`` when the self-correction stage
    ran), so this helper centralizes the one-line rule every caller
    otherwise needs to repeat: use the corrected response when
    correction ran, otherwise the response was never modified, so the
    original text already *is* the final answer.

    Args:
        result: A completed workflow run's result.

    Returns:
        The corrected response text if self-correction ran, otherwise
        ``result.response_text`` unchanged.
    """
    if result.correction is not None:
        return result.correction.corrected_response
    return result.response_text


class HallucinationWorkflow:
    """
    Compiled LangGraph multi-agent workflow combining Person 1's/Person
    2's analysis pipeline with every Person 3 agent.

    Args:
        response_analyzer: The Person 1/2 orchestrator used for claim
            extraction, hallucination detection, and confidence scoring.
            If provided, used as-is (``retriever`` below then has no
            effect on its construction). Defaults to a new
            ``ResponseAnalyzer(retriever=retriever)``.
        retriever: Optional Person 2 evidence retriever. When provided
            *and* ``response_analyzer`` is omitted, the default
            ``ResponseAnalyzer`` is constructed with it, so real,
            embedding-based evidence from ChromaDB backs hallucination
            detection instead of the built-in empty in-memory corpus.
            It is also used directly by this workflow's self-correction
            stage to fetch supplementary evidence for claims that need
            replacing (see :meth:`_gather_supplementary_evidence`).
        knowledge_graph_agent: Validates claim entities/relationships
            against the knowledge graph (Batch 3). Defaults to a new
            ``KnowledgeGraphAgent()``, which itself uses the cached
            Batch 1 ``KnowledgeGraphService`` singleton (Neo4j with
            automatic NetworkX fallback).
        drift_detection_agent: Detects knowledge drift against a
            historical baseline (Batch 4). Defaults to a new
            ``DriftDetectionAgent()`` (in-memory baseline store).
        risk_analysis_agent: Combines every signal into a final risk
            assessment (Batch 4). Defaults to a new ``RiskAnalysisAgent()``.
        self_correction_agent: Rewrites unsupported/contradicted claims
            (Batch 3). Defaults to a new ``SelfCorrectionAgent()``.
        checkpointer: Optional LangGraph checkpointer, forwarded directly
            to ``StateGraph.compile()``. ``None`` (the default) compiles
            without persistence/streaming support -- passing a real
            checkpointer here is how a future API layer can add
            multi-turn or resumable workflow support without any change
            to this class.

    All dependencies default to their standard production
    implementations if omitted, but every one of them can be substituted
    via constructor injection (e.g. with test doubles, or agents tuned
    with different thresholds) -- this class never reaches for a
    collaborator through a hidden global inside its own methods.
    """

    def __init__(
        self,
        response_analyzer: ResponseAnalyzer | None = None,
        retriever: "Retriever | None" = None,
        knowledge_graph_agent: KnowledgeGraphAgent | None = None,
        drift_detection_agent: DriftDetectionAgent | None = None,
        risk_analysis_agent: RiskAnalysisAgent | None = None,
        self_correction_agent: SelfCorrectionAgent | None = None,
        checkpointer: Any | None = None,
    ) -> None:
        self._retriever = retriever
        self._response_analyzer = response_analyzer or ResponseAnalyzer(retriever=retriever)
        self._kg_agent = knowledge_graph_agent or KnowledgeGraphAgent()
        self._drift_agent = drift_detection_agent or DriftDetectionAgent()
        self._risk_agent = risk_analysis_agent or RiskAnalysisAgent()
        self._correction_agent = self_correction_agent or SelfCorrectionAgent()

        graph = self._build_graph()
        self._compiled_graph = graph.compile(checkpointer=checkpointer)

        logger.info(
            "HallucinationWorkflow compiled (retriever_configured=%s).", self._retriever is not None
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def run(
        self,
        response_text: str,
        document_id: str | None = None,
        top_k: int | None = None,
    ) -> WorkflowResult:
        """
        Run the full multi-agent workflow synchronously on a single response.

        Args:
            response_text: The raw LLM-generated response to analyze.
            document_id: Optional Person 2 document filter applied to
                this workflow's own supplementary evidence lookups
                during self-correction (see
                :meth:`_gather_supplementary_evidence`). Has no effect if
                no ``retriever`` was configured.
            top_k: Optional override for how many supplementary evidence
                passages to retrieve per claim needing correction.

        Returns:
            The completed :class:`~app.models.workflow_models.WorkflowResult`.
            Use :func:`get_final_response` to get the single final
            response text to show or act on.

        Raises:
            ValueError: If ``response_text`` is empty or whitespace-only.
            WorkflowExecutionError: If the compiled graph fails to
                execute, or completes without producing a ``WorkflowResult``.
        """
        initial_state = create_initial_state(response_text, document_id=document_id, top_k=top_k)

        try:
            final_state = self._compiled_graph.invoke(initial_state)
        except Exception as exc:  # noqa: BLE001 - normalize to domain error
            logger.exception("HallucinationWorkflow execution failed.")
            raise WorkflowExecutionError("The multi-agent workflow failed to complete.") from exc

        result = final_state.get("workflow_result")
        if result is None:
            raise WorkflowExecutionError(
                "The workflow completed without producing a WorkflowResult.",
                failed_step=WorkflowStep.FINALIZE.value,
            )
        return result

    async def arun(
        self,
        response_text: str,
        document_id: str | None = None,
        top_k: int | None = None,
    ) -> WorkflowResult:
        """
        Async counterpart of :meth:`run`, using the compiled graph's
        ``ainvoke``. Exists so a future async FastAPI route can await
        the workflow directly instead of running it in a thread pool.

        Args, Returns, and Raises are identical to :meth:`run`.
        """
        initial_state = create_initial_state(response_text, document_id=document_id, top_k=top_k)

        try:
            final_state = await self._compiled_graph.ainvoke(initial_state)
        except Exception as exc:  # noqa: BLE001 - normalize to domain error
            logger.exception("HallucinationWorkflow async execution failed.")
            raise WorkflowExecutionError("The multi-agent workflow failed to complete.") from exc

        result = final_state.get("workflow_result")
        if result is None:
            raise WorkflowExecutionError(
                "The workflow completed without producing a WorkflowResult.",
                failed_step=WorkflowStep.FINALIZE.value,
            )
        return result

    # ------------------------------------------------------------------ #
    # Graph construction
    # ------------------------------------------------------------------ #
    def _build_graph(self) -> StateGraph:
        """Assemble (but do not compile) the LangGraph ``StateGraph`` described in this module's docstring."""
        graph: StateGraph = StateGraph(WorkflowState)

        graph.add_node("analyze_response", self._analyze_response_node)
        graph.add_node("validate_knowledge_graph", self._validate_knowledge_graph_node)
        graph.add_node("detect_drift", self._detect_drift_node)
        graph.add_node("analyze_risk", self._analyze_risk_node)
        graph.add_node("self_correct", self._self_correct_node)
        graph.add_node("finalize", self._finalize_node)

        graph.add_edge(START, "analyze_response")
        graph.add_conditional_edges("analyze_response", self._route_after_analysis)
        graph.add_edge("validate_knowledge_graph", "analyze_risk")
        graph.add_edge("detect_drift", "analyze_risk")
        graph.add_edge("analyze_risk", "self_correct")
        graph.add_edge("self_correct", "finalize")
        graph.add_edge("finalize", END)

        return graph

    def _route_after_analysis(self, state: WorkflowState) -> list[str]:
        """
        Conditional routing after ``analyze_response``: fan out to the
        knowledge-graph-validation and drift-detection nodes in parallel
        when claims were extracted, otherwise skip straight to ``finalize``.
        """
        if state.get("extracted_claims"):
            return ["validate_knowledge_graph", "detect_drift"]
        return ["finalize"]

    # ------------------------------------------------------------------ #
    # Nodes
    # ------------------------------------------------------------------ #
    def _analyze_response_node(self, state: WorkflowState) -> dict[str, Any]:
        """
        Run Person 1's/Person 2's existing claim extraction ->
        hallucination detection -> confidence scoring pipeline via
        ``ResponseAnalyzer``, and populate the corresponding
        ``WorkflowState`` fields (``extracted_claims``,
        ``verification_results``, ``hallucination_score``,
        ``confidence_score``) from its result, unchanged.
        """
        response_text = state["original_response"]

        try:
            analysis = self._response_analyzer.analyze(response_text)
        except ResponseAnalysisError as exc:
            logger.exception("Workflow: response analysis failed; proceeding with no claims.")
            return {
                "extracted_claims": [],
                "verification_results": [],
                "hallucination_score": None,
                "confidence_score": None,
                **record_error(WorkflowStep.VERIFY_CLAIMS, exc),
            }
        except Exception as exc:  # noqa: BLE001 - genuinely unexpected: abort, nothing usable remains
            logger.exception("Workflow: response analysis failed unexpectedly.")
            raise WorkflowExecutionError(
                "Response analysis stage failed unexpectedly.",
                failed_step=WorkflowStep.VERIFY_CLAIMS.value,
            ) from exc

        claims = list(analysis.claims)
        outcomes = list(analysis.detection_outcomes)
        hallucination_score = analysis.confidence.hallucination_probability if analysis.confidence else None
        confidence_score = analysis.confidence.confidence_score if analysis.confidence else None

        completed = [WorkflowStep.EXTRACT_CLAIMS]
        if self._retriever is not None:
            completed.append(WorkflowStep.RETRIEVE_EVIDENCE)
        completed.extend([WorkflowStep.VERIFY_CLAIMS, WorkflowStep.SCORE_CONFIDENCE])

        return {
            "extracted_claims": claims,
            "verification_results": outcomes,
            "hallucination_score": hallucination_score,
            "confidence_score": confidence_score,
            **_record_steps(
                completed,
                person1_verdict=analysis.verdict.value,
                person1_verdict_reason=analysis.verdict_reason,
                total_claims=len(claims),
            ),
        }

    def _validate_knowledge_graph_node(self, state: WorkflowState) -> dict[str, Any]:
        """Run the knowledge-graph-validation agent (Batch 3) over the extracted claims."""
        claims = state.get("extracted_claims") or []

        try:
            result = self._kg_agent.validate(claims)
        except (KnowledgeGraphValidationError, Exception) as exc:  # noqa: BLE001 - degrade gracefully
            logger.exception("Workflow: knowledge graph validation failed; continuing without a KG signal.")
            return {
                "knowledge_graph_validation": None,
                **record_error(WorkflowStep.VALIDATE_KNOWLEDGE_GRAPH, exc),
            }

        return {
            "knowledge_graph_validation": result,
            **record_step(WorkflowStep.VALIDATE_KNOWLEDGE_GRAPH, kg_backend=result.backend_used),
        }

    def _detect_drift_node(self, state: WorkflowState) -> dict[str, Any]:
        """Run the knowledge drift detection agent (Batch 4) over the extracted claims."""
        claims = state.get("extracted_claims") or []
        outcomes = state.get("verification_results") or []

        try:
            report = self._drift_agent.detect_drift(claims, outcomes)
        except (DriftDetectionError, Exception) as exc:  # noqa: BLE001 - degrade gracefully
            logger.exception("Workflow: drift detection failed; continuing without a drift signal.")
            return {
                "drift_report": None,
                **record_error(WorkflowStep.DETECT_DRIFT, exc),
            }

        return {
            "drift_report": report,
            **record_step(WorkflowStep.DETECT_DRIFT, has_drift=report.has_drift),
        }

    def _analyze_risk_node(self, state: WorkflowState) -> dict[str, Any]:
        """
        Run the risk analysis module (Batch 4), combining Person 1's
        hallucination/confidence scores with the (possibly ``None``, on
        degraded upstream stages) KG validation and drift report.
        """
        hallucination_score = state.get("hallucination_score")
        confidence_score = state.get("confidence_score")

        if hallucination_score is None or confidence_score is None:
            missing = RiskAnalysisError(
                "Cannot analyze risk without hallucination_score/confidence_score "
                "(response analysis did not produce scores for this run)."
            )
            logger.warning("Workflow: skipping risk analysis; %s", missing)
            return {
                "risk_assessment": None,
                **record_error(WorkflowStep.ANALYZE_RISK, missing),
            }

        try:
            assessment = self._risk_agent.analyze(
                hallucination_score=hallucination_score,
                confidence_score=confidence_score,
                kg_validation=state.get("knowledge_graph_validation"),
                drift_report=state.get("drift_report"),
            )
        except (RiskAnalysisError, Exception) as exc:  # noqa: BLE001 - degrade gracefully
            logger.exception("Workflow: risk analysis failed; continuing without a risk assessment.")
            return {
                "risk_assessment": None,
                **record_error(WorkflowStep.ANALYZE_RISK, exc),
            }

        return {
            "risk_assessment": assessment,
            **record_step(WorkflowStep.ANALYZE_RISK, risk_level=assessment.risk_level.value),
        }

    def _self_correct_node(self, state: WorkflowState) -> dict[str, Any]:
        """
        Run the self-correction agent (Batch 3) as part of the compiled
        graph itself, using Person 1's claims and detection outcomes plus
        any supplementary Person 2 evidence this workflow fetched (see
        :meth:`_gather_supplementary_evidence`), and populate
        ``WorkflowState['corrected_response']`` from its result.
        """
        response_text = state["original_response"]
        claims = state.get("extracted_claims") or []
        outcomes = state.get("verification_results") or []
        evidence_bundles = self._gather_supplementary_evidence(
            claims, state.get("document_id"), state.get("top_k")
        )

        try:
            result = self._correction_agent.correct(
                response_text, claims, outcomes, evidence_bundles=evidence_bundles
            )
        except (SelfCorrectionError, Exception) as exc:  # noqa: BLE001 - degrade gracefully
            logger.exception(
                "Workflow: self-correction failed; falling back to the original, uncorrected response."
            )
            return {
                "corrected_response": response_text,
                **record_error(WorkflowStep.SELF_CORRECT, exc),
            }

        return {
            "corrected_response": result.corrected_response,
            **record_step(
                WorkflowStep.SELF_CORRECT,
                correction_result=result,
                was_corrected=result.was_corrected,
            ),
        }

    def _finalize_node(self, state: WorkflowState) -> dict[str, Any]:
        """
        Assemble the final :class:`~app.models.workflow_models.WorkflowResult`
        from everything accumulated in ``WorkflowState`` over the run,
        carrying forward the original response, Person 1's/Person 2's
        hallucination/confidence scores and verification results, and
        every Person 3 agent's output. Never raises: falls back to a
        minimal, valid result if assembly itself fails unexpectedly, so
        a ``WorkflowResult`` is always produced.
        """
        claims = state.get("extracted_claims") or []
        metadata = state.get("execution_metadata") or {}
        correction_result = metadata.get("correction_result")
        risk_assessment = state.get("risk_assessment")

        final_verdict = metadata.get("person1_verdict", "unresolved")
        final_verdict_reason = metadata.get("person1_verdict_reason", "")

        if not claims:
            final_verdict = metadata.get("person1_verdict", "no_verifiable_claims")
            final_verdict_reason = metadata.get(
                "person1_verdict_reason",
                "No independently checkable factual claims were found in the response.",
            )
        elif risk_assessment is not None and risk_assessment.risk_level is SeverityLevel.CRITICAL:
            # Person 3's risk analysis found something serious enough to
            # override Person 1's own verdict label; Person 1's
            # verdict_reason is still folded into the risk assessment's
            # own explanation, so no information is lost.
            final_verdict = "critical_risk"
            final_verdict_reason = risk_assessment.explanation

        try:
            workflow_result = WorkflowResult(
                response_text=state["original_response"],
                final_verdict=final_verdict,
                final_verdict_reason=final_verdict_reason,
                total_claims=len(claims),
                correction=correction_result,
                kg_validation=state.get("knowledge_graph_validation"),
                drift_report=state.get("drift_report"),
                risk_assessment=risk_assessment,
                completed_steps=tuple(state.get("completed_steps") or ()),
                errors=tuple(state.get("errors") or ()),
            )
        except WorkflowModelValidationError as exc:
            logger.exception(
                "Workflow: failed to assemble the final WorkflowResult; falling back to a minimal result."
            )
            workflow_result = WorkflowResult(
                response_text=state["original_response"],
                final_verdict="internal_error",
                final_verdict_reason=f"Failed to assemble the final workflow result: {exc}",
                total_claims=len(claims),
                completed_steps=tuple(state.get("completed_steps") or ()),
                errors=tuple(list(state.get("errors") or ()) + [f"{WorkflowStep.FINALIZE.value}: {exc}"]),
            )

        logger.info(
            "HallucinationWorkflow run complete: verdict=%s, total_claims=%d, steps=%d, errors=%d.",
            workflow_result.final_verdict,
            workflow_result.total_claims,
            len(workflow_result.completed_steps),
            len(workflow_result.errors),
        )
        return {
            "workflow_result": workflow_result,
            **record_step(WorkflowStep.FINALIZE),
        }

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _gather_supplementary_evidence(
        self,
        claims: list["ClaimModel"],
        document_id: str | None,
        top_k: int | None,
    ) -> dict[int, "EvidenceBundle"]:
        """
        Fetch fresh Person 2 evidence for claims the self-correction
        agent will need to replace (i.e. ``CONTRADICTED`` claims only --
        every other status either needs no replacement text or must not
        invent one), using this workflow's own ``retriever`` directly.

        This is the one place ``document_id``/``top_k`` from
        ``WorkflowState`` are actually consumed: Person 1's/Person 2's
        existing ``ResponseAnalyzer``/``HallucinationDetector`` pipeline
        (reused unmodified in ``_analyze_response_node``) does not expose
        either as call-time parameters, so per-request evidence scoping
        is applied here instead, specifically for correction.

        Returns an empty dict (never raises) if no retriever is
        configured, or if retrieval fails for a given claim -- in either
        case, ``SelfCorrectionAgent`` transparently falls back to Person
        1's already-embedded evidence for that claim.
        """
        if self._retriever is None:
            return {}

        effective_top_k = top_k if top_k is not None else 3
        bundles: dict[int, "EvidenceBundle"] = {}

        for claim in claims:
            if claim.verification_status is not VerificationStatus.CONTRADICTED:
                continue
            try:
                bundles[claim.id] = self._retriever.retrieve(
                    query=claim.text, top_k=effective_top_k, document_id=document_id
                )
            except Exception:  # noqa: BLE001 - supplementary only, never fatal
                logger.exception(
                    "Workflow: supplementary evidence retrieval failed for claim %d; "
                    "self-correction will fall back to Person 1's embedded evidence.",
                    claim.id,
                )
                continue

        return bundles


@lru_cache(maxsize=1)
def get_workflow() -> HallucinationWorkflow:
    """
    Return the process-wide cached, default-configured
    :class:`HallucinationWorkflow` instance.

    Follows the same singleton pattern used throughout the codebase
    (``get_settings``, ``get_document_registry``,
    ``get_knowledge_graph_service``), so a future API route (e.g.
    ``POST /workflow``) can depend on this function directly without
    knowing anything about how the workflow or its agents are
    constructed -- satisfying "support future API integration without
    modification" without requiring any change here when that route is added.

    Callers that need custom dependency injection (a specific retriever,
    tuned agent thresholds, a checkpointer, etc.) should construct
    ``HallucinationWorkflow(...)`` directly instead of using this getter.

    Returns:
        The shared, default-configured ``HallucinationWorkflow`` instance for this process.
    """
    return HallucinationWorkflow()
