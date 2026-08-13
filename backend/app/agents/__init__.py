"""
Agents package (Person 3).

LangGraph multi-agent workflow, self-correction, knowledge graph
validation, knowledge drift detection, and risk analysis, built on top
of Person 1's/Person 2's existing claim extraction, hallucination
detection, evidence retrieval, and fact verification.

Public API
-----------
* :class:`~app.agents.workflow.HallucinationWorkflow` -- the compiled
  LangGraph ``StateGraph`` orchestrating the full pipeline.
* :func:`~app.agents.workflow.get_workflow` -- process-wide cached
  default-configured workflow instance.
* :func:`~app.agents.workflow.get_final_response` -- extracts the
  single final response text from a completed
  :class:`~app.models.workflow_models.WorkflowResult`.
* :class:`~app.agents.state.WorkflowState`,
  :class:`~app.agents.state.WorkflowStep`,
  :func:`~app.agents.state.create_initial_state` -- the LangGraph state
  schema threaded through every node.
* :class:`~app.agents.self_correction_agent.SelfCorrectionAgent`,
  :class:`~app.agents.knowledge_graph_agent.KnowledgeGraphAgent`,
  :class:`~app.agents.drift_detection_agent.DriftDetectionAgent`,
  :class:`~app.agents.risk_analysis_agent.RiskAnalysisAgent` -- the four
  individual Person 3 agents, usable standalone outside the graph as well.
* Every exception in :mod:`app.agents.exceptions`.
"""

from __future__ import annotations

from app.agents.drift_detection_agent import (
    DriftBaselineStore,
    DriftDetectionAgent,
    HistoricalClaimRecord,
    InMemoryDriftBaselineStore,
)
from app.agents.exceptions import (
    AgentWorkflowError,
    DriftDetectionError,
    InvalidWorkflowStateError,
    KnowledgeGraphValidationError,
    RiskAnalysisError,
    SelfCorrectionError,
    WorkflowExecutionError,
)
from app.agents.knowledge_graph_agent import KnowledgeGraphAgent
from app.agents.risk_analysis_agent import RiskAnalysisAgent
from app.agents.self_correction_agent import SelfCorrectionAgent
from app.agents.state import (
    WorkflowState,
    WorkflowStep,
    create_initial_state,
    merge_execution_metadata,
    record_error,
    record_step,
)
from app.agents.workflow import HallucinationWorkflow, get_final_response, get_workflow

__all__ = [
    # Exceptions
    "AgentWorkflowError",
    "WorkflowExecutionError",
    "InvalidWorkflowStateError",
    "SelfCorrectionError",
    "KnowledgeGraphValidationError",
    "DriftDetectionError",
    "RiskAnalysisError",
    # State
    "WorkflowState",
    "WorkflowStep",
    "create_initial_state",
    "record_step",
    "record_error",
    "merge_execution_metadata",
    # Agents
    "SelfCorrectionAgent",
    "KnowledgeGraphAgent",
    "DriftDetectionAgent",
    "DriftBaselineStore",
    "InMemoryDriftBaselineStore",
    "HistoricalClaimRecord",
    "RiskAnalysisAgent",
    # Workflow
    "HallucinationWorkflow",
    "get_workflow",
    "get_final_response",
]
