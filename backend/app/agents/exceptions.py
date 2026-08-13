"""
Agent Workflow Exception Hierarchy
=====================================

Defines every domain-specific exception raised by Person 3's module
(LangGraph multi-agent orchestration, self-correction, knowledge graph
validation, knowledge drift detection, and risk analysis). All
exceptions share a single common base, :class:`AgentWorkflowError`, so
the workflow API route (added in a later batch) can catch one type to
translate any expected domain failure into a clean HTTP error response,
while truly unexpected exceptions still propagate to the application's
generic 500 handler.

This module is intentionally independent of Person 1's exception
hierarchy (defined per-service under ``app.services.*``) and Person 2's
(``app.core.exceptions.RAGFactVerificationError`` and its subclasses)
-- the three error domains do not need to share a base class, since each
is caught and handled by its own API router. This mirrors the rationale
already documented in ``app/core/exceptions.py``'s own module docstring
for why Person 2's hierarchy is independent of Person 1's.

Note on ``app.knowledge_graph`` exceptions
--------------------------------------------
:mod:`app.knowledge_graph.neo4j_client` defines its own low-level
``KnowledgeGraphError`` hierarchy (``Neo4jConnectionError`` /
``Neo4jQueryError``) for backend-selection and query-execution failures
inside the knowledge graph package itself. :class:`KnowledgeGraphValidationError`
here is a distinct, *higher-level* exception raised by the LangGraph
knowledge-graph-validation agent node that wraps
:class:`~app.knowledge_graph.kg_service.KnowledgeGraphService` -- it is
what workflow orchestration code actually catches and re-raises further
up (typically via ``raise KnowledgeGraphValidationError(...) from exc``
where ``exc`` is the lower-level ``KnowledgeGraphError``), keeping the
two exception hierarchies cleanly layered rather than merged.
"""

from __future__ import annotations

from typing import Any, Optional

__all__ = [
    "AgentWorkflowError",
    "WorkflowExecutionError",
    "InvalidWorkflowStateError",
    "SelfCorrectionError",
    "KnowledgeGraphValidationError",
    "DriftDetectionError",
    "RiskAnalysisError",
]


class AgentWorkflowError(Exception):
    """
    Base class for all Person 3 (multi-agent workflow) domain exceptions.

    Attributes:
        message: A human-readable description of the failure.
        details: Optional structured context about the failure (e.g. the
            offending claim id, agent/node name, or configuration value),
            useful for logging or API error payloads without needing to
            parse the message string.
    """

    def __init__(self, message: str, details: Optional[dict[str, Any]] = None) -> None:
        """
        Args:
            message: A human-readable description of the failure.
            details: Optional structured context about the failure.
        """
        super().__init__(message)
        self.message = message
        self.details: dict[str, Any] = details or {}

    def __str__(self) -> str:
        return self.message


class WorkflowExecutionError(AgentWorkflowError):
    """
    Raised when the end-to-end LangGraph multi-agent workflow fails to
    complete (e.g. an unrecoverable failure in a required node, or the
    compiled graph itself raises during ``invoke``/``stream``).
    """

    def __init__(self, message: str, failed_step: Optional[str] = None) -> None:
        """
        Args:
            message: A human-readable description of the failure.
            failed_step: Name of the ``WorkflowStep`` that was executing
                when the failure occurred, if known.
        """
        details = {"failed_step": failed_step} if failed_step else {}
        super().__init__(message, details=details)
        self.failed_step = failed_step


class InvalidWorkflowStateError(AgentWorkflowError):
    """
    Raised when a LangGraph node is invoked with a workflow state that is
    missing data it structurally requires to run (e.g. the self-correction
    node running before claim extraction has populated any claims).
    """


class SelfCorrectionError(AgentWorkflowError):
    """Raised when the self-correction agent fails to produce a corrected response."""


class KnowledgeGraphValidationError(AgentWorkflowError):
    """
    Raised when the knowledge-graph-validation agent fails to complete
    validation for reasons beyond ordinary backend degradation (which
    ``KnowledgeGraphService`` already handles internally by falling back
    to the in-memory graph) -- e.g. malformed claim/entity input.
    """


class DriftDetectionError(AgentWorkflowError):
    """Raised when the knowledge drift detection agent fails to complete its analysis."""


class RiskAnalysisError(AgentWorkflowError):
    """Raised when the risk analysis module fails to compute a risk assessment."""
