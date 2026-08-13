from __future__ import annotations

from app.agents.workflow import HallucinationWorkflow
from app.models.workflow_models import WorkflowResult
from app.services.llm_service import LLMService


class QuestionService:
    """Generate an LLM answer and run it through the hallucination workflow."""

    def __init__(
        self,
        llm_service: LLMService,
        workflow: HallucinationWorkflow,
    ) -> None:
        self._llm_service = llm_service
        self._workflow = workflow

    def ask(
        self,
        question: str,
        document_id: str | None = None,
        top_k: int | None = None,
    ) -> tuple[str, WorkflowResult]:
        """Generate an initial answer and analyze it using Person 3."""

        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        original_response = self._llm_service.generate_answer(question)

        result = self._workflow.run(
            response_text=original_response,
            document_id=document_id,
            top_k=top_k,
        )

        return original_response, result