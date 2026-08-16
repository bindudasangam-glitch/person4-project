from __future__ import annotations

import os

from huggingface_hub import InferenceClient


class LLMService:
    """
    Generate an initial answer using Hugging Face Inference Providers.

    The HF_TOKEN environment variable must be configured in the runtime
    environment, for example on Render.
    """

    def __init__(self) -> None:
        self.model = "Qwen/Qwen2.5-7B-Instruct"

        self.hf_token = os.getenv("HF_TOKEN")

        if not self.hf_token:
            raise RuntimeError(
                "HF_TOKEN environment variable is not configured."
            )

        self.client = InferenceClient(
            api_key=self.hf_token,
            provider="auto",
        )

    def generate_answer(self, question: str) -> str:
        """
        Generate a direct, factual initial answer using Hugging Face.
        """

        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        clean_question = question.strip()

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a factual question-answering assistant. "
                    "Answer the user's question directly and accurately. "
                    "For simple factual questions, give the correct factual "
                    "answer in a short, clear sentence. "
                    "Do not evaluate or criticize a statement unless the "
                    "user explicitly asks you to verify or evaluate it. "
                    "Do not start your answer with phrases such as "
                    "'That is incorrect' unless the user explicitly asked "
                    "whether a provided statement is correct. "
                    "Do not invent facts. "
                    "Do not repeat the question. "
                    "Return only the answer."
                ),
            },
            {
                "role": "user",
                "content": clean_question,
            },
        ]

        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.0,
                max_tokens=256,
            )

        except Exception as exc:
            raise RuntimeError(
                "Hugging Face Inference Providers request failed. "
                "Check HF_TOKEN, token permissions, inference credits, "
                "model availability, and provider availability."
            ) from exc

        try:
            answer = completion.choices[0].message.content
        except (AttributeError, IndexError, TypeError) as exc:
            raise RuntimeError(
                "Hugging Face returned an invalid chat completion response."
            ) from exc

        if not answer or not answer.strip():
            raise RuntimeError(
                "The Hugging Face model returned an empty response."
            )

        return answer.strip()