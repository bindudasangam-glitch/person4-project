from __future__ import annotations

import json
import urllib.error
import urllib.request


class LLMService:
    """Generate an initial answer using a locally running Ollama model."""

    def __init__(self) -> None:
        self.model = "llama3.2:1b"
        self.base_url = "http://127.0.0.1:11434"

    def generate_answer(self, question: str) -> str:
        """Generate the initial LLM response using local Ollama."""

        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Answer the user's question clearly and accurately. "
                        "Do not deliberately invent facts. "
                        "Provide a useful initial answer that can later be "
                        "verified by the hallucination-detection system."
                    ),
                },
                {
                    "role": "user",
                    "content": question.strip(),
                },
            ],
            "stream": False,
            "options": {
                "temperature": 0.2,
            },
        }

        request = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(
                "Could not connect to Ollama. Make sure Ollama is running "
                "and the llama3.2:1b model is installed."
            ) from exc

        answer = result.get("message", {}).get("content")

        if not answer or not answer.strip():
            raise RuntimeError("The local LLM returned an empty response.")

        return answer.strip()