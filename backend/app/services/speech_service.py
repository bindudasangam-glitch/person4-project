from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import InferenceClient


class SpeechService:
    """Convert uploaded speech audio into text using Hugging Face ASR."""

    def __init__(self) -> None:
        self.hf_token = os.getenv("HF_TOKEN")

        if not self.hf_token:
            raise RuntimeError(
                "HF_TOKEN environment variable is not configured."
            )

        self.model_name = os.getenv(
            "WHISPER_MODEL",
            "openai/whisper-large-v3",
        )

        self.client = InferenceClient(
            provider="hf-inference",
            api_key=self.hf_token,
        )

    def transcribe(self, audio_path: str | Path) -> str:
        """Transcribe an audio file using Hugging Face ASR."""

        path = Path(audio_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Audio file not found: {path}"
            )

        if path.stat().st_size == 0:
            raise ValueError("Audio file is empty.")

        try:
            result = self.client.automatic_speech_recognition(
                str(path),
                model=self.model_name,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Hugging Face speech-to-text failed: {exc}"
            ) from exc

        text = getattr(result, "text", None)

        if not text:
            text = str(result)

        text = text.strip()

        if not text:
            raise RuntimeError(
                "Hugging Face speech-to-text returned an empty transcription."
            )

        return text