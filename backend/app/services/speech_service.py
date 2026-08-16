from __future__ import annotations

import os
from pathlib import Path

from faster_whisper import WhisperModel


class SpeechService:
    """Convert uploaded speech audio into text using faster-whisper."""

    def __init__(self) -> None:
        # Render Free has limited RAM (512 MB).
        # The tiny model uses significantly less memory than base.
        self.model_name = os.getenv("WHISPER_MODEL", "tiny")

        self.model = WhisperModel(
            self.model_name,
            device="cpu",
            compute_type="int8",
            cpu_threads=1,
            num_workers=1,
        )

    def transcribe(self, audio_path: str | Path) -> str:
        """Transcribe an audio file using the local Whisper model."""

        path = Path(audio_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Audio file not found: {path}"
            )

        if path.stat().st_size == 0:
            raise ValueError(
                "Audio file is empty."
            )

        segments, _info = self.model.transcribe(
            str(path),
            beam_size=1,
            best_of=1,
            temperature=0.0,
            vad_filter=True,
        )

        text = " ".join(
            segment.text.strip()
            for segment in segments
            if segment.text and segment.text.strip()
        ).strip()

        if not text:
            raise RuntimeError(
                "Speech-to-text returned an empty transcription."
            )

        return text