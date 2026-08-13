from __future__ import annotations

from pathlib import Path

from faster_whisper import WhisperModel


class SpeechService:
    """Convert uploaded speech audio into text using local Whisper."""

    def __init__(self) -> None:
        # Small model is suitable for local CPU-based development/testing.
        # The model is downloaded automatically the first time it is used.
        self.model_name = "base"
        self.model = WhisperModel(
            self.model_name,
            device="cpu",
            compute_type="int8",
        )

    def transcribe(self, audio_path: str | Path) -> str:
        """Transcribe an audio file using the local Whisper model."""

        path = Path(audio_path)

        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        if path.stat().st_size == 0:
            raise ValueError("Audio file is empty.")

        segments, _info = self.model.transcribe(
            str(path),
            beam_size=5,
        )

        text = " ".join(
            segment.text.strip()
            for segment in segments
            if segment.text and segment.text.strip()
        ).strip()

        if not text:
            raise RuntimeError("Local speech-to-text returned an empty transcription.")

        return text