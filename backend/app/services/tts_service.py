from __future__ import annotations

import subprocess
from pathlib import Path


class TTSService:
    """Convert the final corrected answer into speech audio locally."""

    def __init__(self) -> None:
        self.rate = 170

    def synthesize(self, text: str, output_path: str | Path) -> Path:
        """Generate speech audio using Windows built-in speech synthesis."""

        if not text or not text.strip():
            raise ValueError("Text for speech cannot be empty.")

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Windows PowerShell can use the built-in System.Speech synthesizer.
        # Generate a temporary WAV file, then move it to the requested path.
        if path.suffix.lower() != ".wav":
            raise ValueError(
                "Local Windows TTS currently requires a .wav output path."
            )

        escaped_text = text.strip().replace("'", "''")
        escaped_path = str(path.resolve()).replace("'", "''")

        powershell_script = (
            "Add-Type -AssemblyName System.Speech; "
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f"$s.Rate = {self.rate - 200}; "
            f"$s.SetOutputToWaveFile('{escaped_path}'); "
            f"$s.Speak('{escaped_text}'); "
            "$s.Dispose()"
        )

        try:
            result = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-NonInteractive",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-Command",
                    powershell_script,
                ],
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("Local text-to-speech timed out.") from exc

        if result.returncode != 0:
            error = result.stderr.strip() or result.stdout.strip()
            raise RuntimeError(
                f"Local Windows text-to-speech failed: {error}"
            )

        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(
                "Local text-to-speech returned an empty audio file."
            )

        return path