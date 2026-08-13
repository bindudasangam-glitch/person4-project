from __future__ import annotations

import base64
import tempfile
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.encoders import jsonable_encoder

from app.agents.workflow import HallucinationWorkflow
from app.api.routes.workflow import get_hallucination_workflow
from app.services.llm_service import LLMService
from app.services.question_service import QuestionService
from app.services.speech_service import SpeechService
from app.services.tts_service import TTSService


router = APIRouter(prefix="/voice", tags=["voice"])


MAX_AUDIO_SIZE = 20 * 1024 * 1024


ALLOWED_AUDIO_TYPES = {
    "audio/mpeg": ".mp3",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/mp4": ".m4a",
    "audio/x-m4a": ".m4a",
    "audio/webm": ".webm",
}


OutputType = Literal["text", "voice", "both"]


def _validate_output_type(output_type: str) -> OutputType:
    """Validate and normalize the requested output mode."""

    normalized = output_type.strip().lower()

    if normalized not in {"text", "voice", "both"}:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "Invalid output_type. "
                "Use one of: text, voice, both."
            ),
        )

    return normalized  # type: ignore[return-value]


def _serialize_workflow_result(workflow_result) -> dict:
    """
    Convert WorkflowResult into a JSON-compatible dictionary.

    FastAPI's jsonable_encoder handles dataclasses, Pydantic models,
    dictionaries, lists, and other common Python structures safely.
    """

    encoded = jsonable_encoder(workflow_result)

    if isinstance(encoded, dict):
        return encoded

    return {"result": encoded}


def _create_audio_base64(audio_path: Path) -> str:
    """Read a generated WAV file and encode it for JSON transport."""

    if not audio_path.exists():
        raise RuntimeError("Generated audio file does not exist.")

    if audio_path.stat().st_size == 0:
        raise RuntimeError("Generated audio file is empty.")

    return base64.b64encode(audio_path.read_bytes()).decode("ascii")


def _run_question_pipeline(
    question: str,
    workflow: HallucinationWorkflow,
):
    """
    Run the existing LLM + Person 1/2/3 workflow.

    This deliberately reuses QuestionService so the six voice/text
    combinations all use exactly the same existing analysis pipeline.
    """

    if not question or not question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty.",
        )

    llm_service = LLMService()

    question_service = QuestionService(
        llm_service=llm_service,
        workflow=workflow,
    )

    original_response, workflow_result = question_service.ask(
        question.strip()
    )

    final_response = (
        getattr(workflow_result, "corrected_response", None)
        or getattr(workflow_result, "response_text", None)
        or original_response
    )

    if not final_response or not final_response.strip():
        raise RuntimeError(
            "The hallucination workflow did not produce a final response."
        )

    return (
        original_response,
        final_response.strip(),
        workflow_result,
    )


@router.post("/process")
async def process_voice_or_text(
    output_type: str = Form("text"),
    text: str | None = Form(None),
    file: UploadFile | None = File(None),
    workflow: HallucinationWorkflow = Depends(
        get_hallucination_workflow
    ),
):
    """
    Process either text or voice input and return text, voice, or both.

    Supported combinations:

    1. Voice -> Text
    2. Voice -> Voice
    3. Voice -> Both
    4. Text  -> Text
    5. Text  -> Voice
    6. Text  -> Both

    All combinations use the same existing:
        input
        -> Whisper (for voice)
        -> Ollama
        -> Person 1/2 verification
        -> Person 3 workflow
        -> self-correction
        -> optional local Windows TTS
    """

    output_mode = _validate_output_type(output_type)

    audio_input_path: Path | None = None
    audio_output_path: Path | None = None

    try:
        # ---------------------------------------------------------
        # 1. Determine input type
        # ---------------------------------------------------------

        has_text = bool(text and text.strip())
        has_file = file is not None

        if not has_text and not has_file:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Provide either text input or an audio file.",
            )

        if has_text and has_file:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "Provide only one input type at a time: "
                    "either text or audio."
                ),
            )

        # ---------------------------------------------------------
        # 2. Convert voice input to text using local Whisper
        # ---------------------------------------------------------

        if has_file:
            if not file.filename:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Audio file is required.",
                )

            if file.content_type not in ALLOWED_AUDIO_TYPES:
                raise HTTPException(
                    status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                    detail=(
                        f"Unsupported audio format: {file.content_type}. "
                        f"Supported formats: "
                        f"{', '.join(ALLOWED_AUDIO_TYPES.keys())}"
                    ),
                )

            audio_data = await file.read()

            if not audio_data:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Audio file is empty.",
                )

            if len(audio_data) > MAX_AUDIO_SIZE:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail="Audio file is too large.",
                )

            suffix = ALLOWED_AUDIO_TYPES[file.content_type]

            with tempfile.NamedTemporaryFile(
                suffix=suffix,
                delete=False,
            ) as temp_audio:
                temp_audio.write(audio_data)
                audio_input_path = Path(temp_audio.name)

            speech_service = SpeechService()

            question = speech_service.transcribe(
                audio_input_path
            )

            if not question or not question.strip():
                raise RuntimeError(
                    "Speech-to-text did not produce a question."
                )

            input_type = "voice"

        # ---------------------------------------------------------
        # 3. Use text input directly
        # ---------------------------------------------------------

        else:
            question = text.strip()
            input_type = "text"

        # ---------------------------------------------------------
        # 4. Run the existing LLM + Person 1/2/3 workflow
        # ---------------------------------------------------------

        (
            original_response,
            corrected_response,
            workflow_result,
        ) = _run_question_pipeline(
            question=question,
            workflow=workflow,
        )

        # ---------------------------------------------------------
        # 5. Prepare common response
        # ---------------------------------------------------------

        workflow_result_data = _serialize_workflow_result(
            workflow_result
        )

        response_data = {
            "input_type": input_type,
            "output_type": output_mode,
            "question": question,
            "transcription": question if input_type == "voice" else None,
            "original_response": original_response,
            "corrected_response": corrected_response,
            "workflow_result": workflow_result_data,
            "audio": None,
        }

        # ---------------------------------------------------------
        # 6. Generate local TTS only when requested
        # ---------------------------------------------------------

        if output_mode in {"voice", "both"}:
            tts_service = TTSService()

            with tempfile.NamedTemporaryFile(
                suffix=".wav",
                delete=False,
            ) as temp_output:
                audio_output_path = Path(temp_output.name)

            tts_service.synthesize(
                text=corrected_response,
                output_path=audio_output_path,
            )

            audio_base64 = _create_audio_base64(
                audio_output_path
            )

            response_data["audio"] = {
                "mime_type": "audio/wav",
                "filename": "corrected_answer.wav",
                "base64": audio_base64,
            }

        # ---------------------------------------------------------
        # 7. Return final response
        # ---------------------------------------------------------

        return response_data

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Voice/text processing failed: {exc}",
        ) from exc

    finally:
        # ---------------------------------------------------------
        # 8. Clean temporary input/output files
        # ---------------------------------------------------------

        if audio_input_path and audio_input_path.exists():
            audio_input_path.unlink(missing_ok=True)

        if audio_output_path and audio_output_path.exists():
            audio_output_path.unlink(missing_ok=True)


@router.post("/analyze")
async def analyze_voice(
    file: UploadFile = File(...),
    workflow: HallucinationWorkflow = Depends(
        get_hallucination_workflow
    ),
):
    """
    Backward-compatible Voice-only endpoint.

    This preserves the original /voice/analyze behavior while
    internally using the new six-mode processing implementation.
    """

    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Audio file is required.",
        )

    if file.content_type not in ALLOWED_AUDIO_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Unsupported audio format: {file.content_type}. "
                f"Supported formats: "
                f"{', '.join(ALLOWED_AUDIO_TYPES.keys())}"
            ),
        )

    audio_input_path: Path | None = None
    audio_output_path: Path | None = None

    try:
        audio_data = await file.read()

        if not audio_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Audio file is empty.",
            )

        if len(audio_data) > MAX_AUDIO_SIZE:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="Audio file is too large.",
            )

        suffix = ALLOWED_AUDIO_TYPES[file.content_type]

        with tempfile.NamedTemporaryFile(
            suffix=suffix,
            delete=False,
        ) as temp_audio:
            temp_audio.write(audio_data)
            audio_input_path = Path(temp_audio.name)

        speech_service = SpeechService()

        question = speech_service.transcribe(
            audio_input_path
        )

        (
            original_response,
            corrected_response,
            workflow_result,
        ) = _run_question_pipeline(
            question=question,
            workflow=workflow,
        )

        tts_service = TTSService()

        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False,
        ) as temp_output:
            audio_output_path = Path(temp_output.name)

        tts_service.synthesize(
            text=corrected_response,
            output_path=audio_output_path,
        )

        audio_base64 = _create_audio_base64(
            audio_output_path
        )

        return {
            "input_type": "voice",
            "output_type": "both",
            "question": question,
            "transcription": question,
            "original_response": original_response,
            "corrected_response": corrected_response,
            "workflow_result": _serialize_workflow_result(
                workflow_result
            ),
            "audio": {
                "mime_type": "audio/wav",
                "filename": "corrected_answer.wav",
                "base64": audio_base64,
            },
        }

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Voice processing failed: {exc}",
        ) from exc

    finally:
        if audio_input_path and audio_input_path.exists():
            audio_input_path.unlink(missing_ok=True)

        if audio_output_path and audio_output_path.exists():
            audio_output_path.unlink(missing_ok=True)