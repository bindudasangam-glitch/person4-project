from __future__ import annotations

import threading
from typing import Final

import spacy
from spacy.language import Language

from app.core.logging import logger


_SPACY_MODEL_NAME: Final[str] = "en_core_web_sm"

_nlp: Language | None = None
_lock = threading.Lock()


def get_spacy_pipeline() -> Language:
    """
    Return one shared, lazily-loaded spaCy pipeline per worker process.
    """
    global _nlp

    if _nlp is not None:
        return _nlp

    with _lock:
        if _nlp is not None:
            return _nlp

        logger.info(
            "Loading shared spaCy pipeline '%s'.",
            _SPACY_MODEL_NAME,
        )

        try:
            _nlp = spacy.load(_SPACY_MODEL_NAME)
        except OSError as exc:
            raise RuntimeError(
                f"Required spaCy model '{_SPACY_MODEL_NAME}' is not available."
            ) from exc

        logger.info(
            "Shared spaCy pipeline '%s' loaded successfully.",
            _SPACY_MODEL_NAME,
        )

        return _nlp