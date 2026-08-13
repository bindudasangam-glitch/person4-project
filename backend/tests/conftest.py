"""
Shared pytest configuration for the backend test suite.

Responsibilities
-----------------
* Ensures ``backend/`` is importable as the ``app`` package root even when
  the project hasn't been ``pip install -e``'d (common in CI containers that
  just run ``pytest`` from the repo root).
* Registers custom markers so tests can be selectively skipped/run.
* Quiets application logging during test runs so pytest output stays
  readable, while still allowing ``pytest -s --log-cli-level=INFO`` to
  surface it on demand.

Fixtures defined here are automatically available to every test module in
this directory without an explicit import (standard pytest ``conftest.py``
discovery).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

# --------------------------------------------------------------------------
# Ensure `app` is importable regardless of the working directory pytest is
# invoked from (e.g. repo root vs. `backend/`).
# --------------------------------------------------------------------------
_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers and quiet noisy application loggers during tests."""
    config.addinivalue_line(
        "markers",
        "nlp: marks tests that require the en_core_web_sm spaCy model to be installed.",
    )
    config.addinivalue_line(
        "markers",
        "slow: marks tests that are slow due to model loading or large inputs.",
    )

    # Application code logs at INFO/DEBUG for observability in production;
    # keep pytest's default output focused on failures during normal runs.
    logging.getLogger("app").setLevel(logging.WARNING)


@pytest.fixture(scope="session")
def sample_evidence_corpus() -> list[str]:
    """
    A small, reusable corpus of factual reference passages shared across
    hallucination-detection tests, avoiding duplicated literals per test file.
    """
    return [
        "The Eiffel Tower is located in Paris, France, and was completed in 1889.",
        "The Amazon River flows through Brazil and is the largest river by discharge volume.",
        "Mount Everest is located in the Himalayas on the border of Nepal and China.",
        "Python was created by Guido van Rossum and first released in 1991.",
        "The Great Barrier Reef is located off the coast of Queensland, Australia.",
    ]


@pytest.fixture(autouse=True)
def _reset_root_logger_level() -> None:
    """Ensure no single test can leak a changed root log level into the next."""
    root_logger = logging.getLogger()
    original_level = root_logger.level
    yield
    root_logger.setLevel(original_level)