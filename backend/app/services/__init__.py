"""
Services Package
===================

Exposes the core pipeline services.

This file's existence is deliberate and load-bearing: without it,
``app.services`` was an *implicit namespace package* (PEP 420) rather than a
regular package, while its parent (``app``) and sibling (``app.models``) are
both regular packages with their own ``__init__.py``. That asymmetry allowed
``app.services``'s import path to be re-derived dynamically on each lookup
instead of being fixed once — which is what caused ``ConfidenceScorer`` (and
potentially the other service classes) to fail to resolve specifically
during pytest collection, despite importing correctly in a standalone
interpreter.

Imports here are intentionally minimal and one-directional (services only
depend on ``app.models``/``app.core``/``app.utils``, never on each other at
package-init time) to avoid reintroducing any circular import.
"""

from .claim_extractor import ClaimExtractionError, ClaimExtractor
from .confidence_scorer import ConfidenceScorer, ConfidenceScoringError
from .hallucination_detector import HallucinationDetectionError, HallucinationDetector
from .response_analyzer import ResponseAnalysisError, ResponseAnalyzer

__all__ = [
    "ClaimExtractionError",
    "ClaimExtractor",
    "ConfidenceScorer",
    "ConfidenceScoringError",
    "HallucinationDetectionError",
    "HallucinationDetector",
    "ResponseAnalysisError",
    "ResponseAnalyzer",
]