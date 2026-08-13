"""
Deprecated: Legacy Analysis Result Model
===========================================

This module originally defined ``AnalysisResult`` as a bare 4-field
dataclass (``claims``, ``trust_score``, ``hallucination_probability``,
``reliability_score``, ``verdict``). It has been **superseded** by
:class:`app.services.response_analyzer.ResponseAnalysis`, which:

* Carries the full :class:`app.models.claim_model.ClaimModel` objects
  (with entities, claim type, verification status, evidence) rather than a
  bare claim list.
* Wraps the richer :class:`app.services.confidence_scorer.ConfidenceScoreResult`
  (trust score, reliability score, hallucination probability, confidence
  score, risk level, per-claim breakdown) instead of three floats.
* Uses the typed :class:`app.services.response_analyzer.Verdict` enum
  instead of a free-form ``str`` verdict.
* Provides ``to_dict()`` for direct API serialization.

This module is kept only so that any pre-existing imports of
``AnalysisResult`` do not break at import time. **Do not use it in new
code** — construct results via
``ResponseAnalyzer.analyze()`` instead, which returns a ``ResponseAnalysis``.

This module will be removed in a future revision once all call sites have
migrated.
"""

from __future__ import annotations

import warnings

from app.services.response_analyzer import ResponseAnalysis, Verdict

__all__ = ["AnalysisResult"]

_DEPRECATION_MESSAGE = (
    "app.models.analysis_result.AnalysisResult is deprecated and will be "
    "removed in a future revision. Use "
    "app.services.response_analyzer.ResponseAnalyzer.analyze() -> "
    "ResponseAnalysis instead."
)


class AnalysisResult(ResponseAnalysis):
    """
    Deprecated alias for :class:`ResponseAnalysis`.

    Emits a :class:`DeprecationWarning` on construction. Retained purely for
    backward compatibility with pre-existing imports; behaves identically to
    ``ResponseAnalysis`` in every other respect since it is a direct subclass.
    """

    def __new__(cls, *args: object, **kwargs: object) -> "AnalysisResult":
        warnings.warn(_DEPRECATION_MESSAGE, DeprecationWarning, stacklevel=2)
        return super().__new__(cls)


# Re-exported for convenience so legacy call sites that imported the verdict
# enum from this module (if any) continue to resolve correctly.
__deprecated_verdict__ = Verdict