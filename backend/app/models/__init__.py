"""
Models Package
================

Exposes the core domain model(s) used throughout the application.

Deliberately does NOT eagerly import ``app.models.analysis_result``: that
module is a deprecated backward-compatibility shim which itself imports
``app.services.response_analyzer`` (a layering inversion — ``models`` should
never depend on ``services``). Eagerly importing it here at package-init
time forced every plain ``from app.models.claim_model import ClaimModel``
to transitively load the entire services layer, creating a circular import
whenever the entry point was one of the service modules themselves
(``claim_extractor``, ``hallucination_detector``, ``confidence_scorer``,
``response_analyzer``).

``AnalysisResult`` remains fully usable — just import it explicitly from its
own module when needed:

    from app.models.analysis_result import AnalysisResult
"""

from .claim_model import (
    ClaimModel,
    ClaimType,
    ClaimValidationError,
    Entity,
    VerificationStatus,
)

__all__ = [
    "ClaimModel",
    "ClaimType",
    "ClaimValidationError",
    "Entity",
    "VerificationStatus",
]