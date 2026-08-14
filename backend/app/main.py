"""
Lightweight bootstrap for the Hallucination Detection System API.

IMPORTANT:
- Keeps startup extremely lightweight for Render Free (512 MB).
- Does NOT import heavy API routers during startup.
- Does NOT load spaCy, SentenceTransformer, Chroma, Torch, etc.
- Provides / and /health immediately.
- API routers can be re-enabled later after the service is Live.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.core.logging import logger, setup_logging


# -------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------

setup_logging()


# -------------------------------------------------------------------
# Application lifespan
# -------------------------------------------------------------------

@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None, None]:
    """
    Lightweight application startup.

    IMPORTANT:
    Do not load any ML, NLP, embedding, vector-store, or router
    dependencies during startup.
    """

    logger.info(
        "Starting up %s v%s...",
        settings.APP_NAME,
        settings.APP_VERSION,
    )

    # Intentionally empty.
    #
    # DO NOT initialize:
    # - SentenceTransformer
    # - spaCy
    # - Chroma
    # - Torch
    # - ResponseAnalyzer
    # - document/retrieval/verification services

    yield

    logger.info(
        "Shutting down %s.",
        settings.APP_NAME,
    )


# -------------------------------------------------------------------
# FastAPI application
# -------------------------------------------------------------------

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    debug=settings.DEBUG,
    lifespan=lifespan,
)


# -------------------------------------------------------------------
# CORS
# -------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------------------------------------------------
# IMPORTANT
# -------------------------------------------------------------------
#
# API routers are intentionally DISABLED for the first Render deploy.
#
# The previous version imported:
#
#   app.api.analysis
#   app.api.routes.documents
#   app.api.routes.retrieval
#   app.api.routes.verification
#   app.api.routes.voice
#
# Some of those modules can import heavy ML/vector dependencies.
#
# That causes Render to spend too much time/memory before opening
# the HTTP port.
#
# We will re-enable them one at a time AFTER /health works on Render.
#
# -------------------------------------------------------------------

# from app.api.analysis import router as analysis_router
# from app.api.routes.documents import router as documents_router
# from app.api.routes.retrieval import router as retrieval_router
# from app.api.routes.verification import router as verification_router
# from app.api.routes.voice import router as voice_router


# -------------------------------------------------------------------
# Routers temporarily disabled
# -------------------------------------------------------------------

# app.include_router(
#     analysis_router,
#     prefix=settings.API_PREFIX,
# )

# app.include_router(documents_router)

# app.include_router(retrieval_router)

# app.include_router(verification_router)

# app.include_router(
#     voice_router,
#     prefix=settings.API_PREFIX,
# )


# -------------------------------------------------------------------
# Validation error handler
# -------------------------------------------------------------------

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """
    Return a structured 422 response for validation errors.
    """

    logger.warning(
        "Validation error on %s %s: %s",
        request.method,
        request.url.path,
        exc.errors(),
    )

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "validation_error",
            "detail": exc.errors(),
        },
    )


# -------------------------------------------------------------------
# Global exception handler
# -------------------------------------------------------------------

@app.exception_handler(Exception)
async def unhandled_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """
    Catch unexpected exceptions without exposing stack traces.
    """

    logger.exception(
        "Unhandled exception on %s %s.",
        request.method,
        request.url.path,
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "internal_server_error",
            "detail": (
                "An unexpected error occurred. "
                "Please try again later."
            ),
        },
    )


# -------------------------------------------------------------------
# Root endpoint
# -------------------------------------------------------------------

@app.get("/", tags=["Meta"])
def root() -> dict[str, Any]:
    """
    Basic service metadata.
    """

    return {
        "message": "Welcome to the Hallucination Detection System API",
        "version": settings.APP_VERSION,
        "docs_url": "/docs",
    }


# -------------------------------------------------------------------
# Health check
# -------------------------------------------------------------------

@app.get("/health", tags=["Meta"])
def health_check() -> dict[str, str]:
    """
    Lightweight health check.

    This endpoint must NEVER load ML models.
    """

    return {
        "status": "healthy",
    }