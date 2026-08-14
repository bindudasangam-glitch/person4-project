"""
Main FastAPI application for the Hallucination Detection System.

Render / 512 MB friendly startup configuration.

IMPORTANT:
- No ML models are loaded during startup.
- No ResponseAnalyzer is created during startup.
- No SentenceTransformer model is loaded during startup.
- No spaCy model is loaded during startup.
- No Chroma/vector-store data is loaded during startup.
- Only the analysis router is registered here.
- Heavy dependencies are created only when /api/v1/analyze is called.

This keeps the Render Free instance as lightweight as possible.
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


# ============================================================================
# LOGGING
# ============================================================================

setup_logging()


# ============================================================================
# APPLICATION LIFESPAN
# ============================================================================


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None, None]:
    """
    Lightweight application lifespan.

    DO NOT initialize:
    - ResponseAnalyzer
    - Retriever
    - SentenceTransformer
    - spaCy
    - Chroma
    - embedding models
    - voice/Whisper models

    during startup.
    """

    logger.info(
        "Starting up %s v%s...",
        settings.APP_NAME,
        settings.APP_VERSION,
    )

    yield

    logger.info(
        "Shutting down %s.",
        settings.APP_NAME,
    )


# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    debug=settings.DEBUG,
    lifespan=lifespan,
)


# ============================================================================
# CORS
# ============================================================================

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# META ENDPOINTS
# ============================================================================


@app.get(
    "/",
    tags=["Meta"],
)
def root() -> dict[str, Any]:
    """
    Basic API information.
    """

    return {
        "message": "Hallucination Detection System API",
        "status": "running",
        "version": settings.APP_VERSION,
        "docs_url": "/docs",
    }


@app.get(
    "/health",
    tags=["Meta"],
)
def health_check() -> dict[str, str]:
    """
    Lightweight health check.

    IMPORTANT:
    This endpoint does not load any ML models.
    """

    return {
        "status": "healthy",
    }


# ============================================================================
# ANALYSIS ROUTER
# ============================================================================

# IMPORTANT:
# We intentionally register ONLY the analysis router here.
#
# The other routers can import heavy dependencies such as:
# - embeddings
# - vector stores
# - NLP models
# - voice/Whisper dependencies
#
# Loading all of them during Render startup can exceed the 512 MB
# Free-instance memory limit.
#
# The analysis router itself creates ResponseAnalyzer lazily through
# its FastAPI dependency when /api/v1/analyze is actually called.

from app.api.analysis import router as analysis_router


app.include_router(
    analysis_router,
    prefix=settings.API_PREFIX,
)


# ============================================================================
# VALIDATION ERROR HANDLER
# ============================================================================


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """
    Return a structured 422 response for request validation errors.
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


# ============================================================================
# GLOBAL EXCEPTION HANDLER
# ============================================================================


@app.exception_handler(Exception)
async def unhandled_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """
    Catch unexpected application exceptions.

    The complete exception is logged on the server.
    Clients receive only a safe generic message.
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