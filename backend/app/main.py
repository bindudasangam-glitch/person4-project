"""
Main FastAPI application for the Hallucination Detection System.

The application keeps startup lightweight:
- No ML models are loaded during startup.
- No ResponseAnalyzer is created during startup.
- No SentenceTransformer model is loaded during startup.
- No spaCy model is loaded during startup.
- No vector-store data is preloaded during startup.

Heavy components are initialized only when an API endpoint actually
needs them.
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


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

setup_logging()


# ---------------------------------------------------------------------------
# Application lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None, None]:
    """
    Lightweight application startup.

    IMPORTANT:
    Do not initialize ResponseAnalyzer, Retriever, embeddings,
    spaCy, SentenceTransformer, or Chroma here.
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


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    debug=settings.DEBUG,
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Lightweight meta endpoints
# ---------------------------------------------------------------------------

@app.get("/", tags=["Meta"])
def root() -> dict[str, Any]:
    """
    Root endpoint.
    """

    return {
        "message": "Hallucination Detection System API",
        "status": "running",
        "version": settings.APP_VERSION,
        "docs_url": "/docs",
    }


@app.get("/health", tags=["Meta"])
def health_check() -> dict[str, str]:
    """
    Lightweight health check.

    This endpoint does not load any ML models.
    """

    return {
        "status": "healthy",
    }


# ---------------------------------------------------------------------------
# API routers
# ---------------------------------------------------------------------------

# Import routers after the lightweight application has been created.
#
# These imports register the routes but the actual ResponseAnalyzer,
# Retriever, embedding models, spaCy models, etc. are still created
# lazily when their endpoints are called.

from app.api.analysis import router as analysis_router
from app.api.routes.documents import router as documents_router
from app.api.routes.retrieval import router as retrieval_router
from app.api.routes.verification import router as verification_router
from app.api.routes.voice import router as voice_router


# Hallucination analysis:
# /api/v1/analyze
app.include_router(
    analysis_router,
    prefix=settings.API_PREFIX,
)


# Document management
app.include_router(
    documents_router,
)


# Evidence retrieval
app.include_router(
    retrieval_router,
)


# Verification
app.include_router(
    verification_router,
)


# Voice
# /api/v1/...
app.include_router(
    voice_router,
    prefix=settings.API_PREFIX,
)


# ---------------------------------------------------------------------------
# Validation error handler
# ---------------------------------------------------------------------------

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """
    Return a clean 422 response for invalid request data.
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


# ---------------------------------------------------------------------------
# Global exception handler
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def unhandled_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """
    Catch unexpected application errors.

    The full exception is logged on the server while clients receive
    a safe generic error message.
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