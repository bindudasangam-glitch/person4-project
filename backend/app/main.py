"""
Bootstraps the Hallucination Detection System API.

- Configures structured logging before anything else runs.
- Builds the FastAPI app instance from centralized settings.
- Pre-warms the NLP pipeline at startup.
- Registers all API routers.
- Installs global exception handlers.
- Enables CORS so the React/Vite frontend can communicate with the API.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.analysis import (
    get_response_analyzer,
    router as analysis_router,
)
from app.api.routes.documents import router as documents_router
from app.api.routes.retrieval import router as retrieval_router
from app.api.routes.verification import router as verification_router
from app.api.routes.voice import router as voice_router
from app.core.config import settings
from app.core.logging import logger, setup_logging


# -------------------------------------------------------------------
# Logging configuration
# -------------------------------------------------------------------

setup_logging()


# -------------------------------------------------------------------
# Application lifespan
# -------------------------------------------------------------------

@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan hook.

    Pre-warms the ResponseAnalyzer and the spaCy pipelines it depends on
    so the first incoming request does not pay the model-loading cost.
    """

    logger.info(
        "Starting up %s v%s...",
        settings.APP_NAME,
        settings.APP_VERSION,
    )

    try:
        get_response_analyzer()
        logger.info("NLP pipelines pre-warmed successfully.")

    except Exception:  # noqa: BLE001
        logger.exception(
            "Failed to pre-warm NLP pipelines at startup; "
            "they will be loaded lazily on first request instead."
        )

    yield

    logger.info("Shutting down %s.", settings.APP_NAME)


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
# CORS configuration
# -------------------------------------------------------------------

# Allow the React/Vite frontend to run on any localhost or
# 127.0.0.1 port during local development.
#
# This avoids CORS failures when Vite automatically moves from
# 5173 -> 5174 -> 5175 -> 5176 -> etc. because a port is busy.
#
# Only localhost / 127.0.0.1 origins are allowed.

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------------------------------------------------
# API Routers
# -------------------------------------------------------------------

app.include_router(
    analysis_router,
    prefix=settings.API_PREFIX,
)

app.include_router(documents_router)
app.include_router(retrieval_router)
app.include_router(verification_router)

# Voice feature
app.include_router(
    voice_router,
    prefix=settings.API_PREFIX,
)


# -------------------------------------------------------------------
# Validation error handler
# -------------------------------------------------------------------

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """
    Return a structured 422 response for request payload
    validation failures.
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
    Catch-all handler for unexpected exceptions.

    Prevents raw stack traces from being exposed to clients while
    logging full details on the server.
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
    Root endpoint providing basic service metadata.
    """

    return {
        "message": "Welcome to the Hallucination Detection System API",
        "version": settings.APP_VERSION,
        "docs_url": "/docs",
    }


# -------------------------------------------------------------------
# Health check endpoint
# -------------------------------------------------------------------

@app.get("/health", tags=["Meta"])
def health_check() -> dict[str, str]:
    """
    Liveness/readiness probe endpoint.
    """

    return {"status": "healthy"}