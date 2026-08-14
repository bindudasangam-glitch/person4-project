"""
Bootstraps the Hallucination Detection System API.

- Configures structured logging before anything else runs.
- Builds the FastAPI app instance from centralized settings.
- Loads NLP/embedding models lazily only when actually needed.
- Registers all API routers.
- Installs global exception handlers.
- Enables CORS for the React/Vite frontend.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.analysis import router as analysis_router
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

    IMPORTANT:
    NLP, embedding, spaCy, SentenceTransformer, and vector-store
    components are NOT loaded during application startup.

    They are initialized lazily only when an endpoint actually needs
    them. This keeps startup memory usage low and helps the application
    run on Render's 512 MB Free instance.
    """

    logger.info(
        "Starting up %s v%s...",
        settings.APP_NAME,
        settings.APP_VERSION,
    )

    # Do NOT initialize ResponseAnalyzer here.
    # Do NOT load SentenceTransformer here.
    # Do NOT load spaCy models here.
    # Do NOT preload Chroma here.

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
# CORS configuration
# -------------------------------------------------------------------

# Local development:
#   http://localhost:5173
#   http://localhost:5174
#   http://localhost:5175
#   etc.
#
#   http://127.0.0.1:5173
#   http://127.0.0.1:5174
#   etc.
#
# Production frontend can be added later through an environment
# variable if required.

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

# Hallucination analysis
app.include_router(
    analysis_router,
    prefix=settings.API_PREFIX,
)

# Document management
app.include_router(documents_router)

# Evidence retrieval
app.include_router(retrieval_router)

# Verification
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
    logging the complete exception on the server.
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
    Lightweight health check endpoint.

    This endpoint intentionally does not load any ML models.
    """

    return {
        "status": "healthy",
    }