"""
Minimal FastAPI application.

This version is intentionally lightweight so the service can start
reliably on Render's free instance.
"""

from fastapi import FastAPI


app = FastAPI(
    title="Hallucination Detection System API",
    version="1.0.0",
)


@app.get("/", tags=["Meta"])
def root() -> dict[str, str]:
    """Root endpoint."""
    return {
        "message": "Hallucination Detection System API",
        "status": "running",
    }


@app.get("/health", tags=["Meta"])
def health() -> dict[str, str]:
    """Lightweight health-check endpoint."""
    return {
        "status": "healthy",
    }