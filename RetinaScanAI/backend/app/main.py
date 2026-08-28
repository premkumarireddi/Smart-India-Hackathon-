"""FastAPI application entrypoint.

Local dev (backend + static frontend as two separate dev servers):
    uvicorn app.main:app --reload --port 8000
    # in another terminal: cd ../frontend && python -m http.server 5500

Deployed (Hugging Face Spaces, Docker): this same app also serves the
frontend directly (StaticFiles mount below), so it's one container, one
URL, no CORS juggling — see Dockerfile at the repo root.

Interactive API docs: /docs
"""
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from . import config
from .routers import predict

app = FastAPI(
    title="RetinaScan AI",
    description="Explainable AI for Diabetic Retinopathy Screening in Rural India (SIH26038)",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS + ["*"],  # dev convenience; tighten for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router, prefix="/api", tags=["screening"])


@app.get("/api")
def api_info():
    return {
        "service": "RetinaScan AI backend",
        "docs": "/docs",
        "health": "/api/health",
        "predict": "POST /api/predict (multipart/form-data, field name 'file')",
    }


# Serve the frontend (index.html, app.js, styles.css) at "/", so a single
# deployed container is both the API and the UI. Falls back to just the
# API-info JSON above if the frontend folder isn't present (e.g. running
# the backend standalone against a separately-hosted frontend in dev).
_frontend_dir = Path(__file__).resolve().parents[2] / "frontend"
if _frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(_frontend_dir), html=True), name="frontend")
