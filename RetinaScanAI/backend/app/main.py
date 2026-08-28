"""FastAPI application entrypoint.

Run with:
    uvicorn app.main:app --reload --port 8000

Interactive API docs: http://localhost:8000/docs
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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


@app.get("/")
def root():
    return {
        "service": "RetinaScan AI backend",
        "docs": "/docs",
        "health": "/api/health",
        "predict": "POST /api/predict (multipart/form-data, field name 'file')",
    }
