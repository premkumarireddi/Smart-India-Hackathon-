"""Centralized configuration. Everything overridable via environment
variables so this deploys cleanly to a container / cloud run without code
changes."""
import os
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]

MODEL_CHECKPOINT_PATH = os.environ.get(
    "RETINASCAN_MODEL_PATH", str(BACKEND_ROOT / "models" / "retina_cnn_demo.pt")
)
DEVICE = os.environ.get("RETINASCAN_DEVICE", "cpu")

# CORS: the static frontend is served from a different origin/port in dev
ALLOWED_ORIGINS = os.environ.get(
    "RETINASCAN_ALLOWED_ORIGINS", "http://localhost:5500,http://127.0.0.1:5500,http://localhost:8080"
).split(",")

MAX_UPLOAD_SIZE_MB = int(os.environ.get("RETINASCAN_MAX_UPLOAD_MB", "10"))
