# RetinaScan AI — deployed as a single container (Hugging Face Spaces,
# Docker SDK). FastAPI serves both the API and the static frontend
# (see backend/app/main.py's StaticFiles mount), so this one container
# is the whole app.
#
# Local equivalent (two dev servers) is documented in the README; this
# Dockerfile is only for the unified deployed build.

FROM python:3.11-slim

# libglib2.0-0 is the one system lib opencv-python-headless still wants
# even in headless mode, on a slim base image.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY backend/requirements-deploy.txt backend/requirements-deploy.txt
RUN pip install --no-cache-dir -r backend/requirements-deploy.txt

# Preserve the same backend/ + frontend/ sibling layout as local dev, so
# app/main.py's relative path to ../../frontend resolves identically in
# both environments with no special-casing.
COPY backend/app backend/app
COPY backend/models backend/models
COPY frontend frontend

# Hugging Face Spaces (Docker SDK) expects the app on the port declared in
# README.md's `app_port` field — 7860 is the conventional default.
ENV PORT=7860
EXPOSE 7860

# Serve the REAL-data-trained checkpoint by default in the public demo
# (not the synthetic fallback) — override by setting this env var in the
# Space's settings if you want the synthetic one instead.
ENV RETINASCAN_MODEL_PATH=/app/backend/models/retina_cnn_aptos.pt

# Hugging Face Spaces containers run as a non-root user; make sure the app
# directory (which includes the writable .cache128/ dirs some pipeline
# code may touch) is writable by anyone, since the exact UID varies.
RUN chmod -R 777 /app

WORKDIR /app/backend
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
