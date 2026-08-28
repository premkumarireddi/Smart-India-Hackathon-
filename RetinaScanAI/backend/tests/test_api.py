import io
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app import config

client = TestClient(app)

MODEL_EXISTS = Path(config.MODEL_CHECKPOINT_PATH).exists()


def _png_bytes(img_bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img_bgr)
    assert ok
    return buf.tobytes()


def test_root_endpoint():
    r = client.get("/")
    assert r.status_code == 200
    assert "RetinaScan" in r.json()["service"]


def test_health_endpoint_reachable():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_rejects_bad_upload():
    r = client.post("/api/predict", files={"file": ("not_an_image.txt", b"hello world", "text/plain")})
    assert r.status_code == 400


def test_predict_quality_gate_rejects_blurry_image(sample_blurry_image):
    r = client.post(
        "/api/predict",
        files={"file": ("blurry.png", _png_bytes(sample_blurry_image), "image/png")},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["accepted"] is False
    assert body["quality"]["is_gradable"] is False
    assert len(body["quality"]["reasons"]) > 0
    # rejected images never reach the model, so no severity fields should be set
    assert body["severity_level"] is None


@pytest.mark.skipif(not MODEL_EXISTS, reason="Run `python ml/train.py` first to produce a checkpoint")
def test_predict_full_pipeline_on_synthetic_sample(synthetic_data_dir):
    sample_path = synthetic_data_dir / "Proliferative_DR" / "Proliferative_DR_0000.png"
    if not sample_path.exists():
        pytest.skip("Synthetic dataset not generated yet")

    with open(sample_path, "rb") as f:
        r = client.post("/api/predict", files={"file": ("sample.png", f.read(), "image/png")})

    assert r.status_code == 200
    body = r.json()
    assert body["accepted"] is True
    assert body["severity_level"] in range(5)
    assert body["severity_name"] is not None
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["is_referable"] in (True, False)
    assert len(body["class_probabilities"]) == 5
    assert body["gradcam_overlay_png_base64"]  # non-empty base64 string
    assert body["enhanced_image_png_base64"]
