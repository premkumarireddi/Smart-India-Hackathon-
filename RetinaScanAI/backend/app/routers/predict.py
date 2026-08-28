"""
The single most important route in this repo: POST /api/predict.

Wires together every pipeline stage described in PS26038, in order:
    upload -> quality gate -> enhance -> classify -> Grad-CAM -> report
"""
import base64
import io
import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import Image

from .. import config
from ..pipeline import gradcam as gradcam_mod
from ..pipeline import quality as quality_mod
from ..pipeline import report as report_mod
from ..pipeline.model import CLASS_NAMES, load_trained_model, predict as model_predict
from ..pipeline.preprocess import enhance_pipeline, to_model_tensor
from ..schemas import ClassProbability, HealthResponse, PredictResponse, QualityInfo

logger = logging.getLogger(__name__)
router = APIRouter()

_model = None
_model_meta = {}


def get_model():
    """Lazy-loaded singleton so the (potentially slow) checkpoint load
    happens once per process, not once per request."""
    global _model, _model_meta
    if _model is None:
        path = Path(config.MODEL_CHECKPOINT_PATH)
        if not path.exists():
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Model checkpoint not found at {path}. "
                    "Run `python ml/train.py` first (see README Quickstart)."
                ),
            )
        _model = load_trained_model(str(path), device=config.DEVICE)
        ckpt = torch.load(str(path), map_location=config.DEVICE)
        _model_meta = {
            "architecture": ckpt.get("architecture"),
            "trained_on": ckpt.get("trained_on"),
            "final_metrics": ckpt.get("final_metrics"),
        }
    return _model


def _decode_upload(raw_bytes: bytes) -> np.ndarray:
    try:
        pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image. Please upload a JPEG or PNG.")
    rgb = np.array(pil_img)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _encode_png_base64(bgr_image: np.ndarray) -> str:
    success, buf = cv2.imencode(".png", bgr_image)
    if not success:
        raise RuntimeError("PNG encoding failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


@router.get("/health", response_model=HealthResponse)
def health():
    path = Path(config.MODEL_CHECKPOINT_PATH)
    if not path.exists():
        return HealthResponse(status="ok", model_loaded=False)
    try:
        get_model()
        return HealthResponse(
            status="ok",
            model_loaded=True,
            architecture=_model_meta.get("architecture"),
            trained_on=_model_meta.get("trained_on"),
        )
    except Exception as e:
        logger.exception("Health check model load failed")
        return HealthResponse(status=f"model_load_error: {e}", model_loaded=False)


@router.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    raw = await file.read()
    size_mb = len(raw) / (1024 * 1024)
    if size_mb > config.MAX_UPLOAD_SIZE_MB:
        raise HTTPException(status_code=413, detail=f"Image too large ({size_mb:.1f} MB > {config.MAX_UPLOAD_SIZE_MB} MB limit).")

    image_bgr = _decode_upload(raw)

    # --- Stage 1: quality gate ---------------------------------------
    q = quality_mod.assess_quality(image_bgr)
    quality_info = QualityInfo(**quality_mod.quality_report_dict(q))

    if not q.is_gradable:
        return PredictResponse(accepted=False, quality=quality_info)

    # --- Stage 2: enhancement -----------------------------------------
    enhanced = enhance_pipeline(image_bgr)
    tensor = to_model_tensor(enhanced)

    # --- Stage 3: classification ---------------------------------------
    model = get_model()
    pred_class, probs, confidence = model_predict(model, tensor, device=config.DEVICE)

    # --- Stage 4: Grad-CAM explainability --------------------------------
    cam_model = get_model()  # same instance; GradCAM registers hooks on it
    target_layer = getattr(cam_model, "gradcam_layer_name", "layer4")
    cam_engine = gradcam_mod.GradCAM(cam_model, target_layer)
    tensor_for_cam = tensor.clone().requires_grad_(False)
    cam = cam_engine.generate(tensor_for_cam, pred_class)
    overlay = gradcam_mod.overlay_heatmap(enhanced, cam)

    # --- Stage 5: referral report ----------------------------------------
    report = report_mod.build_report(pred_class, confidence, quality_reasons=[])

    class_probs = [
        ClassProbability(label=CLASS_NAMES[i], probability=round(float(probs[i]), 4))
        for i in range(len(CLASS_NAMES))
    ]

    return PredictResponse(
        accepted=True,
        quality=quality_info,
        severity_level=report.severity_level,
        severity_name=report.severity_name,
        severity_description=report.severity_description,
        confidence=round(confidence, 4),
        is_referable=report.is_referable,
        recommendation=report.recommendation,
        class_probabilities=class_probs,
        gradcam_overlay_png_base64=_encode_png_base64(overlay),
        enhanced_image_png_base64=_encode_png_base64(enhanced),
        model_info=_model_meta,
    )
