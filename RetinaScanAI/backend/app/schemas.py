"""Pydantic response models for the API — also doubles as OpenAPI/Swagger
documentation (visible at /docs when the server is running)."""
from typing import List, Optional

from pydantic import BaseModel, Field


class QualityInfo(BaseModel):
    is_gradable: bool
    blur_score: float
    brightness_score: float
    fov_coverage: float
    reasons: List[str]


class ClassProbability(BaseModel):
    label: str
    probability: float


class PredictResponse(BaseModel):
    accepted: bool = Field(..., description="False if the image was rejected at the quality-gate stage")
    quality: QualityInfo
    severity_level: Optional[int] = Field(None, description="ICDR scale 0-4")
    severity_name: Optional[str] = None
    severity_description: Optional[str] = None
    confidence: Optional[float] = None
    is_referable: Optional[bool] = None
    recommendation: Optional[str] = None
    class_probabilities: Optional[List[ClassProbability]] = None
    gradcam_overlay_png_base64: Optional[str] = Field(
        None, description="Grad-CAM heatmap overlaid on the enhanced image, as a base64 PNG"
    )
    enhanced_image_png_base64: Optional[str] = Field(
        None, description="The CLAHE-enhanced image the model actually saw, as a base64 PNG"
    )
    model_info: Optional[dict] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    architecture: Optional[str] = None
    trained_on: Optional[str] = None
