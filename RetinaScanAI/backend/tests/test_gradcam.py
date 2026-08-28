import numpy as np
import torch

from app.pipeline.gradcam import GradCAM, lesion_evidence_summary, overlay_heatmap
from app.pipeline.model import SimpleCNN


def test_gradcam_produces_normalized_heatmap():
    model = SimpleCNN(num_classes=5)
    model.eval()
    cam_engine = GradCAM(model, model.gradcam_layer_name)

    x = torch.randn(1, 3, 224, 224)
    cam = cam_engine.generate(x, class_idx=2)

    assert cam.shape == (224, 224)
    assert cam.min() >= 0.0
    assert cam.max() <= 1.0 + 1e-6


def test_overlay_heatmap_shape_matches_base_image():
    base = np.zeros((224, 224, 3), dtype=np.uint8)
    cam = np.random.rand(224, 224).astype(np.float32)
    overlay = overlay_heatmap(base, cam)
    assert overlay.shape == base.shape
    assert overlay.dtype == np.uint8


def test_lesion_evidence_summary_keys():
    cam = np.random.rand(224, 224).astype(np.float32)
    summary = lesion_evidence_summary(cam)
    assert "hot_region_fraction" in summary
    assert "mean_activation_in_hot_region" in summary
    assert 0.0 <= summary["hot_region_fraction"] <= 1.0
