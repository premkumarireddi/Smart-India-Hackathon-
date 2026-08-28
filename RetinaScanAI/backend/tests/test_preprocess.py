import numpy as np
import torch

from app.pipeline.preprocess import IMG_SIZE, clahe_enhance, enhance_pipeline, to_model_tensor


def test_clahe_enhance_preserves_shape_and_dtype(sample_good_image):
    out = clahe_enhance(sample_good_image)
    assert out.shape == sample_good_image.shape
    assert out.dtype == np.uint8


def test_enhance_pipeline_resizes_to_target(sample_good_image):
    out = enhance_pipeline(sample_good_image, size=IMG_SIZE)
    assert out.shape == (IMG_SIZE, IMG_SIZE, 3)


def test_to_model_tensor_shape_and_normalization(sample_good_image):
    enhanced = enhance_pipeline(sample_good_image)
    tensor = to_model_tensor(enhanced)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 3, IMG_SIZE, IMG_SIZE)
    # normalized around ImageNet stats -> should not still be in [0, 255]
    assert tensor.max().item() < 10
    assert tensor.min().item() > -10
