import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def synthetic_data_dir():
    return Path(__file__).resolve().parents[1] / "data" / "synthetic"


@pytest.fixture
def sample_good_image():
    """A bright, sharp, well-framed fake fundus image (should pass quality gate)."""
    rng = np.random.RandomState(0)
    img = np.full((224, 224, 3), 120, dtype=np.uint8)
    cv2_circle_img = img.copy()
    import cv2
    cv2.circle(cv2_circle_img, (112, 112), 100, (160, 80, 50), -1)
    # add some texture so the Laplacian variance (sharpness) is non-trivial
    noise = rng.randint(-20, 20, size=cv2_circle_img.shape, dtype=np.int16)
    textured = np.clip(cv2_circle_img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return textured


@pytest.fixture
def sample_blurry_image(sample_good_image):
    import cv2
    return cv2.GaussianBlur(sample_good_image, (25, 25), 10)


@pytest.fixture
def sample_dark_image(sample_good_image):
    return (sample_good_image.astype(np.float32) * 0.1).astype(np.uint8)
