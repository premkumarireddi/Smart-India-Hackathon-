"""
Stage 1 of the pipeline: Image Quality Assessment.

Rejects ungradeable fundus images *before* they reach the classifier, instead
of letting the model silently guess on a bad photo. This directly answers
the PS26038 requirement: "Automatically evaluate fundus images for adequacy
(focus, illumination, field of view)... reject ungradeable ones with
recapture feedback."

Three checks, each cheap enough to run in milliseconds on a phone-class CPU:
  - Focus / blur      -> variance of the Laplacian (a standard, well-studied
                          no-reference blur metric: a sharp image has more
                          high-frequency edge content, hence higher variance).
  - Illumination       -> mean pixel brightness in grayscale.
  - Field of view (FOV) -> fraction of the frame actually covered by retina
                          tissue vs. black border, via Otsu thresholding.
"""
from dataclasses import dataclass, asdict

import cv2
import numpy as np

# Thresholds were originally guessed against this repo's synthetic dataset
# (hard vector edges -> unrealistically high Laplacian-variance scores) and
# rejected 98% of REAL APTOS photos once tested against them (real fundus
# photography is naturally much smoother than a procedurally-drawn image,
# even when perfectly in focus). Recalibrated using percentile stats from a
# random 150-image sample of the real APTOS train set:
#   blur:       min 2.9  p10 8.5  median 14.3  p90 48.4  max 74.1
#   brightness: min 22.6            median 66.2           max 120.2
#   fov:        min 0.02            median 0.75            max 0.90
# Thresholds below sit near the low tail of that real distribution (catching
# genuine outliers) rather than near the median (which would reject normal
# photos). FOV_COVERAGE_MIN was left unchanged — at 0.35 it already only
# flagged 4/150 (2.7%) real images, a sane rate, so it wasn't actually
# miscalibrated the way blur/brightness were. Still an unsupervised,
# no-ground-truth calibration overall — re-tune against clinician-labeled
# "gradable/ungradable" data before real use; see docs/adr/0005 and the
# README "Honest status" section.
BLUR_VARIANCE_MIN = 6.0
BRIGHTNESS_MIN = 25.0
BRIGHTNESS_MAX = 220.0
FOV_COVERAGE_MIN = 0.35


@dataclass
class QualityReport:
    is_gradable: bool
    blur_score: float
    brightness_score: float
    fov_coverage: float
    reasons: list


def _blur_score(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _brightness_score(gray: np.ndarray) -> float:
    return float(gray.mean())


def _fov_coverage(gray: np.ndarray) -> float:
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(np.count_nonzero(mask)) / mask.size


def assess_quality(image_bgr: np.ndarray) -> QualityReport:
    """image_bgr: HxWx3 uint8 array (as read by cv2.imread / decoded upload)."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    blur = _blur_score(gray)
    brightness = _brightness_score(gray)
    fov = _fov_coverage(gray)

    reasons = []
    if blur < BLUR_VARIANCE_MIN:
        reasons.append(f"Image is too blurry (sharpness {blur:.1f} < {BLUR_VARIANCE_MIN}). Please refocus and recapture.")
    if brightness < BRIGHTNESS_MIN:
        reasons.append(f"Image is too dark (brightness {brightness:.1f} < {BRIGHTNESS_MIN}). Increase illumination and recapture.")
    if brightness > BRIGHTNESS_MAX:
        reasons.append(f"Image is overexposed (brightness {brightness:.1f} > {BRIGHTNESS_MAX}). Reduce flash/illumination and recapture.")
    if fov < FOV_COVERAGE_MIN:
        reasons.append(f"Retina field of view is too small ({fov * 100:.0f}% of frame). Recenter the eye and recapture.")

    return QualityReport(
        is_gradable=(len(reasons) == 0),
        blur_score=blur,
        brightness_score=brightness,
        fov_coverage=fov,
        reasons=reasons,
    )


def quality_report_dict(report: QualityReport) -> dict:
    return asdict(report)
