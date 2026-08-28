"""
Stage 5: Auto-generated, annotated referral report — the human-in-the-loop
artifact a health worker hands (or a doctor reviews) after screening.
"""
from dataclasses import dataclass

from .model import CLASS_NAMES, REFERABLE_THRESHOLD

SEVERITY_DESCRIPTIONS = {
    0: "No visible signs of diabetic retinopathy.",
    1: "Mild non-proliferative DR: a few microaneurysms only.",
    2: "Moderate non-proliferative DR: more extensive microaneurysms and early hemorrhages.",
    3: "Severe non-proliferative DR: extensive hemorrhages, venous beading, or intraretinal microvascular abnormalities.",
    4: "Proliferative DR: neovascularization or vitreous/preretinal hemorrhage present. Sight-threatening.",
}


@dataclass
class ReferralReport:
    severity_level: int
    severity_name: str
    severity_description: str
    confidence: float
    is_referable: bool
    recommendation: str
    quality_reasons: list


def build_report(severity_level: int, confidence: float, quality_reasons=None) -> ReferralReport:
    quality_reasons = quality_reasons or []
    is_referable = severity_level >= REFERABLE_THRESHOLD

    if is_referable:
        recommendation = (
            f"REFER to an ophthalmologist. Predicted severity: {CLASS_NAMES[severity_level]} "
            f"(ICDR Level {severity_level}), model confidence {confidence * 100:.1f}%. "
            "Please review the Grad-CAM evidence map before confirming referral."
        )
    else:
        recommendation = (
            f"Routine annual re-screening recommended. Predicted severity: {CLASS_NAMES[severity_level]} "
            f"(ICDR Level {severity_level}), model confidence {confidence * 100:.1f}%."
        )

    return ReferralReport(
        severity_level=severity_level,
        severity_name=CLASS_NAMES[severity_level],
        severity_description=SEVERITY_DESCRIPTIONS[severity_level],
        confidence=confidence,
        is_referable=is_referable,
        recommendation=recommendation,
        quality_reasons=quality_reasons,
    )
