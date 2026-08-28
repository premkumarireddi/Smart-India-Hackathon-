from app.pipeline.report import build_report


def test_no_dr_is_not_referable():
    report = build_report(severity_level=0, confidence=0.95)
    assert report.is_referable is False
    assert "Routine" in report.recommendation


def test_moderate_dr_is_referable():
    report = build_report(severity_level=2, confidence=0.88)
    assert report.is_referable is True
    assert "REFER" in report.recommendation


def test_proliferative_dr_is_referable():
    report = build_report(severity_level=4, confidence=0.99)
    assert report.is_referable is True
    assert report.severity_name == "Proliferative_DR"


def test_confidence_appears_in_recommendation_text():
    report = build_report(severity_level=3, confidence=0.7654)
    assert "76.5%" in report.recommendation
