from app.pipeline.quality import assess_quality


def test_good_image_is_gradable(sample_good_image):
    report = assess_quality(sample_good_image)
    assert report.is_gradable is True
    assert report.reasons == []


def test_blurry_image_is_rejected(sample_blurry_image):
    report = assess_quality(sample_blurry_image)
    assert report.is_gradable is False
    assert any("blurry" in r for r in report.reasons)


def test_dark_image_is_rejected(sample_dark_image):
    report = assess_quality(sample_dark_image)
    assert report.is_gradable is False
    assert any("dark" in r for r in report.reasons)


def test_quality_report_has_expected_fields(sample_good_image):
    report = assess_quality(sample_good_image)
    assert hasattr(report, "blur_score")
    assert hasattr(report, "brightness_score")
    assert hasattr(report, "fov_coverage")
    assert 0.0 <= report.fov_coverage <= 1.0
