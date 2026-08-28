import torch

from app.pipeline.model import CLASS_NAMES, NUM_CLASSES, SimpleCNN, build_model, predict


def test_simple_cnn_forward_shape():
    model = SimpleCNN(num_classes=NUM_CLASSES)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, NUM_CLASSES)


def test_build_model_resnet18_has_correct_head():
    model = build_model(architecture="resnet18", pretrained=False)
    x = torch.randn(1, 3, 224, 224)
    out = model(x)
    assert out.shape == (1, NUM_CLASSES)
    assert hasattr(model, "gradcam_layer_name")


def test_build_model_simple_cnn_fallback():
    model = build_model(architecture="simple_cnn")
    assert isinstance(model, SimpleCNN)


def test_predict_returns_valid_class_and_confidence():
    model = build_model(architecture="simple_cnn")
    model.eval()
    x = torch.randn(1, 3, 224, 224)
    pred_class, probs, confidence = predict(model, x)
    assert 0 <= pred_class < NUM_CLASSES
    assert len(probs) == NUM_CLASSES
    assert abs(float(probs.sum()) - 1.0) < 1e-4
    assert 0.0 <= confidence <= 1.0


def test_class_names_match_icdr_scale():
    assert CLASS_NAMES == ["No_DR", "Mild", "Moderate", "Severe", "Proliferative_DR"]
