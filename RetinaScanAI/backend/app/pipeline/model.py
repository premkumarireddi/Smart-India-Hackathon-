"""
Stage 3: The DR severity classifier.

Architecture: ImageNet-pretrained ResNet18 backbone with a fresh 5-class
head, fine-tuned end to end (transfer learning) — this is the Python/PyTorch
equivalent of the "CNN transfer learning (EfficientNet/ResNet backbone)"
approach described in the pitch deck and the MATLAB Deep Learning Toolbox
in the original problem statement. See docs/adr/0001 and 0002 for why this
prototype is implemented in PyTorch rather than MATLAB.

If pretrained weights can't be downloaded (offline / air-gapped dev box),
`build_model()` automatically falls back to a small from-scratch CNN so the
pipeline never hard-fails just because of a missing internet connection.

Output: 5 logits corresponding to the International Clinical Diabetic
Retinopathy (ICDR) severity scale:
    0 = No DR, 1 = Mild, 2 = Moderate, 3 = Severe, 4 = Proliferative DR
"""
import logging

import torch
import torch.nn as nn
import torchvision

logger = logging.getLogger(__name__)

NUM_CLASSES = 5
CLASS_NAMES = ["No_DR", "Mild", "Moderate", "Severe", "Proliferative_DR"]
REFERABLE_THRESHOLD = 2  # ICDR Level 2+ ("Moderate" and above) = referable DR


class SimpleCNN(nn.Module):
    """Dependency-free fallback backbone (no pretrained-weight download
    required). Small enough to train on CPU in seconds on the synthetic set."""

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(128, num_classes)
        # Grad-CAM hooks into this named layer (last conv block)
        self.gradcam_layer_name = "features.12"  # the last Conv2d in `features`

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x).flatten(1)
        return self.classifier(x)


def build_model(architecture: str = "resnet18", num_classes: int = NUM_CLASSES, pretrained: bool = True):
    if architecture == "resnet18":
        try:
            weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            model = torchvision.models.resnet18(weights=weights)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            model.gradcam_layer_name = "layer4"
            return model
        except Exception as e:  # offline env, corrupted cache, etc.
            logger.warning("Falling back to SimpleCNN: could not load resnet18 weights (%s)", e)
            return SimpleCNN(num_classes)
    elif architecture == "simple_cnn":
        return SimpleCNN(num_classes)
    raise ValueError(f"Unknown architecture: {architecture}")


def load_trained_model(checkpoint_path: str, device: str = "cpu"):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = build_model(
        architecture=ckpt.get("architecture", "resnet18"),
        num_classes=ckpt.get("num_classes", NUM_CLASSES),
        pretrained=False,  # weights come from the checkpoint, not ImageNet
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    model.to(device)
    return model


def predict(model, tensor: torch.Tensor, device: str = "cpu"):
    """Returns (predicted_class:int, probs:torch.Tensor[5], confidence:float)."""
    model.eval()
    with torch.no_grad():
        tensor = tensor.to(device)
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        pred = int(torch.argmax(probs).item())
        confidence = float(probs[pred].item())
    return pred, probs, confidence
