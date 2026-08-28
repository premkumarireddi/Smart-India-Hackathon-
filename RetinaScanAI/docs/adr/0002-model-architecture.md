# ADR 0002: ResNet18 transfer learning, with a from-scratch CNN fallback

## Status
Accepted

## Context
PS26038 calls for CNN transfer learning (EfficientNet/ResNet backbone) for
5-class DR severity grading. We need an architecture that:
- works well with a small amount of training data (transfer learning is
  specifically good at this),
- trains in a reasonable time on a CPU-only dev machine (no GPU assumed),
- has a clean, well-known layer to hook Grad-CAM into,
- doesn't hard-fail if the dev box has no internet access to download
  pretrained weights.

## Decision
Default architecture: `torchvision.models.resnet18` with ImageNet-pretrained
weights, final fully-connected layer replaced with a fresh 5-class head,
fine-tuned end-to-end. Grad-CAM hooks into `layer4` (the last conv block),
which is the standard choice in the Grad-CAM literature for ResNet family
models.

Fallback: if pretrained weights can't be downloaded, `build_model()`
automatically falls back to `SimpleCNN`, a 4-block conv net defined in
`app/pipeline/model.py`, trained from scratch. This keeps
`ml/train.py`, the test suite, and CI green even fully offline — at the
cost of lower accuracy, which is an acceptable tradeoff for a fallback path
that should rarely trigger in practice.

We did not choose EfficientNet (also named in the PS) as the *default*
because torchvision's EfficientNet-B0 forward/backward pass is
meaningfully slower per step on CPU than ResNet18 for a similar parameter
count, and this prototype has to train and run on CPU-only environments.
`build_model()` is written so swapping in EfficientNet-B0 is a ~5-line
change (see the function body) if GPU training becomes available.

## Consequences
- Positive: fast to train, robust to offline environments, one well-defined
  Grad-CAM target layer.
- Negative: ResNet18 is a smaller/older backbone than EfficientNet; a
  production model trained on real clinical data should re-evaluate
  EfficientNet-B3/B4 or a purpose-built retinal-imaging architecture once
  GPU training resources are available.
