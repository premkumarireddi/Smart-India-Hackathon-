# ADR 0003: Grad-CAM, implemented from scratch, for explainability

## Status
Accepted

## Context
The entire point of PS26038 (and the pitch's differentiator) is that a
prediction alone isn't good enough — a clinician needs to see *why* the
model flagged an image, verifiable in under 30 seconds. We need a
lesion-level visual explanation method that:
- works with any CNN backbone without architectural changes,
- is well-established / peer-reviewed (not something invented for this repo),
- is cheap enough to compute per-request in an interactive API.

## Decision
Implement Grad-CAM (Selvaraju et al., 2017) directly with PyTorch forward
and backward hooks (`app/pipeline/gradcam.py`), rather than pulling in a
third-party explainability library (e.g. `pytorch-grad-cam`).

Rationale for hand-rolling it: Grad-CAM is ~40 lines of well-understood
math (global-average-pool the gradients flowing into the target conv layer,
weight the activation maps by that, ReLU, upsample). Implementing it
directly means:
- one fewer third-party dependency to pin/audit,
- the code is transparent and auditable by a reviewer without them having
  to go read someone else's library source,
- it's trivial to extend later (e.g. Grad-CAM++ or Score-CAM) without being
  boxed in by a library's API.

## Consequences
- Positive: no extra dependency, full control, easy to explain in a demo
  ("here's literally the 40 lines that make the heatmap").
- Negative: we own maintenance of this code instead of a library community;
  acceptable for a prototype at this scale.
- The `GradCAM` class targets `model.gradcam_layer_name`, a convention set
  on each architecture in `model.py` — swapping backbones (per ADR 0002)
  requires setting this attribute correctly, which is intentionally
  explicit rather than auto-detected, to avoid silently pointing Grad-CAM
  at the wrong layer.
