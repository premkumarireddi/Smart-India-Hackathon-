# ADR 0001: Implement the prototype in Python (PyTorch/FastAPI), not MATLAB

## Status
Accepted

## Context
PS26038 explicitly asks for a MATLAB-based pipeline using the Image
Processing, Computer Vision, Deep Learning, Medical Imaging, and
Statistics/ML Toolboxes, plus Simulink for workflow modeling. Our pitch
deck (slide 3) describes the solution in those terms, matching the official
problem statement's expected tooling.

For this repository, though, the goal is a prototype that:
- runs on any contributor's machine without a paid MATLAB/Simulink license,
- can be checked into GitHub and run in ordinary CI (GitHub Actions has no
  MATLAB runner without extra licensing cost),
- is easy for a judging panel or professor to `git clone` and run in one command.

## Decision
Build the prototype in Python: PyTorch + torchvision for the CNN, OpenCV for
image processing (fills the same role as the Image Processing / Computer
Vision Toolboxes), FastAPI for the serving layer, and a hand-written
Erlang-C queueing model in `simulation/` standing in for the Simulink
workflow simulation (see ADR 0004).

Every pipeline stage named in the problem statement is still implemented —
quality assessment, structure segmentation, CNN classification, Grad-CAM
explainability, confidence calibration, annotated reporting, and
throughput simulation — just with open-source, license-free tooling.

## Consequences
- Positive: zero-cost, zero-license reproducibility; runs in GitHub Actions;
  matches the skill set most contributors already have.
- Negative: diverges from the literal toolchain named in the PS. If this
  progresses toward an actual SIH submission where MATLAB is a scoring
  criterion, a MATLAB port (or a hybrid: MATLAB Coder-exported ONNX model
  loaded here) is a documented extension — see README "Roadmap".
- The `simulation/` module's math (Erlang-C) is the same underlying model a
  Simulink discrete-event simulation would numerically approximate, so
  porting the *logic* later is low-risk even if the *tool* changes.
