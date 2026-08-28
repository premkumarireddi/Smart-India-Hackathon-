# ADR 0005: Synthetic fundus images for the proof-of-concept, real datasets documented separately

## Status
Accepted

## Context
Real training data for this problem — APTOS 2019, IDRiD, DRIVE, Messidor-2
— all require a Kaggle account and API token to download (see
`ml/download_data.py`). This dev environment has no such credentials
configured, and requiring them would mean the repository can't be cloned
and run end-to-end by an arbitrary reviewer without an extra manual signup
step succeeding first.

We still need the *entire* pipeline — data loading, augmentation, training
loop, checkpointing, evaluation, Grad-CAM, the API, and the UI — to be
runnable and demoable with a single command, with no external accounts.

## Decision
`ml/generate_synthetic_data.py` procedurally generates simple synthetic
"fundus-like" images (a retina disc, optic disc, vessel pattern, and a
severity-correlated number of lesion blobs) and trains the shipped demo
checkpoint on those. `ml/download_data.py` is provided, fully documented,
and ready to run the moment a user adds their own `kaggle.json`.

This is stated **loudly and repeatedly** in the README and in this ADR
specifically to avoid the single worst failure mode for a project like
this: presenting synthetic-data results as if they were validated clinical
performance. They are not. They prove the *pipeline* works; they say
nothing about real-world diabetic retinopathy detection accuracy.

## Consequences
- Positive: `git clone` -> `pip install -r requirements.txt` ->
  `python ml/generate_synthetic_data.py` -> `python ml/train.py` -> working
  demo, no external accounts, in minutes.
- Negative: the shipped checkpoint's metrics (see README "Honest status of
  the model") are not comparable to, and must never be quoted alongside,
  the PS26038 clinical targets (Sensitivity >90%, Specificity >85% on real
  patients). Every place these numbers are reported in this repo is labeled
  "synthetic-data proof-of-concept" to make that unmissable.
- Roadmap: swap `--data-dir` to point at a real downloaded dataset and
  rerun `ml/train.py` — no code changes required, since `FundusDataset`
  only assumes the `<class>/<image>` + `labels.csv` layout, not anything
  about where the images actually came from.

**Update:** this has since happened — the real APTOS 2019 dataset (3,662
images) was downloaded and trained on; `ml/prepare_aptos.py` converts
Kaggle's `train.csv` into the manifest format without copying any image
bytes. Both checkpoints ship in this repo: `retina_cnn_demo.pt` (synthetic,
always available, zero setup) and `retina_cnn_aptos.pt` (real data). See
the README's "The model — honest status" section for the real numbers, and
note the *new* honesty concern that replaces this one: real-data metrics
from a single 80/20 in-repo split are still not the same claim as
independently validated clinical sensitivity/specificity on a held-out
clinical population — see that section for exactly what is and isn't
being claimed.
