"""
Trains the DR severity classifier and writes a checkpoint to backend/models/.

    python ml/train.py --data-dir data/synthetic --epochs 8 --arch resnet18

Splits the dataset 80/20 train/val (stratified by class via a fixed seed),
trains with cross-entropy + Adam, and reports per-class metrics (precision,
recall/sensitivity, specificity) plus the referable-DR (ICDR Level 2+)
sensitivity/specificity called out explicitly in PS26038's success criteria
(>90% sensitivity, >85% specificity — see README for how the demo model's
synthetic-data numbers compare to that clinical target, and why they're not
the same thing).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

sys.path.append(str(Path(__file__).resolve().parents[1]))
from app.pipeline.model import build_model, CLASS_NAMES, NUM_CLASSES, REFERABLE_THRESHOLD  # noqa: E402
from ml.dataset import FundusDataset  # noqa: E402


def stratified_split(dataset, val_fraction=0.2, seed=42):
    rng = np.random.RandomState(seed)
    labels = np.array([label for _, label in dataset.samples])
    train_idx, val_idx = [], []
    for c in range(NUM_CLASSES):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        n_val = max(1, int(len(idx) * val_fraction))
        val_idx.extend(idx[:n_val])
        train_idx.extend(idx[n_val:])
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(y.numpy().tolist())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    per_class = {}
    for c in range(NUM_CLASSES):
        tp = int(((all_preds == c) & (all_labels == c)).sum())
        fp = int(((all_preds == c) & (all_labels != c)).sum())
        fn = int(((all_preds != c) & (all_labels == c)).sum())
        tn = int(((all_preds != c) & (all_labels != c)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        per_class[CLASS_NAMES[c]] = {
            "precision": round(precision, 3),
            "recall_sensitivity": round(recall, 3),
            "specificity": round(specificity, 3),
            "support": int((all_labels == c).sum()),
        }

    accuracy = float((all_preds == all_labels).mean())

    # Referable DR (ICDR Level 2+) binary metrics — the clinically-relevant
    # threshold called out in PS26038's success criteria.
    ref_pred = all_preds >= REFERABLE_THRESHOLD
    ref_true = all_labels >= REFERABLE_THRESHOLD
    tp = int((ref_pred & ref_true).sum())
    fn = int((~ref_pred & ref_true).sum())
    tn = int((~ref_pred & ~ref_true).sum())
    fp = int((ref_pred & ~ref_true).sum())
    referable_sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    referable_specificity = tn / (tn + fp) if (tn + fp) else 0.0

    return {
        "accuracy": round(accuracy, 3),
        "per_class": per_class,
        "referable_dr_sensitivity": round(referable_sensitivity, 3),
        "referable_dr_specificity": round(referable_specificity, 3),
        "n_val_samples": len(all_labels),
    }


def train(data_dir: Path, epochs: int, arch: str, batch_size: int, lr: float, out_path: Path, log_path: Path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    full_dataset = FundusDataset(data_dir, augment=False)
    train_ds, val_ds = stratified_split(full_dataset)
    # augmentation only makes sense re-wrapped around the train subset's
    # underlying dataset; simplest correct approach for this repo's scale:
    train_ds.dataset = FundusDataset(data_dir, augment=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = build_model(architecture=arch, pretrained=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Inverse-frequency class weighting: real DR datasets (APTOS included)
    # are heavily imbalanced toward No_DR, and the clinically critical
    # classes (Severe, Proliferative_DR) are the rarest. Unweighted
    # cross-entropy would happily learn to under-call exactly the cases
    # that matter most for referral. Weights are computed from the TRAIN
    # split only (never validation) to avoid leaking val-set statistics
    # into training.
    train_labels = [full_dataset.samples[i][1] for i in train_ds.indices]
    class_counts = np.array([train_labels.count(c) for c in range(NUM_CLASSES)], dtype=np.float32)
    class_weights = class_counts.sum() / (NUM_CLASSES * np.maximum(class_counts, 1))
    print(f"Class counts (train split): {class_counts.tolist()}")
    print(f"Class weights (inverse-frequency): {np.round(class_weights, 2).tolist()}")
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32).to(device))

    history = []
    best_score = -1.0
    best_state = None
    best_metrics = None
    best_epoch = None
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)

        train_loss = running_loss / len(train_ds)
        metrics = evaluate(model, val_loader, device)

        # Model-selection score: mean of referable-DR sensitivity and
        # specificity, since that ICDR-Level-2+ threshold is the clinically
        # relevant one called out in PS26038 — not raw 5-class accuracy,
        # which would happily pick a checkpoint that's great at telling
        # "No DR" from "Mild" but sloppy exactly at the referral boundary.
        score = (metrics["referable_dr_sensitivity"] + metrics["referable_dr_specificity"]) / 2
        is_best = score > best_score
        if is_best:
            best_score = score
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_metrics = metrics
            best_epoch = epoch

        print(f"Epoch {epoch}/{epochs}  train_loss={train_loss:.4f}  "
              f"val_accuracy={metrics['accuracy']:.3f}  "
              f"referable_sensitivity={metrics['referable_dr_sensitivity']:.3f}  "
              f"referable_specificity={metrics['referable_dr_specificity']:.3f}"
              f"{'  <- best so far' if is_best else ''}")
        history.append({"epoch": epoch, "train_loss": round(train_loss, 4), "is_best": is_best, **metrics})

    elapsed = time.time() - t0
    final_metrics = best_metrics
    model.load_state_dict(best_state)  # restore best checkpoint before saving/serving

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": best_state,
        "architecture": arch,
        "num_classes": NUM_CLASSES,
        "class_names": CLASS_NAMES,
        "final_metrics": final_metrics,
        "trained_on": str(data_dir),
        "epochs": epochs,
        "best_epoch": best_epoch,
        "training_seconds": round(elapsed, 1),
    }, out_path)
    print(f"\nBest epoch selected by mean(referable sensitivity, referable specificity) = {best_score:.3f}")
    print(f"Saved checkpoint -> {out_path}")

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        json.dump({"history": history, "training_seconds": round(elapsed, 1), "device": device}, f, indent=2)
    print(f"Saved training log -> {log_path}")

    return final_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    repo_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--data-dir", type=Path, default=repo_root / "data" / "synthetic")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--arch", choices=["resnet18", "simple_cnn"], default="resnet18")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--out", type=Path, default=repo_root / "models" / "retina_cnn_demo.pt")
    parser.add_argument("--log", type=Path, default=repo_root / "models" / "training_log.json")
    args = parser.parse_args()

    train(args.data_dir, args.epochs, args.arch, args.batch_size, args.lr, args.out, args.log)
