"""
Converts a downloaded APTOS 2019 Kaggle folder (train.csv + train_images/)
into the labels.csv manifest format FundusDataset expects — WITHOUT copying
any image bytes. APTOS images are full-resolution (avg. several MB each,
~8GB total for the training set), so this just writes a small CSV of
relative paths pointing at the images where Kaggle put them.

Usage:
    python ml/prepare_aptos.py --aptos-dir data/raw/aptos2019

After this, train directly against the real data:
    python ml/train.py --data-dir data/raw/aptos2019 --epochs 6 --arch resnet18
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from app.pipeline.model import CLASS_NAMES  # noqa: E402


def prepare(aptos_dir: Path):
    train_csv = aptos_dir / "train.csv"
    images_dir = aptos_dir / "train_images"
    if not train_csv.exists():
        raise FileNotFoundError(f"{train_csv} not found — expected the standard Kaggle APTOS layout")

    rows = [("filename", "label", "label_name")]
    counts = [0] * len(CLASS_NAMES)
    missing = 0

    with open(train_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = row["id_code"]
            label = int(row["diagnosis"])
            rel_path = f"train_images/{image_id}.png"
            if not (images_dir / f"{image_id}.png").exists():
                missing += 1
                continue
            rows.append((rel_path, label, CLASS_NAMES[label]))
            counts[label] += 1

    out_csv = aptos_dir / "labels.csv"
    with open(out_csv, "w", newline="") as f:
        csv.writer(f).writerows(rows)

    print(f"Wrote {len(rows) - 1} labeled entries -> {out_csv}")
    if missing:
        print(f"WARNING: {missing} rows in train.csv had no matching image file and were skipped")
    print("\nClass distribution (this is the REAL, imbalanced APTOS distribution —")
    print("train.py's stratified_split still balances val-set sampling per class):")
    for name, c in zip(CLASS_NAMES, counts):
        print(f"  {name:>18}: {c:>5}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    repo_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--aptos-dir", type=Path, default=repo_root / "data" / "raw" / "aptos2019")
    args = parser.parse_args()
    prepare(args.aptos_dir)
