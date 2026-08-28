"""
Downloads the REAL clinical datasets referenced in the SIH problem statement
(SIH26038) via the Kaggle API. This is the path to production-grade training
data — the synthetic generator (`generate_synthetic_data.py`) exists only
because this script needs credentials this dev environment doesn't have.

Setup (one-time):
    1. Create a Kaggle account -> https://www.kaggle.com
    2. Account -> Create New API Token -> downloads kaggle.json
    3. Place it at ~/.kaggle/kaggle.json  (chmod 600 on Linux/Mac)
    4. pip install kaggle
    5. Accept the competition rules at:
       https://www.kaggle.com/c/aptos2019-blindness-detection/rules
       (Kaggle requires this before the API will let you download)

Usage:
    python ml/download_data.py --dataset aptos --out data/raw/aptos2019
"""
import argparse
import subprocess
import sys
from pathlib import Path

DATASETS = {
    "aptos": {
        "kind": "competition",
        "slug": "aptos2019-blindness-detection",
        "note": "3,662 labeled fundus images, 0-4 DR severity. Requires accepting competition rules on the website first.",
    },
    "idrid": {
        "kind": "dataset",
        "slug": "aroraritik/idrid-dataset",  # community mirror; see README for the official IEEE DataPort source
        "note": "Indian Diabetic Retinopathy Image Dataset — lesion-level segmentation ground truth.",
    },
    "diabetic-retinopathy-224": {
        "kind": "dataset",
        "slug": "sovitrath/diabetic-retinopathy-224x224-2019-data",
        "note": "Pre-resized 224x224 APTOS-derived set, convenient for quick local experiments.",
    },
}


def check_kaggle_cli():
    try:
        subprocess.run(["kaggle", "--version"], check=True, capture_output=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("Kaggle CLI not found or not authenticated.\n"
              "Run: pip install kaggle   then place kaggle.json in ~/.kaggle/", file=sys.stderr)
        sys.exit(1)


def download(dataset_key: str, out_dir: Path):
    check_kaggle_cli()
    spec = DATASETS[dataset_key]
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading '{dataset_key}': {spec['note']}")

    if spec["kind"] == "competition":
        cmd = ["kaggle", "competitions", "download", "-c", spec["slug"], "-p", str(out_dir)]
    else:
        cmd = ["kaggle", "datasets", "download", "-d", spec["slug"], "-p", str(out_dir), "--unzip"]

    subprocess.run(cmd, check=True)
    print(f"Done -> {out_dir}")
    if dataset_key == "aptos":
        print("Next: unzip if needed, then run:")
        print(f"  python ml/prepare_aptos.py --aptos-dir {out_dir}")
        print(f"  python ml/train.py --data-dir {out_dir} --epochs 6 --arch resnet18 --batch-size 32")
    else:
        print("Next: point ml/train.py --data-dir at this folder (it must contain a labels.csv "
              "manifest in the <class>/<image> layout — write a small prepare_*.py like "
              "prepare_aptos.py if this dataset's raw layout differs).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS.keys(), default="aptos")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    out = args.out or Path(__file__).resolve().parents[1] / "data" / "raw" / args.dataset
    download(args.dataset, out)
