"""
PyTorch Dataset for fundus images laid out as:
    <root>/<ClassName>/<image>.png
    <root>/labels.csv   (filename,label,label_name,...)

Works unmodified for both the synthetic generator output and the real
Kaggle datasets (e.g. after ml/prepare_aptos.py), as long as they resolve
to `root / filename`.

Resize caching: real datasets like APTOS ship full-resolution photos
(3216x2136, several MB each). Re-decoding and resizing that on every single
epoch is the dominant training cost by far. On first access, each image's
resize_square() output (i.e. pre-augmentation, pre-CLAHE) is cached to
`<root>/.cache128/<index>.png` — a small, fast-decoding file — so every
epoch after the first only pays the cheap augmentation + CLAHE cost, not a
full-resolution JPEG/PNG decode. Harmless (and low-cost) for the small
synthetic images too; the cache directory is git-ignored either way.
"""
import csv
from pathlib import Path

import cv2
import torch
from torch.utils.data import Dataset

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from app.pipeline.preprocess import IMG_SIZE, denoise, clahe_enhance, resize_square, to_model_tensor  # noqa: E402


class FundusDataset(Dataset):
    def __init__(self, root: Path, augment: bool = False, use_cache: bool = True):
        self.root = Path(root)
        self.augment = augment
        self.use_cache = use_cache
        self.cache_dir = self.root / ".cache128"
        if self.use_cache:
            self.cache_dir.mkdir(exist_ok=True)
        self.samples = []  # list of (path, label)

        labels_csv = self.root / "labels.csv"
        with open(labels_csv, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append((self.root / row["filename"], int(row["label"])))

    def __len__(self):
        return len(self.samples)

    def _load_resized_base(self, idx: int, path: Path) -> "cv2.Mat":
        """Returns the resize_square() output for sample `idx`, using the
        on-disk cache when available/enabled."""
        cache_path = self.cache_dir / f"{idx}.png"
        if self.use_cache and cache_path.exists():
            cached = cv2.imread(str(cache_path))
            if cached is not None:
                return cached

        img_bgr = cv2.imread(str(path))
        if img_bgr is None:
            raise FileNotFoundError(path)
        resized = resize_square(img_bgr, IMG_SIZE)

        if self.use_cache:
            cv2.imwrite(str(cache_path), resized)
        return resized

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img_bgr = self._load_resized_base(idx, path)

        if self.augment:
            img_bgr = _augment(img_bgr)

        # denoise + CLAHE only (resize already done, possibly via cache)
        enhanced = clahe_enhance(denoise(img_bgr))
        tensor = to_model_tensor(enhanced).squeeze(0)  # drop the batch dim added for inference
        return tensor, torch.tensor(label, dtype=torch.long)


def _augment(img_bgr):
    import random
    import cv2

    if random.random() < 0.5:
        img_bgr = cv2.flip(img_bgr, 1)
    if random.random() < 0.5:
        angle = random.uniform(-15, 15)
        h, w = img_bgr.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        img_bgr = cv2.warpAffine(img_bgr, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    return img_bgr
