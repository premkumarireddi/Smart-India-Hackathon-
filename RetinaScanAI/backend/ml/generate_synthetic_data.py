"""
Synthetic fundus-image generator for pipeline development and CI testing.

WHY THIS EXISTS
----------------
This repository ships without the real APTOS 2019 / IDRiD / DRIVE / Messidor-2
datasets because they require a Kaggle account + API token (see
`ml/download_data.py`), which is not available in every dev/CI environment.

To keep the *entire* pipeline runnable end-to-end out of the box (data ->
training -> inference -> Grad-CAM -> API -> UI) without any external
credentials, this script procedurally generates simple synthetic
"fundus-like" images: a circular retina disc with a vessel-like radial
pattern, and, for higher severity labels, a growing number of small dark
"lesion" blobs (standing in for microaneurysms / hemorrhages / exudates).

These images are NOT clinically meaningful. A model trained on them proves
the pipeline works end to end (data loading, training loop, checkpointing,
Grad-CAM, quality gating, API serving) — it says nothing about real-world
diabetic retinopathy detection performance. See docs/adr/0005 and the
README "Honest status of the model" section.

Usage:
    python ml/generate_synthetic_data.py --out data/synthetic --n-per-class 80
"""
import argparse
import csv
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

CLASS_NAMES = {
    0: "No_DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative_DR",
}

IMG_SIZE = 224


def _draw_retina_base(rng: random.Random) -> Image.Image:
    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (10, 5, 5))
    draw = ImageDraw.Draw(img)
    cx, cy = IMG_SIZE // 2, IMG_SIZE // 2
    r = IMG_SIZE // 2 - 6

    # retina disc base color (orange/red, like a real fundus photo)
    base = (
        170 + rng.randint(-15, 15),
        70 + rng.randint(-15, 15),
        45 + rng.randint(-10, 10),
    )
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=base)

    # optic disc (bright yellowish circle, off-center)
    od_r = r // 5
    od_cx = cx + rng.randint(-r // 3, -r // 5)
    od_cy = cy + rng.randint(-r // 6, r // 6)
    draw.ellipse(
        [od_cx - od_r, od_cy - od_r, od_cx + od_r, od_cy + od_r],
        fill=(235, 200, 130),
    )

    # vessel-like radial branching lines from the optic disc
    for _ in range(14):
        angle = rng.uniform(0, 2 * np.pi)
        length = rng.uniform(r * 0.5, r * 0.95)
        x2 = od_cx + length * np.cos(angle)
        y2 = od_cy + length * np.sin(angle)
        width = rng.choice([1, 1, 2])
        draw.line([(od_cx, od_cy), (x2, y2)], fill=(110, 30, 25), width=width)

    return img, (cx, cy, r)


def _add_lesions(img: Image.Image, geom, severity: int, rng: random.Random):
    """Severity 0 = none, up to 4 = many + larger, denser lesions."""
    if severity == 0:
        return img
    cx, cy, r = geom
    draw = ImageDraw.Draw(img)
    n_lesions = severity * rng.randint(3, 6)
    for _ in range(n_lesions):
        angle = rng.uniform(0, 2 * np.pi)
        dist = rng.uniform(0.15, 0.85) * r
        lx = cx + dist * np.cos(angle)
        ly = cy + dist * np.sin(angle)
        lr = rng.uniform(1.5, 2.0 + severity * 1.3)
        color = rng.choice([(60, 10, 10), (40, 5, 5), (200, 170, 60)])
        draw.ellipse([lx - lr, ly - lr, lx + lr, ly + lr], fill=color)
    if severity >= 3:
        img = img.filter(ImageFilter.GaussianBlur(radius=0.4))
    return img


def _apply_quality_defect(img: Image.Image, rng: random.Random, defect: str) -> Image.Image:
    """Optionally degrade an image so the quality-gate stage has real work to do."""
    if defect == "blur":
        img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(3, 6)))
    elif defect == "dark":
        arr = np.asarray(img).astype(np.float32) * rng.uniform(0.25, 0.4)
        img = Image.fromarray(arr.clip(0, 255).astype(np.uint8))
    elif defect == "bright":
        arr = np.asarray(img).astype(np.float32)
        arr = arr + (255 - arr) * rng.uniform(0.5, 0.75)
        img = Image.fromarray(arr.clip(0, 255).astype(np.uint8))
    return img


def generate(out_dir: Path, n_per_class: int, seed: int = 42, defect_fraction: float = 0.06):
    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [("filename", "label", "label_name", "synthetic_quality_defect")]

    for label, name in CLASS_NAMES.items():
        class_dir = out_dir / name
        class_dir.mkdir(exist_ok=True)
        for i in range(n_per_class):
            img, geom = _draw_retina_base(rng)
            img = _add_lesions(img, geom, label, rng)

            defect = "none"
            if rng.random() < defect_fraction:
                defect = rng.choice(["blur", "dark", "bright"])
                img = _apply_quality_defect(img, rng, defect)

            fname = f"{name}_{i:04d}.png"
            img.save(class_dir / fname)
            rows.append((f"{name}/{fname}", label, name, defect))

    with open(out_dir / "labels.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows)

    total = n_per_class * len(CLASS_NAMES)
    print(f"Generated {total} synthetic images ({n_per_class} per class) -> {out_dir}")
    print(f"Label manifest -> {out_dir / 'labels.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "synthetic")
    parser.add_argument("--n-per-class", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate(args.out, args.n_per_class, args.seed)
