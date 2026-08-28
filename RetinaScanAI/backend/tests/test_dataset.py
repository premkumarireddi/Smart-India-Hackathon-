import shutil

import torch

from app.pipeline.preprocess import IMG_SIZE
from ml.dataset import FundusDataset


def test_dataset_length_matches_labels_csv(synthetic_data_dir):
    ds = FundusDataset(synthetic_data_dir, augment=False)
    assert len(ds) == 400  # 5 classes * 80 images (see conftest / generator defaults)


def test_dataset_item_shape_and_label_range(synthetic_data_dir):
    ds = FundusDataset(synthetic_data_dir, augment=False)
    tensor, label = ds[0]
    assert tensor.shape == (3, IMG_SIZE, IMG_SIZE)
    assert isinstance(label, torch.Tensor)
    assert 0 <= label.item() <= 4


def test_augmentation_does_not_crash_and_preserves_shape(synthetic_data_dir):
    ds = FundusDataset(synthetic_data_dir, augment=True)
    for i in [0, 50, 200, 399]:
        tensor, _ = ds[i]
        assert tensor.shape == (3, IMG_SIZE, IMG_SIZE)


def test_resize_cache_is_created_and_reused(tmp_path, synthetic_data_dir):
    # Copy a tiny slice of the synthetic set into a scratch dir so this test
    # doesn't pollute the committed dataset's cache and can assert on a
    # clean cache directory.
    scratch = tmp_path / "mini_dataset"
    shutil.copytree(synthetic_data_dir / "No_DR", scratch / "No_DR")
    shutil.copy(synthetic_data_dir / "labels.csv", scratch / "labels_full.csv")

    import csv
    with open(scratch / "labels_full.csv", newline="") as f_in, \
         open(scratch / "labels.csv", "w", newline="") as f_out:
        reader = csv.DictReader(f_in)
        writer = csv.writer(f_out)
        writer.writerow(["filename", "label", "label_name"])
        for row in reader:
            if row["label_name"] == "No_DR":
                writer.writerow([row["filename"], row["label"], row["label_name"]])

    cache_dir = scratch / ".cache128"
    assert not cache_dir.exists()

    ds = FundusDataset(scratch, augment=False, use_cache=True)
    ds[0]  # first access: cache miss -> writes cache file
    assert (cache_dir / "0.png").exists()

    # Second access should hit the cache path without erroring and return
    # a consistent shape.
    tensor_again, _ = ds[0]
    assert tensor_again.shape == (3, IMG_SIZE, IMG_SIZE)


def test_use_cache_false_does_not_create_cache_dir(tmp_path, synthetic_data_dir):
    scratch = tmp_path / "no_cache_dataset"
    shutil.copytree(synthetic_data_dir / "Mild", scratch / "Mild")
    import csv
    with open(synthetic_data_dir / "labels.csv", newline="") as f_in, \
         open(scratch / "labels.csv", "w", newline="") as f_out:
        reader = csv.DictReader(f_in)
        writer = csv.writer(f_out)
        writer.writerow(["filename", "label", "label_name"])
        for row in reader:
            if row["label_name"] == "Mild":
                writer.writerow([row["filename"], row["label"], row["label_name"]])

    ds = FundusDataset(scratch, augment=False, use_cache=False)
    ds[0]
    assert not (scratch / ".cache128").exists()
