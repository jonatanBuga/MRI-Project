from __future__ import annotations

import sys
from pathlib import Path

# Allow running via: python scripts/test_dataset_index.py
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from data.mri_seg_dataset import MRISegmentationDataset  # noqa: E402


def _print_sample(tag: str, sample: dict) -> None:
    print(f"\n[{tag}]")
    print("image_path:", sample.get("image_path"))
    print("mask_path: ", sample.get("mask_path"))
    print("has_mask:  ", sample.get("has_mask"))
    print("meta:      ", sample.get("meta"))


def _assert_sample_ok(sample: dict) -> None:
    image_path = sample.get("image_path")
    mask_path = sample.get("mask_path")
    has_mask = sample.get("has_mask")

    assert isinstance(image_path, Path), f"image_path should be a Path, got: {type(image_path)}"
    assert image_path.exists(), f"Image path does not exist: {image_path}"

    assert has_mask is True, f"Expected has_mask=True in labeled_only mode, got: {has_mask!r}"
    assert isinstance(mask_path, Path), f"mask_path should be a Path, got: {type(mask_path)}"
    assert mask_path.exists(), f"Mask path does not exist: {mask_path}"


def main() -> int:
    ds = MRISegmentationDataset(
        metadata_csv="metadata/metadata_labeled_roboflow.csv",
        project_root=REPO_ROOT,
        mode="labeled_only",
    )

    print("Total samples:", len(ds))
    if hasattr(ds, "summary"):
        print("Summary:", ds.summary())

    assert len(ds) > 0, "Dataset is empty. Check metadata/metadata_labeled_roboflow.csv"

    first = ds[0]
    _print_sample("first", first)
    _assert_sample_ok(first)

    # Random-ish sample: index 100 if available, otherwise last sample.
    idx = 100 if len(ds) > 100 else (len(ds) - 1)
    other = ds[idx]
    _print_sample(f"index {idx}", other)
    _assert_sample_ok(other)

    print("\nStage 1 dataset index looks OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())