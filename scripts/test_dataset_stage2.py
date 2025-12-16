from __future__ import annotations

import sys
from pathlib import Path

import torch

# Allow running via: python scripts/test_dataset_stage2.py
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from data.mri_seg_dataset import MRISegmentationDataset  # noqa: E402


def _short_path(p: str, max_len: int = 120) -> str:
    s = str(p)
    if len(s) <= max_len:
        return s
    return "..." + s[-(max_len - 3) :]


def _print_tensor_stats(tag: str, t: torch.Tensor) -> None:
    print(f"{tag}: shape={tuple(t.shape)} dtype={t.dtype} device={t.device}")
    if t.numel() == 0:
        print(f"{tag}: empty tensor")
        return
    # min/max on float is fine; on long is also fine
    print(f"{tag}: min={t.min().item()} max={t.max().item()}")


def _assert_sample_ok(sample: dict) -> None:
    assert set(sample.keys()) == {"image", "mask", "meta"}, f"Unexpected keys: {set(sample.keys())}"

    image = sample["image"]
    mask = sample["mask"]
    meta = sample["meta"]

    assert isinstance(image, torch.Tensor), f"image should be torch.Tensor, got: {type(image)}"
    assert image.dtype == torch.float32, f"image dtype must be torch.float32, got: {image.dtype}"
    assert image.ndim == 3, f"image must be 3D [C,H,W], got ndim={image.ndim}"
    assert image.shape[0] == 3, f"image must have 3 channels, got C={image.shape[0]}"

    assert isinstance(mask, torch.Tensor), f"mask should be torch.Tensor, got: {type(mask)}"
    assert mask.dtype == torch.long, f"mask dtype must be torch.long, got: {mask.dtype}"
    assert mask.ndim == 2, f"mask must be 2D [H,W], got ndim={mask.ndim}"

    _, h, w = image.shape
    assert tuple(mask.shape) == (h, w), f"mask shape {tuple(mask.shape)} != image spatial {(h, w)}"

    uniq = torch.unique(mask).cpu().tolist()
    assert set(uniq).issubset({0, 1}), f"mask unique values must be subset of {{0,1}}, got: {uniq}"

    assert isinstance(meta, dict), f"meta should be dict, got: {type(meta)}"
    assert "image_path" in meta and "mask_path" in meta, "meta must contain image_path and mask_path"


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

    for i in [0, 1, 100]:
        if i >= len(ds):
            print(f"\n[index {i}] (skipped: out of range)")
            continue

        sample = ds[i]
        image = sample["image"]
        mask = sample["mask"]
        meta = sample["meta"]

        print(f"\n[index {i}]")
        _print_tensor_stats("image", image)
        _print_tensor_stats("mask ", mask)
        print("mask unique:", torch.unique(mask).cpu().tolist())
        print("meta.image_path:", _short_path(meta.get("image_path", "")))
        print("meta.mask_path: ", _short_path(meta.get("mask_path", "")))

        _assert_sample_ok(sample)

    print("\nStage 2 OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())