from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.utils.config import TARGET_SIZE


def _infer_repo_root_from_here() -> Path:
    """
    This file lives at: <repo_root>/src/data/mri_seg_dataset.py
    So repo root is parents[2].
    """
    return Path(__file__).resolve().parents[2]


def _parse_bool(value: object) -> bool:
    """
    Accept common CSV boolean encodings.
    """
    if value is None:
        return False
    s = str(value).strip().lower()
    return s in {"1", "true", "t", "yes", "y"}


def load_and_preprocess_mask(mask_path: Path, target_size: int | tuple[int, int]) -> "torch.Tensor":
    """
    Load a binary mask PNG and preprocess it to match image preprocessing.

    - Reads as grayscale
    - Binarizes to {0,1}
    - Resizes to target_size using NEAREST interpolation only
    - Returns torch.long tensor with shape [H, W] and values in {0,1}

    Note: Imports torch/numpy/PIL locally so Stage 1 indexing can still run without torch.
    """
    import numpy as np
    import torch
    from PIL import Image

    if isinstance(target_size, int):
        out_w, out_h = target_size, target_size
    else:
        out_h, out_w = int(target_size[0]), int(target_size[1])

    img = Image.open(mask_path).convert("L")
    arr = np.asarray(img, dtype=np.uint8)

    # Binarize: any non-zero value becomes 1
    arr = (arr > 0).astype(np.uint8)

    # Resize with NEAREST only (preserve labels)
    try:
        # Pillow >= 9
        arr_img = Image.fromarray(arr, mode="L").resize((out_w, out_h), resample=Image.Resampling.NEAREST)
    except AttributeError:
        # Older Pillow
        arr_img = Image.fromarray(arr, mode="L").resize((out_w, out_h), resample=Image.NEAREST)

    arr_resized = np.asarray(arr_img, dtype=np.uint8)
    arr_resized = (arr_resized > 0).astype(np.uint8)  # keep strict {0,1} after resize

    mask_tensor = torch.from_numpy(arr_resized).to(dtype=torch.long)  # [H, W]
    return mask_tensor


class MRISegmentationDataset:
    """
    Stage 0/1 dataset: indexes Roboflow image/mask pairs from a labeled metadata CSV.

    - Reads metadata (repo-root-relative paths)
    - Builds self.samples (stable order = CSV order)

    Stage 2: image/mask loading/preprocessing is introduced (torch + transforms) in __getitem__ only,
    to avoid importing heavy deps at module import time.
    """

    def __init__(
        self,
        metadata_csv: str = "metadata/metadata_labeled_roboflow_all.csv",
        project_root: Optional[Union[str, Path]] = None,
        mode: str = "labeled_only",
        split: Optional[str] = None,
    ) -> None:
        self.project_root = Path(project_root) if project_root is not None else _infer_repo_root_from_here()
        self.metadata_csv = (self.project_root / metadata_csv).resolve()

        if mode not in {"labeled_only", "mixed"}:
            raise ValueError(f"mode must be 'labeled_only' or 'mixed', got: {mode!r}")
        self.mode = mode

        if split not in {None, "train", "val", "test"}:
            raise ValueError(f"split must be one of None/'train'/'val'/'test', got: {split!r}")
        self.split = split

        if not self.metadata_csv.exists():
            raise FileNotFoundError(f"Metadata CSV not found: {self.metadata_csv.as_posix()}")

        self.samples: List[Dict[str, Any]] = []
        self._build_index()

    def _build_index(self) -> None:
        required_cols = {"image_path", "mask_path", "has_label", "mask_valid"}

        with self.metadata_csv.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError(f"Metadata CSV has no header: {self.metadata_csv.as_posix()}")

            fieldnames = set(reader.fieldnames)
            missing = required_cols.difference(fieldnames)
            if missing:
                raise ValueError(
                    f"Metadata CSV missing required columns {sorted(missing)}. "
                    f"Found: {reader.fieldnames}. File: {self.metadata_csv.as_posix()}"
                )

            has_split_col = "split" in fieldnames
            if not has_split_col and self.split not in {None, "train"}:
                raise ValueError(
                    "This metadata CSV has no 'split' column, so only split=None or split='train' is allowed. "
                    f"Got split={self.split!r}. File: {self.metadata_csv.as_posix()}"
                )

            for i, row in enumerate(reader):
                image_rel = (row.get("image_path") or "").strip()
                mask_rel = (row.get("mask_path") or "").strip()
                has_label = _parse_bool(row.get("has_label"))
                mask_valid = _parse_bool(row.get("mask_valid"))

                # Basic guard: skip rows with no image path
                if not image_rel:
                    continue

                row_split = (row.get("split") or "").strip() if has_split_col else "train"
                if self.split is not None:
                    if row_split != self.split:
                        continue

                # Filtering rules
                if self.mode == "labeled_only":
                    if not (has_label and mask_valid):
                        continue
                    if not mask_rel:
                        continue

                image_abs = (self.project_root / Path(image_rel)).resolve()
                mask_abs: Optional[Path] = (self.project_root / Path(mask_rel)).resolve() if mask_rel else None
                has_mask = mask_abs is not None

                # Sanity checks (Stage 0/1)
                assert image_abs.exists(), f"Image file not found on disk: {image_abs.as_posix()}"
                if has_mask:
                    assert mask_abs is not None
                    assert mask_abs.exists(), f"Mask file not found on disk: {mask_abs.as_posix()}"

                self.samples.append(
                    {
                        "image_path": image_abs,
                        "mask_path": mask_abs,
                        "has_mask": has_mask,
                        "meta": {
                            "source": "roboflow",
                            "row_index": i,
                            "split": row_split,
                        },
                    }
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        # Keep indexing logic unchanged: self.samples is the stable index
        sample = self.samples[index]

        image_path: Path = sample["image_path"]
        mask_path: Optional[Path] = sample.get("mask_path")

        # Stage 2 (step 1): load + preprocess the image using existing pipeline.
        # Import here to avoid importing torch / image libs at module import time.
        try:
            from src.data.transforms import preprocess_image  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "Failed to import preprocess_image from src/data/transforms.py. "
                "Stage 2 requires torch + the preprocessing pipeline to be available."
            ) from e

        if mask_path is None:
            raise ValueError(
                "mask_path is missing for this sample. "
                "In 'labeled_only' mode this indicates invalid metadata."
            )

        image_tensor = preprocess_image(image_path)
        mask_tensor = load_and_preprocess_mask(mask_path, TARGET_SIZE)

        # Sanity: ensure mask spatial size matches image spatial size
        # image_tensor expected shape: [C, H, W]
        if hasattr(image_tensor, "shape") and len(getattr(image_tensor, "shape", ())) == 3:
            _, h, w = image_tensor.shape
            assert tuple(mask_tensor.shape) == (h, w), (
                f"Mask size {tuple(mask_tensor.shape)} does not match image size {(h, w)} "
                f"for image={image_path.as_posix()} mask={mask_path.as_posix()}"
            )

        meta = dict(sample.get("meta", {}))
        meta.update(
            {
                "source": "roboflow",
                "row_index": int(meta.get("row_index", index)),
                "split": str(meta.get("split", "train")),
                "image_path": str(image_path),
                "mask_path": str(mask_path),
            }
        )

        # Standardized training output (stable API)
        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "meta": meta,
        }

    def output_keys(self) -> List[str]:
        return ["image", "mask", "meta"]

    def summary(self) -> Dict[str, int]:
        total = len(self.samples)
        with_mask = sum(1 for s in self.samples if s.get("has_mask", False))
        without_mask = total - with_mask
        return {
            "total": total,
            "with_mask": with_mask,
            "without_mask": without_mask,
        }


if __name__ == "__main__":
    # Smoke test for split filtering using merged Roboflow metadata.
    import torch

    ds_train = MRISegmentationDataset(split="train")
    ds_val = MRISegmentationDataset(split="val")
    ds_test = MRISegmentationDataset(split="test")

    print("train:", len(ds_train))
    print("val:  ", len(ds_val))
    print("test: ", len(ds_test))

    total = len(ds_train) + len(ds_val) + len(ds_test)
    print("total:", total)
    assert total == 825, f"Expected total=825, got {total}"

    for tag, ds in [("train", ds_train), ("val", ds_val), ("test", ds_test)]:
        if len(ds) == 0:
            raise RuntimeError(f"{tag} split dataset is empty")
        s = ds[0]
        img = s["image"]
        msk = s["mask"]
        uniq = torch.unique(msk).cpu().tolist()
        print(f"[{tag}] image:", tuple(img.shape), img.dtype, "mask:", tuple(msk.shape), msk.dtype, "unique:", uniq)