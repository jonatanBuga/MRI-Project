from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset


def _infer_repo_root_from_here() -> Path:
    """
    This file lives at: <repo_root>/src/data/mri_inference_dataset.py
    So repo root is parents[2].
    """
    return Path(__file__).resolve().parents[2]


class MRIInferenceDataset(Dataset):
    """
    Lightweight inference-only dataset:
      - loads images
      - applies the same preprocessing as training (preprocess_image)
      - does not require masks/labels
    """

    def __init__(self, image_paths: list[str] | None = None, root_dir: str | None = None):
        if image_paths is None and root_dir is None:
            raise ValueError("Provide either image_paths or root_dir.")

        repo_root = _infer_repo_root_from_here()

        paths: List[Path] = []
        if image_paths is not None:
            for p in image_paths:
                pp = Path(p)
                if not pp.is_absolute():
                    pp = (repo_root / pp).resolve()
                else:
                    pp = pp.resolve()
                paths.append(pp)
        else:
            assert root_dir is not None
            root = Path(root_dir)
            if not root.is_absolute():
                root = (repo_root / root).resolve()
            else:
                root = root.resolve()

            exts = {".png", ".jpg", ".jpeg"}
            for p in root.rglob("*"):
                if p.is_file() and p.suffix.lower() in exts:
                    paths.append(p.resolve())

        # Stable sorted list
        self.image_paths: List[Path] = sorted(paths, key=lambda x: x.as_posix())

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path = self.image_paths[idx]

        # Import here so only inference usage pulls in preprocessing deps.
        from src.data.transforms import preprocess_image  # type: ignore

        image_tensor = preprocess_image(path)
        return {
            "image": image_tensor,
            "meta": {
                "image_path": str(path),
                "index": idx,
            },
        }


if __name__ == "__main__":
    ds = MRIInferenceDataset(root_dir="data/roboflow_coco/test")
    print("len:", len(ds))
    if len(ds) > 0:
        sample = ds[0]
        img = sample["image"]
        meta = sample["meta"]
        print("image:", tuple(img.shape), img.dtype)
        print("meta:", meta)