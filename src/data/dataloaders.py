from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader

from src.data.mri_seg_dataset import MRISegmentationDataset


def _collate_segmentation_batch(batch: List[Dict]) -> Dict:
    """
    Collate function that stacks tensors and keeps `meta` as a list of dicts.

    Output:
      - image: float32 Tensor [B, C, H, W]
      - mask:  float32 Tensor [B, 1, H, W]
      - meta:  list[dict]
    """
    images = torch.stack([b["image"] for b in batch], dim=0).to(dtype=torch.float32)  # [B, C, H, W]

    # Dataset mask is [H, W] long in {0,1}; batch mask should be [B, 1, H, W] float32
    masks = torch.stack([b["mask"] for b in batch], dim=0)  # [B, H, W]
    masks = masks.unsqueeze(1).to(dtype=torch.float32)  # [B, 1, H, W]

    metas = [b["meta"] for b in batch]  # list[dict]
    return {"image": images, "mask": masks, "meta": metas}


def build_dataloaders(
    metadata_csv: str = "metadata/metadata_labeled_roboflow_all.csv",
    batch_size: int = 8,
    num_workers: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    ds_train = MRISegmentationDataset(metadata_csv=metadata_csv, split="train", mode="labeled_only")
    ds_val = MRISegmentationDataset(metadata_csv=metadata_csv, split="val", mode="labeled_only")
    ds_test = MRISegmentationDataset(metadata_csv=metadata_csv, split="test", mode="labeled_only")

    if num_workers == 0:
        persistent_workers = False

    g = torch.Generator()
    g.manual_seed(1337)

    train_loader = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        generator=g,
        collate_fn=_collate_segmentation_batch,
    )
    val_loader = DataLoader(
        ds_val,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        collate_fn=_collate_segmentation_batch,
    )
    test_loader = DataLoader(
        ds_test,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        collate_fn=_collate_segmentation_batch,
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = build_dataloaders(batch_size=4)

    for name, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
        batch = next(iter(loader))
        images = batch["image"]
        masks = batch["mask"]
        meta = batch["meta"]

        print(f"\n[{name}]")
        print("image:", tuple(images.shape), images.dtype)
        print("mask: ", tuple(masks.shape), masks.dtype)
        print("mask unique:", torch.unique(masks).cpu().tolist())
        if isinstance(meta, list) and meta:
            print("meta[0] keys:", sorted(list(meta[0].keys())))
        else:
            print("meta type:", type(meta))