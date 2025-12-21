from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path
from typing import Optional, Tuple

import torch
from PIL import Image

# Ensure runnable from repo root: `python scripts/infer_unet_resnet50.py`
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def _now_timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)

    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _save_u8_png(path: Path, arr_u8_hw: torch.Tensor) -> None:
    """
    Save a single-channel image (H,W) uint8 tensor as PNG.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if arr_u8_hw.dtype != torch.uint8:
        raise ValueError(f"Expected uint8 tensor for PNG, got {arr_u8_hw.dtype}")
    img = Image.fromarray(arr_u8_hw.cpu().numpy(), mode="L")
    img.save(path)


def _tensor_proba_to_u8(proba_hw: torch.Tensor) -> torch.Tensor:
    """
    Convert float32 probability [H,W] in [0,1] to uint8 [H,W] in [0,255].
    """
    proba_hw = proba_hw.clamp(0.0, 1.0)
    return (proba_hw * 255.0).round().to(dtype=torch.uint8)


def _tensor_mask_to_u8(mask_hw: torch.Tensor) -> torch.Tensor:
    """
    Convert binary mask [H,W] bool/int/float to uint8 [H,W] in {0,255}.
    """
    mask_hw = (mask_hw > 0).to(dtype=torch.uint8) * 255
    return mask_hw


def _get_loader(
    csv_path: str,
    split: str,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
):
    from src.data.dataloaders import build_dataloaders

    train_loader, val_loader, test_loader = build_dataloaders(
        metadata_csv=csv_path,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    if split == "train":
        return train_loader
    if split == "val":
        return val_loader
    if split == "test":
        return test_loader
    raise ValueError(f"split must be one of 'train'/'val'/'test', got: {split!r}")


def _build_model(device: torch.device) -> torch.nn.Module:
    try:
        import segmentation_models_pytorch as smp
    except Exception as e:
        raise ImportError(
            "Failed to import segmentation_models_pytorch. Install it in your environment, e.g.\n"
            "  pip install segmentation-models-pytorch\n"
            "and ensure torchvision/torch are installed via conda as per setup docs."
        ) from e

    model = smp.Unet(
        encoder_name="resnet50",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    )
    return model.to(device)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inference on Roboflow split using UNet(ResNet-50) baseline.")
    p.add_argument("--csv_path", default="metadata/metadata_labeled_roboflow_all.csv", type=str)
    p.add_argument("--split", default="val", choices=["train", "val", "test"], type=str)
    p.add_argument("--batch_size", default=8, type=int)
    p.add_argument("--num_workers", default=4, type=int)
    p.add_argument("--num_batches", default=5, type=int)
    p.add_argument("--device", default="auto", type=str, help="auto | cpu | cuda | mps | cuda:0 ...")
    p.add_argument("--threshold", default=0.5, type=float, help="Binarization threshold for masks (0..1)")
    p.add_argument(
        "--out_dir",
        default=None,
        type=str,
        help="Optional output directory. If omitted, uses outputs/inference/unet_resnet50/{timestamp}/",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    device = _resolve_device(args.device)
    print(f"Device: {device}")

    thr = float(args.threshold)

    out_root = Path(args.out_dir) if args.out_dir else (
        REPO_ROOT / "outputs" / "inference" / "unet_resnet50" / _now_timestamp()
    )
    thr_tag = f"{thr:.2f}".replace(".", "p")
    masks_dir = out_root / f"masks_thr_{thr_tag}"
    proba_dir = out_root / "proba_heatmaps"

    pin_memory = True
    persistent_workers = True if args.num_workers > 0 else False

    try:
        loader = _get_loader(
            csv_path=args.csv_path,
            split=args.split,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
    except Exception:
        print("Failed to build DataLoader.")
        traceback.print_exc()
        return 2

    try:
        model = _build_model(device)
    except Exception:
        traceback.print_exc()
        return 2

    model.eval()

    print(f"Output dir: {out_root.as_posix()}")
    out_root.mkdir(parents=True, exist_ok=True)

    num_done = 0
    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader):
            if num_done >= args.num_batches:
                break

            try:
                images = batch["image"]  # [B,3,H,W] float32 in [0,1]
                metas = batch["meta"]    # list[dict]
            except Exception:
                print(f"[batch {batch_idx}] Unexpected batch format. Expected keys: image/mask/meta.")
                traceback.print_exc()
                return 3

            images = images.to(device, non_blocking=True)

            logits = model(images)  # expected [B,1,H,W]
            probs = torch.sigmoid(logits).to(dtype=torch.float32)  # [B,1,H,W] in [0,1]

            # Logging
            pmin = probs.min().item()
            pmax = probs.max().item()
            frac = (probs.detach().cpu() > thr).float().mean().item()
            print(f"[batch {batch_idx}] probs min/max=({pmin:.4f}, {pmax:.4f})  frac(p>{thr:g})={frac:.4f}")

            # Save per-sample outputs
            b = probs.shape[0]
            for i in range(b):
                meta = metas[i] if isinstance(metas, list) and i < len(metas) else {}
                image_path = str(meta.get("image_path", ""))
                img_stem = Path(image_path).stem if image_path else f"sample_{batch_idx:04d}_{i:02d}"

                proba_hw = probs[i, 0].detach().cpu()  # [H,W]
                mask_hw = (proba_hw > thr)

                proba_u8 = _tensor_proba_to_u8(proba_hw)
                mask_u8 = _tensor_mask_to_u8(mask_hw)

                proba_path = proba_dir / f"{img_stem}_proba.png"
                mask_path = masks_dir / f"{img_stem}_mask.png"

                _save_u8_png(proba_path, proba_u8)
                _save_u8_png(mask_path, mask_u8)

            num_done += 1

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())