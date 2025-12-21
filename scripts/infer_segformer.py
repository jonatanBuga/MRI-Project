from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

# Ensure runnable from repo root: `python scripts/infer_segformer.py`
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
    path.parent.mkdir(parents=True, exist_ok=True)
    if arr_u8_hw.dtype != torch.uint8:
        raise ValueError(f"Expected uint8 tensor for PNG, got {arr_u8_hw.dtype}")
    Image.fromarray(arr_u8_hw.cpu().numpy(), mode="L").save(path)


def _tensor_proba_to_u8(proba_hw: torch.Tensor) -> torch.Tensor:
    proba_hw = proba_hw.clamp(0.0, 1.0)
    return (proba_hw * 255.0).round().to(dtype=torch.uint8)


def _tensor_mask_to_u8(mask_hw: torch.Tensor) -> torch.Tensor:
    return (mask_hw > 0).to(dtype=torch.uint8) * 255


def _get_loader(csv_path: str, split: str, batch_size: int, num_workers: int, pin_memory: bool, persistent_workers: bool):
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


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inference on Roboflow split using SegFormer (Hugging Face Transformers).")
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
        help="Optional output directory. If omitted, uses outputs/inference/segformer/{timestamp}/",
    )

    # SegFormer-specific
    p.add_argument("--model_name", default="nvidia/segformer-b0-finetuned-ade-512-512", type=str)
    p.add_argument("--num_labels", default=1, type=int, help="Desired number of output labels/channels (binary=1).")

    return p.parse_args()


def _build_model(model_name: str, num_labels: int, device: torch.device) -> torch.nn.Module:
    try:
        from transformers import SegformerConfig, SegformerForSemanticSegmentation
    except Exception as e:
        raise ImportError(
            "Failed to import Hugging Face transformers.\n"
            "Install with:\n"
            "  pip install transformers\n"
            "Also ensure you have a compatible PyTorch installed."
        ) from e

    if num_labels <= 0:
        raise ValueError(f"num_labels must be > 0, got: {num_labels}")

    # Prefer configuring num_labels at load time. If this fails (some models require matching classifier shapes),
    # fall back to loading default and adapting channel selection at runtime.
    try:
        cfg = SegformerConfig.from_pretrained(model_name)
        cfg.num_labels = int(num_labels)
        cfg.problem_type = "single_label_classification"  # harmless; kept explicit
        model = SegformerForSemanticSegmentation.from_pretrained(model_name, config=cfg, ignore_mismatched_sizes=True)
    except Exception:
        # Robust fallback
        model = SegformerForSemanticSegmentation.from_pretrained(model_name, ignore_mismatched_sizes=True)

    return model.to(device)


def main() -> int:
    args = _parse_args()
    device = _resolve_device(args.device)
    print(f"Device: {device}")

    thr = float(args.threshold)
    thr_tag = f"{thr:.2f}".replace(".", "p")

    out_root = Path(args.out_dir) if args.out_dir else (REPO_ROOT / "outputs" / "inference" / "segformer" / _now_timestamp())
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
        model = _build_model(args.model_name, args.num_labels, device)
    except Exception:
        print("Failed to build SegFormer model.")
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

            if images.ndim != 4:
                print(f"[batch {batch_idx}] Unexpected image tensor shape: {tuple(images.shape)}")
                return 3

            b, c, h, w = images.shape
            images = images.to(device, non_blocking=True)

            # HF models typically accept pixel_values in [0,1] float; no processor needed here
            try:
                outputs = model(pixel_values=images)
            except TypeError:
                # Some versions accept forward(images) directly
                outputs = model(images)

            if not hasattr(outputs, "logits"):
                print(f"[batch {batch_idx}] Model output missing .logits. Got type={type(outputs)}")
                return 3

            logits = outputs.logits  # usually [B, C, h', w']
            if not isinstance(logits, torch.Tensor) or logits.ndim != 4:
                print(f"[batch {batch_idx}] Unexpected logits type/shape: {type(logits)} {getattr(logits, 'shape', None)}")
                return 3

            # Ensure single-channel foreground logits [B,1,h',w']
            if logits.shape[1] == 1:
                logits_1c = logits
            elif logits.shape[1] >= 2:
                # Common convention: channel 1 is foreground
                logits_1c = logits[:, 1:2, :, :]
            else:
                print(f"[batch {batch_idx}] Invalid logits channels: {logits.shape[1]}")
                return 3

            # Upsample to input spatial size (SegFormer logits are often low-res)
            if logits_1c.shape[-2:] != (h, w):
                logits_1c = F.interpolate(logits_1c, size=(h, w), mode="bilinear", align_corners=False)

            probs = torch.sigmoid(logits_1c).to(dtype=torch.float32)  # [B,1,H,W] in [0,1]

            pmin = probs.min().item()
            pmax = probs.max().item()
            frac = (probs.detach().cpu() > thr).float().mean().item()
            print(f"[batch {batch_idx}] probs min/max=({pmin:.4f}, {pmax:.4f})  frac(p>{thr:g})={frac:.4f}")

            for i in range(b):
                meta = metas[i] if isinstance(metas, list) and i < len(metas) else {}
                image_path = str(meta.get("image_path", ""))
                img_stem = Path(image_path).stem if image_path else f"sample_{batch_idx:04d}_{i:02d}"

                proba_hw = probs[i, 0].detach().cpu()  # [H,W]
                mask_hw = (proba_hw > thr)

                _save_u8_png(proba_dir / f"{img_stem}_proba.png", _tensor_proba_to_u8(proba_hw))
                _save_u8_png(masks_dir / f"{img_stem}_mask.png", _tensor_mask_to_u8(mask_hw))

            num_done += 1

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())