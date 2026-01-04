from __future__ import annotations

import argparse
import csv
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader

# Ensure runnable from repo root: `python scripts/eval_healthy_samples.py ...`
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

THRESHOLD = 0.5


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


def _proba_to_u8(proba_hw: torch.Tensor) -> torch.Tensor:
    proba_hw = proba_hw.detach().to(dtype=torch.float32).cpu().clamp(0.0, 1.0).contiguous()
    return (proba_hw * 255.0).round().to(dtype=torch.uint8)


def _mask_to_u8(mask_hw: torch.Tensor) -> torch.Tensor:
    mask_hw = mask_hw.detach().cpu()
    if mask_hw.dtype != torch.bool:
        mask_hw = mask_hw.to(dtype=torch.float32) > 0.5
    return (mask_hw.to(dtype=torch.uint8) * 255).contiguous()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run UNet(ResNet50) baseline inference on healthy MRI images (no masks) using MRIInferenceDataset."
    )
    p.add_argument("--healthy_root", required=True, type=str, help="Root folder containing healthy MRI slice images.")
    p.add_argument("--checkpoint", required=True, type=str, help="Path to checkpoint (.pt) containing model weights.")
    p.add_argument("--batch_size", default=8, type=int)
    p.add_argument("--num_workers", default=4, type=int)
    p.add_argument("--device", default="auto", type=str, help="auto | cpu | cuda | mps | cuda:0 ...")
    p.add_argument("--limit", default=None, type=int, help="Optional max number of images to process.")
    return p.parse_args()


def _normalize_meta_batch(meta: Any, batch_size: int) -> List[Dict[str, Any]]:
    """
    MRIInferenceDataset typically returns meta as dict per sample, and DataLoader collation
    may produce either:
      - list[dict]
      - dict[str, list]
    Normalize to list[dict] of length B.
    """
    if meta is None:
        return [{} for _ in range(batch_size)]

    if isinstance(meta, list):
        return [m if isinstance(m, dict) else {} for m in meta]

    if isinstance(meta, dict):
        out: List[Dict[str, Any]] = []
        for i in range(batch_size):
            d: Dict[str, Any] = {}
            for k, v in meta.items():
                try:
                    d[k] = v[i]
                except Exception:
                    d[k] = v
            out.append(d)
        return out

    return [{} for _ in range(batch_size)]


def _build_model(device: torch.device) -> torch.nn.Module:
    """
    Build the baseline UNet+ResNet50 exactly as used in this repo.

    Prefer reusing the existing helper (to avoid drifting definitions).
    Fall back to a local equivalent definition if the helper is unavailable.
    """
    # Reuse the baseline builder from the sanity training script if available.
    # This matches the project's canonical definition for UNet(ResNet50) logits output.
    try:
        from scripts.sanity_train import build_model as build_sanity_model

        model = build_sanity_model("unet_r50", device=device)
        return model
    except Exception:
        # Fallback: minimal equivalent architecture (activation=None -> logits)
        try:
            import segmentation_models_pytorch as smp
        except Exception as e:
            raise ImportError(
                "Failed to import segmentation_models_pytorch.\n"
                "Install it with:\n"
                "  pip install segmentation-models-pytorch"
            ) from e

        model = smp.Unet(
            encoder_name="resnet50",
            encoder_weights=None,  # weights come from checkpoint; avoids downloading at inference time
            in_channels=3,
            classes=1,
            activation=None,  # logits
        )
        return model.to(device)


def _load_checkpoint(model: torch.nn.Module, ckpt_path: Path, device: torch.device) -> torch.nn.Module:
    """
    Load checkpoint from --checkpoint and restore model state_dict, then move to device and eval().
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")

    if isinstance(ckpt, dict):
        state = (
        ckpt.get("model")
        or ckpt.get("state_dict")
        or ckpt.get("model_state_dict")
        or ckpt
        )
    else:
        raise ValueError(f"Unexpected checkpoint format at {ckpt_path.as_posix()}: {type(ckpt)}")

    model.load_state_dict(state, strict=True)
    model = model.to(device)
    model.eval()
    return model



def main() -> int:
    args = _parse_args()
    device = _resolve_device(args.device)

    healthy_root = Path(args.healthy_root)
    ckpt_path = Path(args.checkpoint)

    if not healthy_root.exists():
        print(f"healthy_root does not exist: {healthy_root.as_posix()}")
        return 2
    if not ckpt_path.exists():
        print(f"checkpoint does not exist: {ckpt_path.as_posix()}")
        return 2

    out_root = REPO_ROOT / "outputs" / "inference" / "healthy_baseline" / _now_timestamp()
    probs_dir = out_root / "probs"
    masks_dir = out_root / "masks"
    probs_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []
    summary_csv_path = out_root / "summary.csv"

    try:
        from src.data.mri_inference_dataset import MRIInferenceDataset
    except Exception:
        print("Failed to import MRIInferenceDataset from src/data/mri_inference_dataset.py")
        traceback.print_exc()
        return 2

    # Dataset uses the repo’s standard preprocessing (RGB, resize to 256x256, min-max to [0,1], tensor CHW).
    dataset = MRIInferenceDataset(root_dir=str(healthy_root))
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=True,
        persistent_workers=True if int(args.num_workers) > 0 else False,
    )

    try:
        model = _build_model(device)
        model = _load_checkpoint(model, ckpt_path, device=device)
    except Exception:
        print("Failed to build/load baseline model checkpoint.")
        traceback.print_exc()
        return 2

    total = 0
    limit = int(args.limit) if args.limit is not None else None

    with torch.no_grad():
        for batch in loader:
            images = batch.get("image")
            meta = batch.get("meta", None)

            if not isinstance(images, torch.Tensor) or images.ndim != 4:
                raise ValueError(
                    f"Expected batch['image'] as Tensor [B,3,H,W], got: {type(images)} {getattr(images, 'shape', None)}"
                )

            b, c, h, w = images.shape
            if c != 3 or (h, w) != (256, 256):
                raise ValueError(f"Expected images [B,3,256,256] from preprocessing. Got: {tuple(images.shape)}")

            metas = _normalize_meta_batch(meta, b)

            images = images.to(device, non_blocking=True)
            logits = model(images)

            if not isinstance(logits, torch.Tensor) or logits.shape != (b, 1, h, w):
                raise ValueError(f"Expected logits [B,1,256,256], got: {type(logits)} {getattr(logits, 'shape', None)}")

            probs = torch.sigmoid(logits).to(dtype=torch.float32).detach().cpu().contiguous()  # [B,1,H,W]
            pred = (probs > THRESHOLD)

            for i in range(b):
                image_path = str(metas[i].get("image_path", ""))
                idx = metas[i].get("index", None)

                stem = Path(image_path).stem if image_path else f"sample_{total+i:06d}"
                if idx is not None:
                    stem = f"{stem}_idx{int(idx)}"

                proba_hw = probs[i, 0]  # [H,W]
                pred_hw = pred[i, 0]    # [H,W] bool

                _save_u8_png(probs_dir / f"{stem}_proba.png", _proba_to_u8(proba_hw))
                _save_u8_png(masks_dir / f"{stem}_mask.png", _mask_to_u8(pred_hw))

                # Sanity summary for healthy controls (no labels): quantify probability mass / false positives.
                mean_prob = float(proba_hw.mean().item())
                max_prob = float(proba_hw.max().item())
                positive_pixel_ratio = float((proba_hw >= THRESHOLD).to(dtype=torch.float32).mean().item())

                summary_rows.append(
                    {
                        "image_path": image_path,
                        "mean_prob": mean_prob,
                        "max_prob": max_prob,
                        "positive_pixel_ratio": positive_pixel_ratio,
                    }
                )

            total += b
            if limit is not None and total >= limit:
                break

    if limit is not None:
        total = min(total, limit)

    # Write summary CSV once at the end
    summary_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv_path.open("w", newline="") as f:
        fieldnames = ["image_path", "mean_prob", "max_prob", "positive_pixel_ratio"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in summary_rows:
            w.writerow(row)

    print(f"Processed {total} images.")
    print(f"Outputs saved to: {out_root.as_posix()}")
    print(f"Summary CSV saved to {summary_csv_path.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())