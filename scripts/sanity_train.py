from __future__ import annotations

import argparse
import csv
import sys
import traceback
from itertools import islice
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure runnable from repo root: `python scripts/sanity_train.py`
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)

    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class LogitsWrapper(nn.Module):
    """
    Unifies model outputs to a single Tensor of logits [B,1,H,W].

    This lets us keep src/training/engine.py unchanged while supporting:
      - models that return Tensor logits directly (e.g., SMP UNet, TransUNetWrapper)
      - Hugging Face models that return an object with .logits (e.g., SegFormer)
    """

    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        self.base = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected input [B,3,H,W], got shape={tuple(x.shape)}")
        b, c, h, w = x.shape
        if c != 3:
            raise ValueError(f"Expected 3-channel input, got C={c}")

        out = None
        # HF SegFormer typically expects pixel_values=...
        try:
            out = self.base(pixel_values=x)
        except TypeError:
            out = self.base(x)

        logits = out
        if not isinstance(logits, torch.Tensor):
            if hasattr(out, "logits"):
                logits = out.logits
            else:
                raise TypeError(f"Model output is not a Tensor and has no .logits attribute: {type(out)}")

        if logits.ndim != 4:
            raise ValueError(f"Expected logits [B,C,h,w], got shape={tuple(logits.shape)}")
        if logits.shape[0] != b:
            raise ValueError(f"Logits batch mismatch: logits B={logits.shape[0]} vs input B={b}")

        # Fail fast in training if channel count is not binary (C=1).
        # We do not auto-select a "foreground" channel here because it can hide config/checkpoint mistakes.
        if logits.shape[1] == 1:
            logits_1c = logits
        else:
            raise ValueError(
                f"Expected binary logits with C=1. Got C={logits.shape[1]}. Check SegFormer config/checkpoint."
            )

        # Upsample if model outputs lower-res logits (e.g., SegFormer)
        if logits_1c.shape[-2:] != (h, w):
            logits_1c = F.interpolate(logits_1c, size=(h, w), mode="bilinear", align_corners=False)

        # Ensure contiguous logits to avoid .view() issues in loss/metrics.
        logits_1c = logits_1c.contiguous()

        if logits_1c.shape != (b, 1, h, w):
            raise ValueError(f"Unified logits must be [B,1,H,W]; got {tuple(logits_1c.shape)}")

        return logits_1c


def build_model(model_name: str, device: torch.device, segformer_model_name: str | None = None) -> torch.nn.Module:
    """
    Build a model and move it to the target device.

    Supported:
      - unet_r50: segmentation_models_pytorch UNet with ResNet-50 encoder
      - transunet: local TransUNetWrapper (vendored TransUNet code required)
      - segformer: Hugging Face SegFormer, binary head (num_labels=1), ignore_mismatched_sizes=True
    """
    model_name = str(model_name)

    if model_name == "unet_r50":
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
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
            activation=None,  # logits
        )
        return model.to(device)

    if model_name == "transunet":
        try:
            from src.models.transunet_wrapper import TransUNetWrapper
        except Exception as e:
            raise ImportError(
                "Failed to import TransUNetWrapper.\n"
                "Ensure TransUNet is vendored under third_party/transunet (see third_party/transunet/README.md)."
            ) from e

        model = TransUNetWrapper()
        return model.to(device)

    if model_name == "segformer":
        # Must match scripts/infer_segformer.py default; do not hardcode here.
        if not segformer_model_name:
            raise ValueError("segformer_model_name must be provided when model_name == 'segformer'.")

        try:
            from transformers import SegformerConfig, SegformerForSemanticSegmentation
        except Exception as e:
            raise ImportError(
                "Failed to import Hugging Face transformers.\n"
                "Install with:\n"
                "  pip install transformers"
            ) from e

        try:
            cfg = SegformerConfig.from_pretrained(segformer_model_name)
            cfg.num_labels = 1
            model = SegformerForSemanticSegmentation.from_pretrained(
                segformer_model_name, config=cfg, ignore_mismatched_sizes=True
            )
        except Exception:
            # Robust fallback
            model = SegformerForSemanticSegmentation.from_pretrained(segformer_model_name, ignore_mismatched_sizes=True)

        return model.to(device)

    raise ValueError(f"Unknown model: {model_name!r}. Expected one of: unet_r50, transunet, segformer")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sanity training run (limited batches) for UNet(ResNet50) baseline.")
    p.add_argument("--csv_path", default="metadata/metadata_labeled_roboflow_all.csv", type=str)
    p.add_argument("--epochs", default=2, type=int)
    p.add_argument("--batch_size", default=8, type=int)
    p.add_argument("--num_workers", default=4, type=int)
    p.add_argument("--device", default="auto", type=str, help="auto | cpu | cuda | mps | cuda:0 ...")
    p.add_argument("--threshold", default=0.5, type=float)
    p.add_argument("--lr", default=1e-3, type=float)
    p.add_argument("--max_train_batches", default=10, type=int)
    p.add_argument("--max_val_batches", default=10, type=int)

    # New: multi-model switch
    p.add_argument("--model", default="unet_r50", choices=["unet_r50", "transunet", "segformer"], type=str)

    # SegFormer checkpoint (must match scripts/infer_segformer.py default)
    p.add_argument("--segformer_model_name", default="nvidia/segformer-b0-finetuned-ade-512-512", type=str)

    # New: checkpoint + fixed-sample visualization
    p.add_argument("--save_ckpt_path", default="outputs/checkpoints/sanity_{model}.pt", type=str)
    p.add_argument("--viz_samples", default=8, type=int)
    p.add_argument("--viz_split", default="val", choices=["train", "val", "test"], type=str)

    # New: history logging + plots
    p.add_argument("--history_csv_path", default="outputs/logs/sanity_{model}.csv", type=str)
    p.add_argument("--plots_dir", default="outputs/plots/sanity/{model}", type=str)

    return p.parse_args()


def main() -> int:
    args = _parse_args()
    device = _resolve_device(args.device)
    print(f"Device: {device}")
    print(f"Model: {args.model}")

    try:
        from src.data.dataloaders import build_dataloaders
        from src.training.engine import train_one_epoch, validate_one_epoch
        from src.training.losses import dice_bce_loss
        from src.training.visualization import save_png_triplet
    except Exception:
        print("Failed to import training components.")
        traceback.print_exc()
        return 2

    # Build full loaders, then limit batches via islice for sanity runs
    pin_memory = True
    persistent_workers = True if args.num_workers > 0 else False

    train_loader, val_loader, test_loader = build_dataloaders(
        metadata_csv=args.csv_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    # Select loader for fixed visualization samples
    if args.viz_split == "train":
        viz_loader = train_loader
    elif args.viz_split == "val":
        viz_loader = val_loader
    else:
        viz_loader = test_loader

    # Fixed samples are taken from the FIRST batch (intentional + sufficient for sanity-stage evaluation).
    # Pick fixed samples deterministically: first batch, first N items
    try:
        fixed_batch = next(iter(viz_loader))
        fixed_images = fixed_batch["image"]
        fixed_masks = fixed_batch["mask"]
        fixed_metas = fixed_batch.get("meta", None)

        n = min(int(args.viz_samples), int(fixed_images.shape[0]))
        fixed_images = fixed_images[:n].detach().cpu().contiguous()
        fixed_masks = fixed_masks[:n].detach().cpu().contiguous()

        stems: list[str] = []
        if isinstance(fixed_metas, list):
            for i in range(n):
                image_path = str(fixed_metas[i].get("image_path", "")) if i < len(fixed_metas) else ""
                stem = Path(image_path).stem if image_path else f"sample_{i:02d}"
                stems.append(stem)
        else:
            stems = [f"sample_{i:02d}" for i in range(n)]
    except Exception:
        print("Failed to select fixed visualization samples from the first batch.")
        traceback.print_exc()
        return 2

    # Model (wrapped to always return logits [B,1,H,W])
    try:
        base_model = build_model(args.model, device=device, segformer_model_name=args.segformer_model_name)
        model = LogitsWrapper(base_model).to(device)
    except Exception:
        print("Failed to build selected model.")
        traceback.print_exc()
        return 2

    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr))

    def loss_fn(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Extra safety for MPS: avoid non-contiguous tensors reaching losses that may use .view()
        logits = logits.contiguous()
        targets = targets.contiguous()
        return dice_bce_loss(logits, targets, dice_weight=1.0, bce_weight=1.0)

    thr = float(args.threshold)

    history: list[dict] = []

    def _save_fixed_viz(epoch_tag: str) -> None:
        out_dir = REPO_ROOT / "outputs" / "sanity_viz" / str(args.model) / str(epoch_tag)
        model.eval()
        with torch.no_grad():
            logits = model(fixed_images.to(device))
            probs = torch.sigmoid(logits).to(dtype=torch.float32).detach().cpu()
            pred = (probs > thr)

        for i in range(fixed_images.shape[0]):
            stem = f"{stems[i]}_{i:02d}"
            save_png_triplet(
                out_dir=out_dir,
                stem=stem,
                image_chw=fixed_images[i],
                gt_hw=fixed_masks[i, 0],
                pred_hw=pred[i, 0],
            )

    # Optional but recommended: save epoch_0 predictions before training
    _save_fixed_viz("epoch_0")

    for epoch in range(1, int(args.epochs) + 1):
        # Limit batches for sanity run (do NOT materialize loaders into lists)
        train_iter = islice(train_loader, int(args.max_train_batches))
        val_iter = islice(val_loader, int(args.max_val_batches))

        train_stats = train_one_epoch(
            model=model,
            loader=train_iter,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            epoch=epoch,
            log_every=0,
        )
        val_stats = validate_one_epoch(
            model=model,
            loader=val_iter,
            loss_fn=loss_fn,
            device=device,
            threshold=thr,
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_stats["loss"],
                "val_loss": val_stats["loss"],
                "val_dice": val_stats["dice"],
                "val_iou": val_stats["iou"],
                "val_precision": val_stats["precision"],
                "val_recall": val_stats["recall"],
            }
        )

        print(
            f"[epoch {epoch}] "
            f"train_loss={train_stats['loss']:.4f}  "
            f"val_loss={val_stats['loss']:.4f}  "
            f"val_dice={val_stats['dice']:.4f}  "
            f"val_iou={val_stats['iou']:.4f}  "
            f"val_precision={val_stats['precision']:.4f}  "
            f"val_recall={val_stats['recall']:.4f}"
        )

    # Save checkpoint (base model weights only)
    save_ckpt_path = Path(str(args.save_ckpt_path).format(model=str(args.model)))
    save_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": base_model.state_dict()}, save_ckpt_path)

    # Reload checkpoint into the same base model object, then re-run fixed-sample inference
    try:
        ckpt = torch.load(save_ckpt_path, map_location="cpu")
        base_model.load_state_dict(ckpt["model"])
    except Exception:
        print(f"Failed to reload checkpoint from: {save_ckpt_path.as_posix()}")
        traceback.print_exc()
        return 2

    # Ensure the reloaded model runs on the requested device (avoid silent CPU inference).
    base_model = base_model.to(device)
    model = LogitsWrapper(base_model).to(device)

    _save_fixed_viz("epoch_last")

    # Save training history CSV (stdlib only)
    history_csv_path = Path(str(args.history_csv_path).format(model=str(args.model)))
    history_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with history_csv_path.open("w", newline="") as f:
        fieldnames = ["epoch", "train_loss", "val_loss", "val_dice", "val_iou", "val_precision", "val_recall"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in history:
            w.writerow(row)

    # Save simple plots (headless)
    plots_dir = Path(str(args.plots_dir).format(model=str(args.model)))
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Import matplotlib only here to keep startup fast when plotting is not needed elsewhere.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [r["epoch"] for r in history]

    # loss_curve.png
    plt.figure()
    plt.plot(epochs, [r["train_loss"] for r in history], label="train_loss")
    plt.plot(epochs, [r["val_loss"] for r in history], label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "loss_curve.png")
    plt.close()

    # metrics_curve.png
    plt.figure()
    plt.plot(epochs, [r["val_dice"] for r in history], label="val_dice")
    plt.plot(epochs, [r["val_iou"] for r in history], label="val_iou")
    plt.plot(epochs, [r["val_precision"] for r in history], label="val_precision")
    plt.plot(epochs, [r["val_recall"] for r in history], label="val_recall")
    plt.xlabel("epoch")
    plt.ylabel("metric")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "metrics_curve.png")
    plt.close()

    print(
        f"Sanity training complete. Checkpoint saved to: {save_ckpt_path.as_posix()}. "
        "Visualizations written under: outputs/sanity_viz/"
    )
    print(f"History saved to: {history_csv_path.as_posix()}")
    print(f"Plots saved under: {plots_dir.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())