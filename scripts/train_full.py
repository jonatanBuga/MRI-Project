# scripts/train_full.py
from __future__ import annotations

import argparse
import csv
import sys
import traceback
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        self.base = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        try:
            out = self.base(pixel_values=x)
        except TypeError:
            out = self.base(x)

        logits = out
        if not isinstance(logits, torch.Tensor):
            if hasattr(out, "logits"):
                logits = out.logits
            else:
                raise TypeError(f"Model output is not Tensor and has no .logits: {type(out)}")

        if logits.shape[1] != 1:
            raise ValueError(f"Expected binary logits C=1, got C={logits.shape[1]}")

        if logits.shape[-2:] != (h, w):
            logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)

        return logits.contiguous()


def build_model_unet_r50(device: torch.device) -> nn.Module:
    import segmentation_models_pytorch as smp
    model = smp.Unet(
        encoder_name="resnet50",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None,
    )
    return model.to(device)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Full training: UNet+ResNet50 baseline")
    p.add_argument("--csv_path", default="metadata/metadata_labeled_roboflow_all.csv", type=str)

    p.add_argument("--epochs", default=50, type=int)
    p.add_argument("--batch_size", default=8, type=int)
    p.add_argument("--num_workers", default=4, type=int)
    p.add_argument("--device", default="auto", type=str)
    p.add_argument("--threshold", default=0.5, type=float)

    p.add_argument("--lr", default=1e-4, type=float)
    p.add_argument("--weight_decay", default=1e-4, type=float)
    p.add_argument("--dice_weight", default=1.5, type=float)
    p.add_argument("--bce_weight", default=1.0, type=float)

    p.add_argument("--use_plateau", action="store_true", help="Use ReduceLROnPlateau on val_dice")
    p.add_argument("--plateau_factor", default=0.5, type=float)
    p.add_argument("--plateau_patience", default=3, type=int)
    p.add_argument("--plateau_min_lr", default=1e-6, type=float)

    p.add_argument("--viz_samples", default=8, type=int)
    p.add_argument("--viz_split", default="val", choices=["train", "val", "test"], type=str)

    # output root
    p.add_argument("--out_root", default="outputs/training", type=str)
    p.add_argument("--run_name", default="", type=str)  # optional custom name

    return p.parse_args()


def main() -> int:
    args = _parse_args()
    device = _resolve_device(args.device)
    print(f"Device: {device}")
    print("Model: unet_r50")

    try:
        from src.data.dataloaders import build_dataloaders
        from src.training.engine import train_one_epoch, validate_one_epoch
        from src.training.losses import dice_bce_loss
        from src.training.visualization import save_png_triplet
    except Exception:
        print("Failed to import training components.")
        traceback.print_exc()
        return 2

    # ---- run dir ----
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_tag = args.run_name.strip() or f"unet_r50_{ts}"
    run_dir = REPO_ROOT / args.out_root / run_tag
    ckpt_dir = run_dir / "checkpoints"
    viz_dir = run_dir / "viz"
    plots_dir = run_dir / "plots"
    logs_dir = run_dir / "logs"

    for d in [ckpt_dir, viz_dir, plots_dir, logs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    history_csv_path = logs_dir / "history.csv"
    best_ckpt_path = ckpt_dir / "best_by_val_dice.pt"
    last_ckpt_path = ckpt_dir / "last.pt"

    # ---- dataloaders ----
    pin_memory = True
    persistent_workers = True if args.num_workers > 0 else False

    train_loader, val_loader, test_loader = build_dataloaders(
        metadata_csv=args.csv_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    # ---- fixed viz batch ----
    if args.viz_split == "train":
        viz_loader = train_loader
    elif args.viz_split == "val":
        viz_loader = val_loader
    else:
        viz_loader = test_loader

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

    # ---- model ----
    base_model = build_model_unet_r50(device=device)
    model = LogitsWrapper(base_model).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    scheduler = None
    if args.use_plateau:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",                 
            factor=float(args.plateau_factor),
            patience=int(args.plateau_patience),
            min_lr=float(args.plateau_min_lr),
            verbose=True,
        )

    def loss_fn(logits: torch.Tensor, targets: torch.Tensor):
        return dice_bce_loss(
                logits.contiguous(),
                targets.contiguous(),
                dice_weight=float(args.dice_weight),
                bce_weight=float(args.bce_weight),
            )
    thr = float(args.threshold)
    best_dice = -1.0
    history: list[dict] = []

    def _save_fixed_viz(epoch_tag: str) -> None:
        out_dir = viz_dir / epoch_tag
        out_dir.mkdir(parents=True, exist_ok=True)
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

    # epoch_0 viz
    _save_fixed_viz("epoch_0")

    # ---- training loop ----
    for epoch in range(1, int(args.epochs) + 1):
        train_stats = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            epoch=epoch,
            log_every=0,
        )
        val_stats = validate_one_epoch(
            model=model,
            loader=val_loader,
            loss_fn=loss_fn,
            device=device,
            threshold=thr,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "val_loss": val_stats["loss"],
            "val_dice": val_stats["dice"],
            "val_iou": val_stats["iou"],
            "val_precision": val_stats["precision"],
            "val_recall": val_stats["recall"],
        }
        history.append(row)
        if scheduler is not None:
            scheduler.step(float(row["val_dice"]))
        current_lr = optimizer.param_groups[0]["lr"]    
        print(
            f"[epoch {epoch}] "
            f"train_loss={row['train_loss']:.4f} "
            f"val_loss={row['val_loss']:.4f} "
            f"val_dice={row['val_dice']:.4f} "
            f"val_iou={row['val_iou']:.4f} "
            f"val_precision={row['val_precision']:.4f} "
            f"val_recall={row['val_recall']:.4f}"
            f"lr={current_lr:.2e}"
        )
        # update best first (if needed)
        if row["val_dice"] > best_dice:
            best_dice = float(row["val_dice"])
            torch.save(
                {"model": base_model.state_dict(), "epoch": epoch, "best_dice": best_dice},
                best_ckpt_path,
            )
        # save last every epoch
        torch.save(
            {"model": base_model.state_dict(), "epoch": epoch, "best_dice": best_dice},
            last_ckpt_path,
        )
        

    _save_fixed_viz("epoch_last")

    # ---- save history csv ----
    if history:
        with history_csv_path.open("w", newline="") as f:
            fieldnames = list(history[0].keys())
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(history)
    else:
        print("WARNING: history is empty, skipping CSV export.")

    # ---- plots ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [r["epoch"] for r in history]

    plt.figure()
    plt.plot(epochs, [r["train_loss"] for r in history], label="train_loss")
    plt.plot(epochs, [r["val_loss"] for r in history], label="val_loss")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.tight_layout()
    plt.savefig(plots_dir / "loss_curve.png"); plt.close()

    plt.figure()
    plt.plot(epochs, [r["val_dice"] for r in history], label="val_dice")
    plt.plot(epochs, [r["val_iou"] for r in history], label="val_iou")
    plt.plot(epochs, [r["val_precision"] for r in history], label="val_precision")
    plt.plot(epochs, [r["val_recall"] for r in history], label="val_recall")
    plt.xlabel("epoch"); plt.ylabel("metric"); plt.legend(); plt.tight_layout()
    plt.savefig(plots_dir / "metrics_curve.png"); plt.close()

    print(f"\nDONE. Run dir: {run_dir.as_posix()}")
    print(f"Best ckpt: {best_ckpt_path.as_posix()} (best_dice={best_dice:.4f})")
    print(f"Last ckpt: {last_ckpt_path.as_posix()}")
    print(f"History: {history_csv_path.as_posix()}")
    print(f"Viz: {viz_dir.as_posix()}")
    print(f"Plots: {plots_dir.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
