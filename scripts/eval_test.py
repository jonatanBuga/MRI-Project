from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# Ensure runnable from repo root: `python scripts/eval_test.py ...`
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

THRESHOLD = 0.5
TARGET_SIZE = (256, 256)


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


def _is_image_file(p: Path) -> bool:
    return p.suffix.lower() in {".png", ".jpg", ".jpeg"}

def _normalize_stem(stem: str) -> str:
    stem = stem.strip()
    # Roboflow masks often add "__mask"
    for suf in ["__mask", "_mask", "-mask", "_seg", "_label"]:
        if stem.endswith(suf):
            stem = stem[: -len(suf)]
            break
    return stem
def _save_u8_png(path: Path, arr_u8_hw: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not isinstance(arr_u8_hw, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor for PNG save, got {type(arr_u8_hw)}")
    if arr_u8_hw.dtype != torch.uint8:
        raise ValueError(f"Expected uint8 tensor for PNG, got {arr_u8_hw.dtype}")
    if arr_u8_hw.ndim != 2:
        raise ValueError(f"Expected [H,W] uint8 tensor, got shape={tuple(arr_u8_hw.shape)}")
    Image.fromarray(arr_u8_hw.cpu().numpy(), mode="L").save(path)


def _proba_to_u8(proba_hw: torch.Tensor) -> torch.Tensor:
    x = proba_hw.detach().to(dtype=torch.float32).cpu().clamp(0.0, 1.0).contiguous()
    return (x * 255.0).round().to(dtype=torch.uint8)


def _mask_to_u8(mask_hw: torch.Tensor) -> torch.Tensor:
    m = mask_hw.detach().cpu()
    if m.dtype == torch.bool:
        m_bin = m
    else:
        m_bin = m.to(dtype=torch.float32) > 0.5
    return (m_bin.to(dtype=torch.uint8) * 255).contiguous()


def _rgb_u8_from_image_chw(image_chw: torch.Tensor) -> Image.Image:
    """
    image_chw: [3,H,W] float in [0,1] -> PIL RGB
    """
    if image_chw.ndim != 3 or image_chw.shape[0] != 3:
        raise ValueError(f"Expected image_chw [3,H,W], got {tuple(image_chw.shape)}")
    x = image_chw.detach().to(dtype=torch.float32).cpu().clamp(0.0, 1.0)
    x = (x * 255.0).round().to(dtype=torch.uint8)
    x = x.permute(1, 2, 0).contiguous()  # [H,W,3]
    return Image.fromarray(x.numpy(), mode="RGB")


def _gray_u8_from_hw_u8(u8_hw: torch.Tensor) -> Image.Image:
    return Image.fromarray(u8_hw.detach().cpu().numpy(), mode="L")


def _save_viz_panel(
    out_path: Path,
    image_chw: torch.Tensor,   # [3,H,W] float [0,1]
    gt_hw: torch.Tensor,       # [H,W] {0,1} float/bool
    pred_hw: torch.Tensor,     # [H,W] bool
    proba_hw: torch.Tensor,    # [H,W] float [0,1]
) -> None:
    """
    Saves a simple 2x2 panel: input | GT | pred | prob
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    img = _rgb_u8_from_image_chw(image_chw)
    gt = _gray_u8_from_hw_u8(_mask_to_u8(gt_hw))
    pred = _gray_u8_from_hw_u8(_mask_to_u8(pred_hw))
    prob = _gray_u8_from_hw_u8(_proba_to_u8(proba_hw))

    w, h = img.size
    canvas = Image.new("RGB", (w * 2, h * 2), color=(0, 0, 0))
    canvas.paste(img, (0, 0))
    canvas.paste(gt.convert("RGB"), (w, 0))
    canvas.paste(pred.convert("RGB"), (0, h))
    canvas.paste(prob.convert("RGB"), (w, h))
    canvas.save(out_path)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate UNet(ResNet50) baseline on a matched image/mask test set.")
    p.add_argument("--images_root", default="data", type=str)
    p.add_argument("--masks_root", default="data/masks_raw", type=str)
    p.add_argument("--checkpoint", required=True, type=str)
    p.add_argument("--batch_size", default=8, type=int)
    p.add_argument("--num_workers", default=4, type=int)
    p.add_argument("--device", default="auto", type=str, help="auto | cpu | cuda | mps | cuda:0 ...")
    p.add_argument("--limit", default=None, type=int, help="Optional limit on number of matched samples to process.")
    p.add_argument("--save_viz_n", default=16, type=int, help="Save N qualitative 2x2 panels under viz/")
    return p.parse_args()


def _load_mask_256(mask_path: Path) -> torch.Tensor:
    """
    Load mask from disk and resize to 256x256 using nearest-neighbor, then binarize to {0,1}.
    Returns: [1,256,256] float32
    """
    m = Image.open(mask_path).convert("L")
    if m.size != TARGET_SIZE:
        m = m.resize(TARGET_SIZE, resample=Image.NEAREST)
    m_t = torch.from_numpy(np.array(m))  # local import via __import__ to keep deps minimal
    m_bin = (m_t > 0).to(dtype=torch.float32)  # [H,W]
    return m_bin.unsqueeze(0).contiguous()


class MatchedImageMaskDataset(Dataset):
    """
    Loads images using the repo's standard preprocess_image() and masks from disk via filename-stem matching.
    """

    def __init__(self, pairs: List[Tuple[Path, Path]]) -> None:
        self.pairs = pairs

        try:
            from src.data.transforms import preprocess_image
        except Exception as e:
            raise ImportError(
                "Failed to import preprocess_image from src/data/transforms.py. "
                "This script requires the repo's standard preprocessing."
            ) from e

        self._preprocess_image = preprocess_image

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_path, mask_path = self.pairs[idx]

        image = self._preprocess_image(str(img_path))  # Tensor [3,256,256] float32 in [0,1]
        if not isinstance(image, torch.Tensor) or image.shape != (3, 256, 256):
            raise ValueError(f"preprocess_image returned unexpected shape for {img_path.as_posix()}: {getattr(image, 'shape', None)}")

        mask = _load_mask_256(mask_path)  # [1,256,256] float32 {0,1}
        return {
            "image": image.contiguous(),
            "mask": mask.contiguous(),
            "meta": {"image_path": str(img_path), "mask_path": str(mask_path)},
        }


def _collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    images = torch.stack([b["image"] for b in batch], dim=0)  # [B,3,H,W]
    masks = torch.stack([b["mask"] for b in batch], dim=0)    # [B,1,H,W]
    metas = [b.get("meta", {}) for b in batch]
    return {"image": images, "mask": masks, "meta": metas}


def _index_masks_by_stem(masks_root: Path) -> Dict[str, Path]:
    masks = sorted([p for p in masks_root.rglob("*") if p.is_file() and p.suffix.lower() == ".png"])
    by_stem: Dict[str, Path] = {}
    for p in masks:
        stem = _normalize_stem(p.stem)
        if stem not in by_stem:
            by_stem[stem] = p
    return by_stem


def _collect_matched_pairs(images_root: Path, masks_root: Path) -> Tuple[List[Tuple[Path, Path]], int, int, int]:
    images_root = images_root.resolve()
    masks_root = masks_root.resolve()

    mask_by_stem = _index_masks_by_stem(masks_root)

    all_images = sorted([p for p in images_root.rglob("*") if p.is_file() and _is_image_file(p)])
    found_images = 0
    matched = 0
    skipped = 0
    pairs: List[Tuple[Path, Path]] = []

    for img_path in all_images:
        # Avoid treating masks as images if images_root includes masks_root
        try:
            if masks_root in img_path.resolve().parents:
                continue
        except Exception:
            pass

        found_images += 1
        stem = _normalize_stem(img_path.stem)
        mask_path = mask_by_stem.get(stem, None)
        if mask_path is None:
            skipped += 1
            continue
        matched += 1
        pairs.append((img_path, mask_path))

    return pairs, found_images, matched, skipped


def _build_baseline_model(device: torch.device) -> torch.nn.Module:
    """
    Reuse the baseline builder used elsewhere in the repo to avoid architecture drift.
    """
    try:
        from scripts.sanity_train import build_model as build_sanity_model
    except Exception as e:
        raise ImportError("Failed to import scripts.sanity_train.build_model for baseline UNet+ResNet50.") from e

    model = build_sanity_model("unet_r50", device=device)
    return model


def _load_checkpoint(model: torch.nn.Module, ckpt_path: Path, device: torch.device) -> torch.nn.Module:
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

    images_root = Path(args.images_root)
    masks_root = Path(args.masks_root)
    ckpt_path = Path(args.checkpoint)

    if not images_root.exists():
        print(f"images_root does not exist: {images_root.as_posix()}")
        return 2
    if not masks_root.exists():
        print(f"masks_root does not exist: {masks_root.as_posix()}")
        return 2
    if not ckpt_path.exists():
        print(f"checkpoint does not exist: {ckpt_path.as_posix()}")
        return 2

    out_root = REPO_ROOT / "outputs" / "eval" / "test" / _now_timestamp()
    preds_dir = out_root / "preds"
    probs_dir = out_root / "probs"
    viz_dir = out_root / "viz"
    preds_dir.mkdir(parents=True, exist_ok=True)
    probs_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    pairs, total_found_images, total_matched_masks, total_skipped = _collect_matched_pairs(images_root, masks_root)

    limit = int(args.limit) if args.limit is not None else None
    if limit is not None:
        pairs = pairs[:limit]

    if len(pairs) == 0:
        print("No matched image/mask pairs found. Check roots and naming rule (same stem + .png).")
        print(f"images_root={images_root.as_posix()} masks_root={masks_root.as_posix()}")
        return 2

    dataset = MatchedImageMaskDataset(pairs=pairs)
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=True,
        persistent_workers=True if int(args.num_workers) > 0 else False,
        collate_fn=_collate_fn,
    )

    try:
        from src.training.metrics import dice_score, iou_score, precision_recall
    except Exception:
        print("Failed to import metrics from src/training/metrics.py")
        traceback.print_exc()
        return 2

    try:
        model = _build_baseline_model(device)
        model = _load_checkpoint(model, ckpt_path, device=device)
    except Exception:
        print("Failed to build/load baseline model checkpoint.")
        traceback.print_exc()
        return 2

    per_image_rows: List[Dict[str, Any]] = []
    dice_list: List[float] = []
    iou_list: List[float] = []
    prec_list: List[float] = []
    rec_list: List[float] = []

    viz_saved = 0
    save_viz_n = int(args.save_viz_n)

    with torch.no_grad():
        for batch in loader:
            images = batch["image"]  # [B,3,256,256]
            masks = batch["mask"]    # [B,1,256,256]
            metas = batch["meta"]    # list[dict]

            if images.shape[1:] != (3, 256, 256):
                raise ValueError(f"Expected images [B,3,256,256], got {tuple(images.shape)}")
            if masks.shape[1:] != (1, 256, 256):
                raise ValueError(f"Expected masks [B,1,256,256], got {tuple(masks.shape)}")

            images_dev = images.to(device, non_blocking=True)
            logits = model(images_dev)

            if not isinstance(logits, torch.Tensor) or logits.shape != (images.shape[0], 1, 256, 256):
                raise ValueError(f"Expected logits [B,1,256,256], got {type(logits)} {getattr(logits, 'shape', None)}")

            probs = torch.sigmoid(logits).to(dtype=torch.float32).detach().cpu().contiguous()  # [B,1,H,W]
            pred = (probs > THRESHOLD)

            # Compute per-image metrics using the repo's metric functions (same definitions as training logs)
            for i in range(images.shape[0]):
                img_path = str(metas[i].get("image_path", ""))
                mask_path = str(metas[i].get("mask_path", ""))

                stem = Path(img_path).stem if img_path else f"sample_{len(per_image_rows):06d}"

                proba_hw = probs[i, 0]
                pred_hw = pred[i, 0]
                gt = masks[i].detach().cpu().to(dtype=torch.float32).contiguous()  # [1,H,W]

                # metric fns expect [B,1,H,W]
                probs_i = probs[i : i + 1]
                gt_i = gt.unsqueeze(0)  # [1,1,H,W]

                d = float(dice_score(probs_i, gt_i, threshold=THRESHOLD))
                j = float(iou_score(probs_i, gt_i, threshold=THRESHOLD))

                p_raw, r_raw = precision_recall(probs_i, gt_i, threshold=THRESHOLD)
                p = float(p_raw)
                r = float(r_raw)

                dice_list.append(d)
                iou_list.append(j)
                prec_list.append(p)
                rec_list.append(r)

                per_image_rows.append(
                    {
                        "image_path": img_path,
                        "mask_path": mask_path,
                        "dice": d,
                        "iou": j,
                        "precision": p,
                        "recall": r,
                    }
                )

                # Save qualitative outputs
                _save_u8_png(probs_dir / f"{stem}_proba.png", _proba_to_u8(proba_hw))
                _save_u8_png(preds_dir / f"{stem}_pred.png", _mask_to_u8(pred_hw))

                if viz_saved < save_viz_n:
                    _save_viz_panel(
                        viz_dir / f"{stem}_panel.png",
                        image_chw=images[i].detach().cpu(),
                        gt_hw=gt[0],
                        pred_hw=pred_hw,
                        proba_hw=proba_hw,
                    )
                    viz_saved += 1

    # Write per-image CSV
    metrics_csv_path = out_root / "metrics.csv"
    with metrics_csv_path.open("w", newline="") as f:
        fieldnames = ["image_path", "mask_path", "dice", "iou", "precision", "recall"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in per_image_rows:
            w.writerow(row)

    # Aggregate summary (mean + std over images)
    def _mean_std(xs: List[float]) -> Tuple[float, float]:
        if len(xs) == 0:
            return 0.0, 0.0
        # Use population std (pstdev) for a stable "spread" summary.
        return float(mean(xs)), float(pstdev(xs))

    summary = {
        "count": len(per_image_rows),
        "threshold": THRESHOLD,
        "dice": {"mean": _mean_std(dice_list)[0], "std": _mean_std(dice_list)[1]},
        "iou": {"mean": _mean_std(iou_list)[0], "std": _mean_std(iou_list)[1]},
        "precision": {"mean": _mean_std(prec_list)[0], "std": _mean_std(prec_list)[1]},
        "recall": {"mean": _mean_std(rec_list)[0], "std": _mean_std(rec_list)[1]},
    }

    summary_json_path = out_root / "metrics_summary.json"
    summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        f"Found images: {total_found_images} | matched masks: {total_matched_masks} | skipped: {total_skipped} | evaluated: {len(per_image_rows)}"
    )
    print(
        f"Dice mean/std: {summary['dice']['mean']:.4f}/{summary['dice']['std']:.4f} | "
        f"IoU mean/std: {summary['iou']['mean']:.4f}/{summary['iou']['std']:.4f} | "
        f"Precision mean/std: {summary['precision']['mean']:.4f}/{summary['precision']['std']:.4f} | "
        f"Recall mean/std: {summary['recall']['mean']:.4f}/{summary['recall']['std']:.4f}"
    )
    print(f"Outputs saved to: {out_root.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())