from __future__ import annotations

from typing import Any, Dict

import torch

from src.training.metrics import dice_score, iou_score, precision_recall
from src.training.utils import AverageMeter


def _extract_batch(batch: Dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(batch, dict):
        raise TypeError(f"Expected batch to be dict, got: {type(batch)}")

    if "image" not in batch or "mask" not in batch:
        raise KeyError(f"Batch must contain keys 'image' and 'mask'. Got keys: {list(batch.keys())}")

    images = batch["image"]
    masks = batch["mask"]

    if not isinstance(images, torch.Tensor) or not isinstance(masks, torch.Tensor):
        raise TypeError(f"Batch['image'] and Batch['mask'] must be torch.Tensor. Got {type(images)} and {type(masks)}")

    if images.ndim != 4:
        raise ValueError(f"Expected images shape [B,3,H,W], got {tuple(images.shape)}")
    if masks.ndim != 4:
        raise ValueError(f"Expected masks shape [B,1,H,W], got {tuple(masks.shape)}")
    if images.shape[0] != masks.shape[0]:
        raise ValueError(f"Batch size mismatch: images B={images.shape[0]} masks B={masks.shape[0]}")
    if images.shape[1] != 3:
        raise ValueError(f"Expected images C=3, got C={images.shape[1]}")
    if masks.shape[1] != 1:
        raise ValueError(f"Expected masks C=1, got C={masks.shape[1]}")
    if images.shape[2:] != masks.shape[2:]:
        raise ValueError(f"Spatial mismatch: images {tuple(images.shape[2:])} masks {tuple(masks.shape[2:])}")

    return images, masks.to(dtype=torch.float32)


def _check_logits_shape(logits: torch.Tensor, images: torch.Tensor, masks: torch.Tensor) -> None:
    if not isinstance(logits, torch.Tensor):
        raise TypeError(f"Model output must be torch.Tensor, got {type(logits)}")
    if logits.ndim != 4:
        raise ValueError(f"Expected logits shape [B,1,H,W], got {tuple(logits.shape)}")
    if logits.shape[0] != images.shape[0]:
        raise ValueError(f"Logits batch mismatch: logits B={logits.shape[0]} images B={images.shape[0]}")
    if logits.shape[1] != 1:
        raise ValueError(f"Expected logits C=1 for binary segmentation, got C={logits.shape[1]}")
    if logits.shape[2:] != masks.shape[2:]:
        raise ValueError(f"Logits spatial mismatch: logits {tuple(logits.shape[2:])} masks {tuple(masks.shape[2:])}")


def train_one_epoch(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    loss_fn,
    device: torch.device,
    epoch: int,
    log_every: int = 20,
) -> Dict[str, float]:
    model.train()
    loss_meter = AverageMeter(name="train_loss")

    for step, batch in enumerate(loader):
        images, masks = _extract_batch(batch)
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        logits = model(images)
        _check_logits_shape(logits, images, masks)

        loss = loss_fn(logits, masks)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        loss_meter.update(float(loss.item()), n=images.shape[0])

        if log_every and (step + 1) % int(log_every) == 0:
            print(f"[train][epoch {epoch} step {step+1}] loss={loss_meter.avg:.4f}")

    return {"loss": loss_meter.avg}


def validate_one_epoch(
    model: torch.nn.Module,
    loader,
    loss_fn,
    device: torch.device,
    threshold: float = 0.5,
) -> Dict[str, float]:
    model.eval()

    loss_meter = AverageMeter(name="val_loss")
    dice_meter = AverageMeter(name="val_dice")
    iou_meter = AverageMeter(name="val_iou")
    prec_meter = AverageMeter(name="val_precision")
    rec_meter = AverageMeter(name="val_recall")

    thr = float(threshold)

    with torch.inference_mode():
        for batch in loader:
            images, masks = _extract_batch(batch)
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            logits = model(images)
            _check_logits_shape(logits, images, masks)

            loss = loss_fn(logits, masks)
            loss_meter.update(float(loss.item()), n=images.shape[0])

            probs = torch.sigmoid(logits).to(dtype=torch.float32)

            # Metrics on CPU for stability / consistent reductions across backends
            probs_cpu = probs.detach().cpu()
            masks_cpu = masks.detach().cpu()

            d = dice_score(probs_cpu, masks_cpu, threshold=thr)
            j = iou_score(probs_cpu, masks_cpu, threshold=thr)
            p, r = precision_recall(probs_cpu, masks_cpu, threshold=thr)

            # Treat metrics as sample-weighted averages across batches
            dice_meter.update(d, n=images.shape[0])
            iou_meter.update(j, n=images.shape[0])
            prec_meter.update(p, n=images.shape[0])
            rec_meter.update(r, n=images.shape[0])

    return {
        "loss": loss_meter.avg,
        "dice": dice_meter.avg,
        "iou": iou_meter.avg,
        "precision": prec_meter.avg,
        "recall": rec_meter.avg,
    }