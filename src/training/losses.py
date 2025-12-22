from __future__ import annotations

import torch
import torch.nn.functional as F


def _check_shapes(probs_or_logits: torch.Tensor, targets: torch.Tensor, name: str) -> None:
    if not isinstance(probs_or_logits, torch.Tensor) or not isinstance(targets, torch.Tensor):
        raise TypeError(f"{name}: expected torch.Tensor inputs, got {type(probs_or_logits)} and {type(targets)}")

    if probs_or_logits.ndim != 4 or targets.ndim != 4:
        raise ValueError(
            f"{name}: expected tensors with shape [B,1,H,W]. "
            f"Got {name} shape={tuple(probs_or_logits.shape)}, targets shape={tuple(targets.shape)}"
        )
    if probs_or_logits.shape != targets.shape:
        raise ValueError(
            f"{name}: shape mismatch. {name} shape={tuple(probs_or_logits.shape)} vs targets shape={tuple(targets.shape)}"
        )
    if probs_or_logits.shape[1] != 1 or targets.shape[1] != 1:
        raise ValueError(
            f"{name}: expected single-channel tensors [B,1,H,W]. "
            f"Got {name} C={probs_or_logits.shape[1]}, targets C={targets.shape[1]}"
        )


def dice_loss(probs: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Dice loss for binary segmentation.

    Args:
      probs:   [B,1,H,W] float32 in [0,1]
      targets: [B,1,H,W] float32 in {0,1} (or [0,1])
    """
    targets = targets.to(dtype=torch.float32)
    probs = probs.to(dtype=torch.float32)
    _check_shapes(probs, targets, "dice_loss")

    # Compute Dice over the whole batch (stable for small batch sizes)
    probs_f = probs.reshape(probs.shape[0], -1)
    targets_f = targets.reshape(targets.shape[0], -1)

    intersection = (probs_f * targets_f).sum(dim=1)
    denom = probs_f.sum(dim=1) + targets_f.sum(dim=1)
    dice = (2.0 * intersection + eps) / (denom + eps)

    return 1.0 - dice.mean()


def bce_with_logits(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    BCEWithLogits for binary segmentation.

    Args:
      logits:  [B,1,H,W] float32 (pre-sigmoid)
      targets: [B,1,H,W] float32 in {0,1}
    """
    targets = targets.to(dtype=torch.float32)
    logits = logits.to(dtype=torch.float32)
    _check_shapes(logits, targets, "bce_with_logits")

    return F.binary_cross_entropy_with_logits(logits, targets)


def dice_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    dice_weight: float = 1.0,
    bce_weight: float = 1.0,
) -> torch.Tensor:
    """
    Combined Dice + BCE loss (common baseline for binary segmentation).

    Args:
      logits:  [B,1,H,W] float32 (pre-sigmoid)
      targets: [B,1,H,W] float32 in {0,1}
    """
    if dice_weight < 0 or bce_weight < 0:
        raise ValueError(f"dice_weight and bce_weight must be >= 0. Got {dice_weight=}, {bce_weight=}")

    targets = targets.to(dtype=torch.float32)
    logits = logits.to(dtype=torch.float32)
    _check_shapes(logits, targets, "dice_bce_loss")

    probs = torch.sigmoid(logits)
    dl = dice_loss(probs, targets)
    bl = bce_with_logits(logits, targets)
    return dice_weight * dl + bce_weight * bl