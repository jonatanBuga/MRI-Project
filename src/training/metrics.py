from __future__ import annotations

from typing import Tuple

import torch


def _check_shapes(probs: torch.Tensor, targets: torch.Tensor, name: str) -> None:
    if probs.ndim != 4 or targets.ndim != 4:
        raise ValueError(
            f"{name}: expected tensors with shape [B,1,H,W]. "
            f"Got probs shape={tuple(probs.shape)}, targets shape={tuple(targets.shape)}"
        )
    if probs.shape != targets.shape:
        raise ValueError(f"{name}: shape mismatch probs={tuple(probs.shape)} targets={tuple(targets.shape)}")
    if probs.shape[1] != 1:
        raise ValueError(f"{name}: expected probs with C=1, got C={probs.shape[1]}")
    if targets.shape[1] != 1:
        raise ValueError(f"{name}: expected targets with C=1, got C={targets.shape[1]}")


def _binarize(probs: torch.Tensor, targets: torch.Tensor, threshold: float) -> Tuple[torch.Tensor, torch.Tensor]:
    probs = probs.to(dtype=torch.float32)
    targets = targets.to(dtype=torch.float32)

    _check_shapes(probs, targets, "_binarize")
    pred = (probs > float(threshold))
    tgt = (targets > 0.5)
    return pred, tgt


def dice_score(probs: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6) -> float:
    """
    Dice score on binarized predictions (computed over full batch).
    """
    pred, tgt = _binarize(probs, targets, threshold)
    pred_f = pred.reshape(-1).to(dtype=torch.float32)
    tgt_f = tgt.reshape(-1).to(dtype=torch.float32)

    inter = (pred_f * tgt_f).sum()
    denom = pred_f.sum() + tgt_f.sum()
    dice = (2.0 * inter + eps) / (denom + eps)
    return float(dice.item())


def iou_score(probs: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6) -> float:
    """
    IoU (Jaccard) on binarized predictions (computed over full batch).
    """
    pred, tgt = _binarize(probs, targets, threshold)
    pred_f = pred.reshape(-1)
    tgt_f = tgt.reshape(-1)

    inter = (pred_f & tgt_f).sum().to(dtype=torch.float32)
    union = (pred_f | tgt_f).sum().to(dtype=torch.float32)
    iou = (inter + eps) / (union + eps)
    return float(iou.item())


def precision_recall(
    probs: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6
) -> Tuple[float, float]:
    """
    Precision/Recall on binarized predictions (computed over full batch).
    """
    pred, tgt = _binarize(probs, targets, threshold)
    pred_f = pred.reshape(-1)
    tgt_f = tgt.reshape(-1)

    tp = (pred_f & tgt_f).sum().to(dtype=torch.float32)
    fp = (pred_f & ~tgt_f).sum().to(dtype=torch.float32)
    fn = (~pred_f & tgt_f).sum().to(dtype=torch.float32)

    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    return float(precision.item()), float(recall.item())