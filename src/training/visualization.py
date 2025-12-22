from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image


def _to_u8_image_rgb(image_chw: torch.Tensor) -> torch.Tensor:
    """
    image_chw: [3,H,W] float in [0,1] -> uint8 [H,W,3]
    """
    if not isinstance(image_chw, torch.Tensor):
        raise TypeError(f"image_chw must be a torch.Tensor, got {type(image_chw)}")
    if image_chw.ndim != 3 or image_chw.shape[0] != 3:
        raise ValueError(f"Expected image_chw with shape [3,H,W], got {tuple(image_chw.shape)}")

    x = image_chw.detach().to(dtype=torch.float32).cpu().clamp(0.0, 1.0)
    x = (x * 255.0).round().to(dtype=torch.uint8)
    x = x.permute(1, 2, 0).contiguous()  # [H,W,3]
    return x


def _to_u8_mask(mask_hw: torch.Tensor) -> torch.Tensor:
    """
    mask_hw: [H,W] float/bool -> uint8 [H,W] in {0,255}
    """
    if not isinstance(mask_hw, torch.Tensor):
        raise TypeError(f"mask_hw must be a torch.Tensor, got {type(mask_hw)}")
    if mask_hw.ndim != 2:
        raise ValueError(f"Expected mask_hw with shape [H,W], got {tuple(mask_hw.shape)}")

    m = mask_hw.detach().cpu()
    if m.dtype == torch.bool:
        m_bin = m
    else:
        m_bin = m.to(dtype=torch.float32) > 0.5
    return (m_bin.to(dtype=torch.uint8) * 255).contiguous()


def save_png_triplet(
    out_dir: Path,
    stem: str,
    image_chw: torch.Tensor,   # [3,H,W] float in [0,1]
    gt_hw: torch.Tensor,       # [H,W] float/bool
    pred_hw: torch.Tensor,     # [H,W] float/bool or proba
) -> None:
    """
    Saves:
      - {stem}_img.png  (RGB)
      - {stem}_gt.png   (0/255)
      - {stem}_pred.png (0/255)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img_u8_hwc = _to_u8_image_rgb(image_chw)
    gt_u8 = _to_u8_mask(gt_hw)
    pred_u8 = _to_u8_mask(pred_hw)

    Image.fromarray(img_u8_hwc.numpy(), mode="RGB").save(out_dir / f"{stem}_img.png")
    Image.fromarray(gt_u8.numpy(), mode="L").save(out_dir / f"{stem}_gt.png")
    Image.fromarray(pred_u8.numpy(), mode="L").save(out_dir / f"{stem}_pred.png")