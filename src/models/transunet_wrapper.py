from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn

from src.utils.config import TARGET_SIZE


def _repo_root_from_here() -> Path:
    # src/models/transunet_wrapper.py -> parents[2] == <repo_root>
    return Path(__file__).resolve().parents[2]


def _ensure_transunet_on_syspath() -> Path:
    """
    Adds the vendored TransUNet code to sys.path.

    Supports either:
      - third_party/transunet/                (repo root contents)
      - third_party/transunet/TransUNet/      (nested repo)
    """
    repo_root = _repo_root_from_here()
    base = repo_root / "third_party" / "transunet"

    candidates = [
        base / "TransUNet",  # Layout 2
        base,                # Layout 1
    ]

    for p in candidates:
        if (p / "networks").exists():
            sys.path.insert(0, str(p))
            return p

    raise ImportError(
        "TransUNet code not found under third_party/transunet.\n"
        "Expected either:\n"
        "  third_party/transunet/networks/\n"
        "or:\n"
        "  third_party/transunet/TransUNet/networks/\n\n"
        "See: third_party/transunet/README.md"
    )


class TransUNetWrapper(nn.Module):
    """
    Thin wrapper around Beckschen/TransUNet model for our repo.

    Goals:
      - Keep imports robust (vendored under third_party/transunet)
      - Build the common 'R50-ViT-B_16' config
      - Ensure forward(x) returns logits shaped [B, 1, H, W] for binary segmentation

    Notes:
      - Our preprocessing already resizes to a square TARGET_SIZE.
      - TransUNet typically expects img_size divisible by 16.
      - Some upstream configs/models output [B, 2, H, W] (softmax-style). We adapt to 1 logit channel.
    """

    def __init__(
        self,
        img_size: int = TARGET_SIZE,
        vit_name: str = "R50-ViT-B_16",
        num_classes: int = 1,
    ) -> None:
        super().__init__()

        if img_size <= 0:
            raise ValueError(f"img_size must be > 0, got: {img_size}")
        if img_size % 16 != 0:
            raise ValueError(
                f"TransUNet expects img_size divisible by 16. Got img_size={img_size}. "
                "Fix by setting TARGET_SIZE to a multiple of 16 (e.g., 256)."
            )
        if num_classes not in (1, 2):
            raise ValueError("For this project wrapper, num_classes should be 1 (preferred) or 2.")

        self.img_size = int(img_size)
        self.vit_name = str(vit_name)
        self.num_classes = int(num_classes)

        # Ensure vendored code import works
        self._vendor_root = _ensure_transunet_on_syspath()

        try:
            # Upstream TransUNet imports (official repo structure)
            from networks.vit_seg_modeling import VisionTransformer as ViT_seg  # type: ignore
            from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg  # type: ignore
        except Exception as e:
            raise ImportError(
                "Failed to import TransUNet modules from vendored code.\n"
                "Expected to import:\n"
                "  networks.vit_seg_modeling (VisionTransformer, CONFIGS)\n\n"
                "Check that third_party/transunet contains the upstream repo.\n"
                "See third_party/transunet/README.md"
            ) from e

        if self.vit_name not in CONFIGS_ViT_seg:
            available = ", ".join(sorted(list(CONFIGS_ViT_seg.keys())))
            raise ValueError(f"Unknown vit config {self.vit_name!r}. Available: {available}")

        config = CONFIGS_ViT_seg[self.vit_name]
        # Upstream uses n_classes + n_skip in common training scripts
        config.n_classes = self.num_classes
        # n_skip controls number of skip connections (common default: 3)
        if hasattr(config, "n_skip"):
            config.n_skip = 3

        # For R50-based hybrid, upstream expects patch grid set from img_size
        if "R50" in self.vit_name and hasattr(config, "patches") and hasattr(config.patches, "grid"):
            config.patches.grid = (self.img_size // 16, self.img_size // 16)

        # Instantiate model. Some upstream versions accept (config, img_size, num_classes),
        # others accept (config, img_size).
        try:
            self.model = ViT_seg(config, img_size=self.img_size, num_classes=self.num_classes)
        except TypeError:
            self.model = ViT_seg(config, img_size=self.img_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:
          x: [B, 3, H, W] float32 in [0,1]

        Output:
          logits: [B, 1, H, W] float32 (pre-sigmoid)
        """
        if x.ndim != 4:
            raise ValueError(f"Expected input [B,3,H,W], got shape={tuple(x.shape)}")
        b, c, h, w = x.shape
        if c != 3:
            raise ValueError(f"Expected 3-channel input, got C={c}")
        if h != w:
            raise ValueError(f"Expected square input, got H={h}, W={w}")
        if h != self.img_size:
            raise ValueError(
                f"Input size mismatch. TransUNetWrapper was built with img_size={self.img_size} "
                f"but got H=W={h}. Ensure preprocessing resizes to TARGET_SIZE."
            )

        out: Any = self.model(x)

        # Some implementations return tuple/list; first element is logits
        if isinstance(out, (tuple, list)) and len(out) > 0:
            out = out[0]

        if not isinstance(out, torch.Tensor):
            raise TypeError(f"TransUNet forward returned non-tensor type: {type(out)}")

        if out.ndim != 4:
            raise ValueError(f"Expected logits [B,C,H,W], got shape={tuple(out.shape)}")

        if out.shape[0] != b or out.shape[2] != h or out.shape[3] != w:
            raise ValueError(
                f"Unexpected logits shape {tuple(out.shape)} for input {tuple(x.shape)}. "
                "Check vendored TransUNet version/config."
            )

        # Adapt channels to binary logit [B,1,H,W]
        if out.shape[1] == 1:
            return out
        if out.shape[1] >= 2:
            # Common convention for 2-class logits: channel 1 is foreground
            return out[:, 1:2, :, :]

        raise ValueError(f"Invalid channel dimension in logits: {out.shape[1]}")