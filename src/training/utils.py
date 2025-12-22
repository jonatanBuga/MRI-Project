from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    Best-effort reproducibility for Python/NumPy/PyTorch.
    """
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Determinism knobs (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class AverageMeter:
    """
    Running average tracker (optionally weighted).
    """
    name: str = "meter"
    val: float = 0.0
    sum: float = 0.0
    count: int = 0

    def reset(self) -> None:
        self.val = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        v = float(value)
        self.val = v
        self.sum += v * int(n)
        self.count += int(n)

    @property
    def avg(self) -> float:
        if self.count == 0:
            return 0.0
        return self.sum / self.count