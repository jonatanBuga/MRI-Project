from pathlib import Path
from typing import Union
import numpy as np
from PIL import Image
import cv2 
import torch 

def load_image(path: Union[str, Path]) -> np.ndarray:
    """
    Load an image from disk and return a NumPy array (H, W, C) with dtype=float32.

    - Accepts a filesystem path (str or Path).
    - Loads image using PIL.
    - Converts grayscale images to 3 channels (H, W, 3).
    - Does NOT perform resizing or normalization.
    - Returns a float32 NumPy array.

    Raises:
        IOError: if the file cannot be opened/read.
    """
    p = Path(path)
    try:
        im = Image.open(p)
    except Exception as e:
        raise IOError(f"Unable to open image '{p}': {e}")

    # Convert to RGB so grayscale -> 3 channels; RGB images remain 3-channel.
    im = im.convert("RGB")

    arr = np.asarray(im, dtype=np.float32)  # shape (H, W, 3)
    # Ensure output shape (H, W, C)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.concatenate([arr, arr, arr], axis=2)

    return arr

def resize_image(img: np.ndarray, target_size: int) -> np.ndarray:
    """
    Resize an image array to (target_size, target_size).

    - Inputs:
        img: NumPy array with shape (H, W) or (H, W, C).
        target_size: desired integer size for both height and width.
    - Uses cv2.INTER_AREA for downscaling and cv2.INTER_LINEAR for upscaling.
    - Preserves and returns dtype float32.
    - Does NOT perform normalization or channel reordering.

    Returns:
        Resized NumPy array with shape (target_size, target_size) or
        (target_size, target_size, C) and dtype float32.
    """
    if not isinstance(img, np.ndarray):
        raise TypeError("img must be a NumPy array")
    if img.ndim not in (2, 3):
        raise ValueError(f"img must have 2 or 3 dimensions, got shape {img.shape}")
    if not isinstance(target_size, int) or target_size <= 0:
        raise ValueError("target_size must be a positive integer")

    h, w = img.shape[:2]
    if h == target_size and w == target_size:
        return img.astype(np.float32, copy=False)

    interp = cv2.INTER_AREA if target_size < min(h, w) else cv2.INTER_LINEAR
    # cv2.resize accepts multi-channel arrays directly; dsize is (width, height)
    resized = cv2.resize(img, (target_size, target_size), interpolation=interp)

    return resized.astype(np.float32, copy=False)

def normalize_image(img: np.ndarray) -> np.ndarray:
    """
    Min-max normalize an image array to the range [0, 1].

    - Input: NumPy array with pixel values expected in [0, 255] (shape (H,W) or (H,W,C)).
    - Performs min-max scaling to [0,1]. Does NOT perform Z-score standardization.
    - If the image is constant (max == min) the function returns an array of zeros.
    - Ensures returned dtype is float32.

    Returns:
        NumPy array (same shape) with values in [0,1] and dtype float32.
    """
    if not isinstance(img, np.ndarray):
        raise TypeError("img must be a NumPy array")

    arr = img.astype(np.float32, copy=False)
    # Clip to expected input range to avoid unexpected values
    arr = np.clip(arr, 0.0, 255.0)

    mn = float(np.min(arr))
    mx = float(np.max(arr))
    if mx <= mn:
        # flat image -> return zeros (no dynamic range)
        return np.zeros_like(arr, dtype=np.float32)

    out = (arr - mn) / (mx - mn)
    return out.astype(np.float32, copy=False)

def preprocess_image(path: Union[str, Path], target_size: int = 256) -> torch.Tensor:
    """
    Full preprocessing pipeline: load -> resize -> normalize -> to-tensor (CHW).

    - Input: filesystem path (str or Path).
    - Pipeline:
        1. load_image(path)         -> NumPy HWC float32 in [0,255]
        2. resize_image(img, size)  -> NumPy HWC float32 (size x size)
        3. TODO: orientation correction (mirror/rotation) will be added here
        4. normalize_image(img)     -> NumPy HWC float32 in [0,1]
        5. convert to torch.Tensor, reorder to CHW, dtype=torch.float32
    - Returns: torch.Tensor with shape (C, H, W) and dtype torch.float32.

    Note: target_size defaults to 256; pass a different size if required.
    """
    # 1) load
    img = load_image(path)

    # 2) resize
    img = resize_image(img, target_size)

    # 3) orientation correction placeholder
    # TODO: apply orientation correction (rotation / mirroring) here if needed

    # 4) normalize to [0,1]
    img = normalize_image(img)

    # 5) convert to tensor and reorder to CHW
    img = np.ascontiguousarray(img, dtype=np.float32)  # ensure contiguous
    tensor = torch.from_numpy(img)                      # H x W x C
    tensor = tensor.permute(2, 0, 1).contiguous().to(dtype=torch.float32)  # C x H x W

    return tensor

if __name__ == "__main__":
    """
    Simple sanity check:
    - Search data/raw/ for image files (*.png, *.jpg, *.jpeg)
    - Randomly select 5-10 images
    - Run preprocess_image(path) and print tensor shape, dtype, min, max
    """
    import random

    # locate project root (two levels up from src/data/transforms.py) and data/raw
    project_root = Path(__file__).resolve().parents[2]
    raw_dir = project_root / "data" / "raw"

    patterns = ["*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"]
    files = []
    if raw_dir.exists():
        for pat in patterns:
            files.extend(list(raw_dir.rglob(pat)))
    else:
        print(f"Raw data directory not found: {raw_dir}")

    files = [p for p in files if p.is_file()]
    if not files:
        print(f"No image files found under {raw_dir}.")
    else:
        n_samples = min(10, max(5, len(files)))
        sample_paths = random.sample(files, n_samples) if len(files) >= n_samples else files

        for p in sample_paths:
            try:
                tensor = preprocess_image(p)
                t_min = float(tensor.min().item())
                t_max = float(tensor.max().item())
                print(f"{p} -> shape={tuple(tensor.shape)}, dtype={tensor.dtype}, min={t_min:.6f}, max={t_max:.6f}")
            except Exception as exc:
                print(f"Error processing {p}: {exc}")