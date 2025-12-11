from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import cv2


def validate_mask(
    mask: Union[np.ndarray, str, Path],
    expected_shape: Optional[Tuple[int, int]] = None,
    min_foreground_ratio: float = 1e-4,
    max_foreground_ratio: float = 0.9,
) -> Dict[str, Any]:
    """
    Run basic sanity checks on a single segmentation mask.

    Parameters
    ----------
    mask : np.ndarray or path-like
        Either a 2D NumPy array or a path to a grayscale PNG on disk.
    expected_shape : optional (H, W)
        If provided, we check that the mask shape matches this.
    min_foreground_ratio : float
        Minimum acceptable fraction of foreground pixels (0..1).
    max_foreground_ratio : float
        Maximum acceptable fraction of foreground pixels (0..1).

    Returns
    -------
    report : dict
        A dictionary with keys:
          - "shape": (H, W)
          - "matches_shape": bool or None
          - "unique_values": list of unique values in the raw mask
          - "is_binary_like": bool  (only 0/1 or 0/255 or small noise)
          - "foreground_pixels": int
          - "total_pixels": int
          - "foreground_ratio": float
          - "is_empty": bool
          - "too_small": bool
          - "too_large": bool
          - "is_valid": bool
    """
    # Load if path-like
    if isinstance(mask, (str, Path)):
        p = Path(mask)
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not load mask image from path: {mask}")
        arr = img
    elif isinstance(mask, np.ndarray):
        arr = mask
    else:
        raise TypeError("mask must be a numpy array or a path-like string/Path")

    # Ensure 2D array
    if arr.ndim == 3:
        # If 3 channels, convert to grayscale by taking the first channel (assumes identical channels)
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY) if arr.shape[2] == 3 else arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"Mask must be 2D array after conversion, got shape {arr.shape}")

    # Convert to numeric array
    arr = np.asarray(arr)
    H, W = arr.shape
    shape = (H, W)

    # Unique raw values
    try:
        unique_vals = np.unique(arr)
        unique_list = [int(v) for v in unique_vals.tolist()]
    except Exception:
        unique_list = []

    # Treat any non-zero as foreground
    binary = (arr > 0).astype(np.uint8)
    foreground_pixels = int(binary.sum())
    total_pixels = int(binary.size)
    foreground_ratio = float(foreground_pixels) / float(total_pixels) if total_pixels > 0 else 0.0

    # Determine if mask is binary-like:
    # Normalize by 255 if max>1 (common for 0/255 masks), otherwise keep as-is.
    max_val = float(arr.max()) if arr.size else 0.0
    if max_val > 1.0:
        norm = arr.astype(np.float32) / 255.0
    else:
        norm = arr.astype(np.float32)
    uniq_norm = np.unique(norm)

    eps = 1e-3
    # Compute distance to nearest of {0,1} for each unique value
    deviations = np.minimum(np.abs(uniq_norm - 0.0), np.abs(uniq_norm - 1.0))
    max_deviation = float(np.max(deviations)) if deviations.size else 0.0
    is_binary_like = bool(max_deviation <= eps)

    # Shape check
    matches_shape: Optional[bool] = None
    if expected_shape is not None:
        matches_shape = (shape == expected_shape)

    # Threshold flags
    is_empty = (foreground_pixels == 0)
    too_small = (foreground_ratio < min_foreground_ratio) and (not is_empty)
    too_large = (foreground_ratio > max_foreground_ratio)

    # Validity
    shape_ok = (matches_shape is None) or (matches_shape is True)
    is_valid = bool(
        is_binary_like and shape_ok and (not is_empty) and (not too_small) and (not too_large)
    )

    report: Dict[str, Any] = {
        "shape": shape,
        "matches_shape": matches_shape,
        "unique_values": unique_list,
        "is_binary_like": is_binary_like,
        "foreground_pixels": foreground_pixels,
        "total_pixels": total_pixels,
        "foreground_ratio": foreground_ratio,
        "is_empty": is_empty,
        "too_small": too_small,
        "too_large": too_large,
        "is_valid": is_valid,
    }
    return report


def validate_mask_file(
    mask_path: Union[str, Path],
    expected_shape: Optional[Tuple[int, int]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Convenience wrapper that calls `validate_mask` on a mask file path.
    """
    return validate_mask(mask_path, expected_shape=expected_shape, **kwargs)


if __name__ == "__main__":
    # Simple demo: validate a few masks from data/masks_raw/train
    project_root = Path(__file__).resolve().parents[2]
    masks_dir = project_root / "data" / "masks_raw" / "train"

    if not masks_dir.exists():
        print(f"No masks directory found at: {masks_dir}")
    else:
        mask_files = sorted(masks_dir.glob("*.png"))
        sample_files = mask_files[:5]

        print(f"Found {len(mask_files)} mask files, validating {len(sample_files)} samples:\n")
        for p in sample_files:
            try:
                report = validate_mask_file(p)
            except Exception as e:
                print(f"Error validating {p}: {e}")
                continue
            print(f"Mask: {p}")
            print(
                f"  shape={report['shape']}, unique={report['unique_values']}, "
                f"foreground_ratio={report['foreground_ratio']:.6f}, "
                f"is_binary_like={report['is_binary_like']}, is_valid={report['is_valid']}"
            )
            print()
