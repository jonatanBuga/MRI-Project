"""
End-to-end pipeline for lesion/edema quantification using UNet-ResNet50.
NIfTI (.nii) -> slice-wise inference -> 3D binary mask -> lesion volume calculation.
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import nibabel as nib
import torch
from PIL import Image
from skimage.transform import resize
from scipy import ndimage
import segmentation_models_pytorch as smp


# Minimum voxel count for connected components to be retained
MIN_VOXELS = 30


def parse_args():
    parser = argparse.ArgumentParser(
        description="Lesion quantification from NIfTI using UNet-ResNet50"
    )
    parser.add_argument(
        "--nii_path",
        type=str,
        default="20250702_1051542T2TurboRAREnex4Rare10s50001a001.nii",
        help="Path to input NIfTI file"
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="outputs/training/baseline_full_v2/checkpoints/best_by_val_dice.pt",
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--thr",
        type=float,
        default=0.5,
        help="Sigmoid threshold for binary mask"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="outputs/quantification",
        help="Output directory"
    )
    parser.add_argument(
        "--min_voxels",
        type=int,
        default=MIN_VOXELS,
        help="Minimum voxels for connected component filtering"
    )
    return parser.parse_args()


def load_nifti(nii_path: str) -> tuple:
    """
    Load NIfTI file and extract volume data and metadata.
    
    Returns:
        volume: 3D numpy array (H, W, Z)
        x_res, y_res, z_res: voxel spacing
        spatial_units: string describing spatial units
        header: NIfTI header object
    """
    img = nib.load(nii_path)
    volume = img.get_fdata()
    header = img.header
    
    # Extract voxel spacing (pixdim)
    zooms = header.get_zooms()
    x_res, y_res, z_res = zooms[0], zooms[1], zooms[2]
    
    # Validate spacing
    if x_res <= 0 or y_res <= 0 or z_res <= 0:
        raise ValueError(f"Invalid voxel spacing: ({x_res}, {y_res}, {z_res})")
    
    # Get spatial units
    spatial_units, time_units = header.get_xyzt_units()
    
    # Warn if units are not mm
    if spatial_units != "mm":
        warnings.warn(
            f"Spatial units are '{spatial_units}', not 'mm'. "
            f"Volume will be computed in {spatial_units}^3."
        )
    
    print(f"Loaded NIfTI: {nii_path}")
    print(f"  Shape: {volume.shape}")
    print(f"  Voxel spacing: x={x_res}, y={y_res}, z={z_res} {spatial_units}")
    
    return volume, x_res, y_res, z_res, spatial_units, header


def normalize_volume(volume: np.ndarray) -> np.ndarray:
    """
    Apply volume-level percentile normalization.
    
    Why percentile-based normalization?
    - MRI intensities can have outliers (noise, artifacts at edges)
    - Using 1st and 99th percentiles clips extreme values
    - This provides consistent intensity scaling across all slices
    - Ensures the model sees similar intensity distributions as during training
    
    Args:
        volume: 3D numpy array (H, W, Z), raw intensities
    
    Returns:
        Normalized 3D volume in [0, 1] range
    """
    # Compute intensity percentiles across the entire volume
    # Using 1st and 99th percentiles to exclude outliers
    p_low = np.percentile(volume, 1)
    p_high = np.percentile(volume, 99)
    
    print(f"  Volume normalization: p1={p_low:.2f}, p99={p_high:.2f}")
    
    # Clip volume to percentile range
    # This removes extreme outlier intensities that could skew normalization
    volume_clipped = np.clip(volume, p_low, p_high)
    
    # Normalize to [0, 1] using the clipped range
    # Handle edge case where p_low == p_high (constant volume)
    if p_high - p_low > 1e-8:
        volume_normalized = (volume_clipped - p_low) / (p_high - p_low)
    else:
        warnings.warn("Volume has constant intensity, returning zeros")
        volume_normalized = np.zeros_like(volume_clipped)
    
    return volume_normalized.astype(np.float32)


def preprocess_slice(slice_2d: np.ndarray, target_size: int = 256) -> np.ndarray:
    """
    Preprocess a single 2D slice for model input.
    
    NOTE: This function now expects the slice to be ALREADY normalized
    at the volume level. No per-slice normalization is performed here.
    
    Args:
        slice_2d: 2D array (H, W), already normalized to [0, 1]
        target_size: resize target (256x256)
    
    Returns:
        Preprocessed array (3, target_size, target_size) ready for model
    """
    # Cast to float32 (should already be, but ensure consistency)
    slice_2d = slice_2d.astype(np.float32)
    
    # Resize to target_size x target_size
    # Using bilinear interpolation for smooth intensity resampling
    slice_resized = resize(
        slice_2d,
        (target_size, target_size),
        order=1,  # bilinear interpolation
        preserve_range=True,
        anti_aliasing=True
    ).astype(np.float32)
    
    # Convert grayscale to 3 channels for ResNet encoder
    # The encoder expects RGB input, so we replicate the grayscale channel
    slice_3ch = np.stack([slice_resized] * 3, axis=0)
    
    return slice_3ch


def load_model_from_ckpt(ckpt_path: str, device: torch.device) -> torch.nn.Module:
    """
    Load UNet-ResNet50 model from checkpoint using segmentation_models_pytorch.
    
    Args:
        ckpt_path: Path to checkpoint file
        device: torch device
    
    Returns:
        Loaded model in eval mode
    """
    # Initialize model using segmentation_models_pytorch (matches training script)
    model = smp.Unet(
        encoder_name="resnet50",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None,
    )
    
    # Load checkpoint
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    # Validate checkpoint format
    assert isinstance(checkpoint, dict) and "model" in checkpoint, (
        f"Checkpoint must be a dict with 'model' key. Got keys: {checkpoint.keys() if isinstance(checkpoint, dict) else type(checkpoint)}"
    )
    
    # Extract state_dict
    state_dict = checkpoint["model"]
    
    # Debug: print state_dict info
    print(f"  State dict keys: {len(state_dict)} total")
    print(f"  First 5 keys: {list(state_dict.keys())[:5]}")
    
    # Load state_dict with strict matching
    model.load_state_dict(state_dict, strict=True)
    
    # Log checkpoint info if available
    if "epoch" in checkpoint:
        print(f"  Checkpoint epoch: {checkpoint['epoch']}")
    if "best_dice" in checkpoint:
        print(f"  Checkpoint best_dice: {checkpoint['best_dice']:.4f}")
    
    # Move to device and set eval mode
    model.to(device)
    model.eval()
    
    # Debug: print model type
    print(f"  Model type: {model.__class__.__name__}")
    print(f"Loaded model from: {ckpt_path}")
    
    return model


def infer_volume(
    model: torch.nn.Module,
    volume: np.ndarray,
    threshold: float,
    device: torch.device
) -> np.ndarray:
    """
    Run slice-wise inference on entire volume.
    
    Args:
        model: UNet-ResNet50 model
        volume: 3D numpy array (H, W, Z), ALREADY normalized at volume level
        threshold: sigmoid threshold
        device: torch device
    
    Returns:
        mask_3d: Binary mask (H, W, Z) in original resolution, dtype uint8
    """
    original_h, original_w, num_slices = volume.shape
    mask_3d = np.zeros((original_h, original_w, num_slices), dtype=np.uint8)
    
    print(f"Running inference on {num_slices} slices...")
    
    with torch.no_grad():
        for z in range(num_slices):
            # Get slice (already normalized at volume level)
            slice_2d = volume[:, :, z]
            
            # Preprocess (resize and convert to 3-channel)
            slice_preprocessed = preprocess_slice(slice_2d, target_size=256)
            
            # Convert to tensor and add batch dimension
            input_tensor = torch.from_numpy(slice_preprocessed).unsqueeze(0).to(device)
            
            # Forward pass
            output = model(input_tensor)
            
            # Apply sigmoid
            prob_map = torch.sigmoid(output).squeeze().cpu().numpy()
            
            # Threshold to binary
            binary_mask_256 = (prob_map > threshold).astype(np.uint8)
            
            # Resize back to original slice dimensions using nearest-neighbor
            # Nearest-neighbor preserves binary values without interpolation artifacts
            binary_mask_original = resize(
                binary_mask_256,
                (original_h, original_w),
                order=0,  # nearest-neighbor interpolation
                preserve_range=True,
                anti_aliasing=False
            ).astype(np.uint8)
            
            # Ensure binary values
            binary_mask_original = (binary_mask_original > 0.5).astype(np.uint8)
            
            mask_3d[:, :, z] = binary_mask_original
            
            if (z + 1) % 10 == 0 or z == num_slices - 1:
                print(f"  Processed slice {z + 1}/{num_slices}")
    
    # Validate output shape
    assert mask_3d.shape == volume.shape, (
        f"Mask shape {mask_3d.shape} != volume shape {volume.shape}"
    )
    
    return mask_3d


def filter_small_components(mask_3d: np.ndarray, min_voxels: int = MIN_VOXELS) -> np.ndarray:
    """
    Remove small connected components from the 3D binary mask.
    
    Why connected component filtering?
    - Small isolated predictions are often false positives (noise)
    - True lesions typically form spatially coherent 3D structures
    - Removing tiny components improves specificity without losing real lesions
    - The min_voxels threshold should be tuned based on expected lesion size
    
    Args:
        mask_3d: Binary mask (H, W, Z), dtype uint8, values {0, 1}
        min_voxels: Minimum number of voxels for a component to be retained
    
    Returns:
        Filtered binary mask (H, W, Z), dtype uint8
    """
    if np.sum(mask_3d) == 0:
        # No foreground voxels, return as-is
        print("  No lesion voxels detected, skipping component filtering")
        return mask_3d
    
    # Label connected components in 3D
    # Using 26-connectivity (full 3D neighborhood) to connect diagonal voxels
    # This is more permissive and groups nearby voxels into same component
    structure = ndimage.generate_binary_structure(3, 3)  # 26-connectivity
    labeled_mask, num_components = ndimage.label(mask_3d, structure=structure)
    
    print(f"  Found {num_components} connected components before filtering")
    
    if num_components == 0:
        return mask_3d
    
    # Count voxels in each component
    component_sizes = ndimage.sum(mask_3d, labeled_mask, range(1, num_components + 1))
    
    # Create filtered mask - keep only components with >= min_voxels
    filtered_mask = np.zeros_like(mask_3d, dtype=np.uint8)
    components_kept = 0
    
    for i, size in enumerate(component_sizes, start=1):
        if size >= min_voxels:
            filtered_mask[labeled_mask == i] = 1
            components_kept += 1
    
    voxels_before = np.sum(mask_3d)
    voxels_after = np.sum(filtered_mask)
    voxels_removed = voxels_before - voxels_after
    
    print(f"  Components kept: {components_kept}/{num_components} (min_voxels={min_voxels})")
    print(f"  Voxels: {voxels_before} -> {voxels_after} (removed {voxels_removed})")
    
    return filtered_mask


def compute_volume_metrics(
    mask_3d: np.ndarray,
    x_res: float,
    y_res: float,
    z_res: float,
    spatial_units: str
) -> dict:
    """
    Compute lesion volume using the lab formula.
    
    Lab formula:
        num_voxels = sum(mask(:))
        spatial_res = x_res * y_res * z_res
        lesion_volume = spatial_res * num_voxels
    
    Returns:
        Dictionary with all metrics
    """
    # Count lesion voxels
    num_voxels = int(np.sum(mask_3d))
    
    # Compute spatial resolution (voxel volume)
    spatial_res = x_res * y_res * z_res
    
    # Compute lesion volume
    lesion_volume_mm3 = spatial_res * num_voxels
    
    # Convert to um^3 (1 mm = 1000 um, so 1 mm^3 = 10^9 um^3)
    lesion_volume_um3 = lesion_volume_mm3 * 1e9
    
    metrics = {
        "num_voxels": num_voxels,
        "voxel_volume_mm3": float(spatial_res),
        "lesion_volume_mm3": float(lesion_volume_mm3),
        "lesion_volume_um3": float(lesion_volume_um3),
        "spatial_units": spatial_units,
        "x_res": float(x_res),
        "y_res": float(y_res),
        "z_res": float(z_res),
    }
    
    return metrics


def save_outputs(
    mask_3d: np.ndarray,
    metrics: dict,
    nii_path: str,
    ckpt_path: str,
    threshold: float,
    min_voxels: int,
    out_dir: str
) -> Path:
    """
    Save all outputs to disk.
    
    Creates:
        - pred_mask_3d.npy
        - pred_mask_slices_png/
        - volume_metrics.json
    
    Returns:
        Path to output directory
    """
    # Derive animal name from filename stem
    stem = Path(nii_path).stem
    
    # Create output directory
    output_path = Path(out_dir) / stem
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save 3D mask
    mask_npy_path = output_path / "pred_mask_3d.npy"
    np.save(mask_npy_path, mask_3d)
    print(f"Saved 3D mask: {mask_npy_path}")
    
    # Save slice PNGs
    slices_dir = output_path / "pred_mask_slices_png"
    slices_dir.mkdir(exist_ok=True)
    
    num_slices = mask_3d.shape[2]
    for z in range(num_slices):
        slice_mask = mask_3d[:, :, z]
        # Scale to 0-255 for visualization
        slice_img = (slice_mask * 255).astype(np.uint8)
        img = Image.fromarray(slice_img, mode='L')
        img.save(slices_dir / f"slice_{z:03d}_mask.png")
    print(f"Saved {num_slices} slice PNGs to: {slices_dir}")
    
    # Prepare full metrics dict
    full_metrics = {
        "shape": list(mask_3d.shape),
        "spacing": {
            "x_res": metrics["x_res"],
            "y_res": metrics["y_res"],
            "z_res": metrics["z_res"],
        },
        "spatial_units": metrics["spatial_units"],
        "num_voxels": metrics["num_voxels"],
        "voxel_volume_mm3": metrics["voxel_volume_mm3"],
        "lesion_volume_mm3": metrics["lesion_volume_mm3"],
        "lesion_volume_um3": metrics["lesion_volume_um3"],
        "threshold": threshold,
        "min_voxels_filter": min_voxels,
        "ckpt_path": str(ckpt_path),
        "nii_path": str(nii_path),
    }
    
    # Save metrics JSON
    metrics_path = output_path / "volume_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(full_metrics, f, indent=2)
    print(f"Saved metrics: {metrics_path}")
    
    return output_path


def main():
    args = parse_args()
    
    # Set device (MPS > CUDA > CPU)
    device = torch.device(
        "mps" if torch.backends.mps.is_available() 
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")
    
    # Step 1: Load NIfTI (raw intensities)
    volume_raw, x_res, y_res, z_res, spatial_units, header = load_nifti(args.nii_path)
    
    # Step 2: Apply volume-level normalization (ONCE for entire volume)
    # This replaces the previous per-slice normalization
    print("Applying volume-level percentile normalization...")
    volume_normalized = normalize_volume(volume_raw)
    
    # Step 3: Load model
    model = load_model_from_ckpt(args.ckpt_path, device)
    
    # Step 4: Run inference on normalized volume
    mask_3d_raw = infer_volume(model, volume_normalized, args.thr, device)
    
    # Step 5: Post-process - filter small connected components
    print("Applying 3D connected component filtering...")
    mask_3d_filtered = filter_small_components(mask_3d_raw, min_voxels=args.min_voxels)
    
    # Step 6: Compute volume metrics using filtered mask
    metrics = compute_volume_metrics(mask_3d_filtered, x_res, y_res, z_res, spatial_units)
    
    # Step 7: Save outputs
    output_path = save_outputs(
        mask_3d_filtered, metrics, args.nii_path, args.ckpt_path, 
        args.thr, args.min_voxels, args.out_dir
    )
    
    # Step 8: Print final result
    animal_name = Path(args.nii_path).stem
    lesion_volume_um3 = metrics["lesion_volume_um3"]
    
    print("\n" + "=" * 60)
    print(f"{animal_name}: Lesion Volume is {lesion_volume_um3} in um^3")
    print("=" * 60)
    
    # Additional summary
    print(f"\nSummary:")
    print(f"  Total lesion voxels: {metrics['num_voxels']}")
    print(f"  Voxel volume: {metrics['voxel_volume_mm3']:.6f} mm^3")
    print(f"  Lesion volume: {metrics['lesion_volume_mm3']:.4f} mm^3")
    print(f"  Lesion volume: {lesion_volume_um3:.2f} um^3")
    print(f"  Outputs saved to: {output_path}")


if __name__ == "__main__":
    main()