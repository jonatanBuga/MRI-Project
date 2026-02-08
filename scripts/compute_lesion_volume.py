import numpy as np
import nibabel as nib
from pathlib import Path
from skimage.transform import resize


def load_nifti_metadata(nii_path: str) -> tuple:
    """Load NIfTI and extract resolution metadata."""
    img = nib.load(nii_path)
    header = img.header
    
    # Extract voxel spacing (x_res, y_res, z_res)
    zooms = header.get_zooms()
    x_res, y_res, z_res = zooms[0], zooms[1], zooms[2]
    
    # Confirm spatial units
    spatial_units, _ = header.get_xyzt_units()
    print(f"Spatial units: {spatial_units}")
    
    # Get original volume shape
    original_shape = img.shape  # (192, 144, 28)
    
    return img, x_res, y_res, z_res, original_shape


def get_predicted_mask(original_shape: tuple) -> np.ndarray:
    """
    Placeholder function to load/receive the predicted 3D mask.
    
    Replace this with your actual mask loading logic.
    The mask should be binary (0 or 1) with shape matching original_shape.
    
    If your segmentation model outputs 256x256 slices, load them here
    and they will be resized in resize_mask_to_original().
    """
    # Example: Load from file or receive from segmentation pipeline
    # mask = np.load("path/to/predicted_mask.npy")
    
    # Placeholder: return zeros (replace with actual mask)
    # For testing, you can create a dummy mask with some lesion voxels
    mask = np.zeros(original_shape, dtype=np.uint8)
    
    return mask


def resize_mask_to_original(mask_256: np.ndarray, original_shape: tuple) -> np.ndarray:
    """
    Resize mask from 256x256 per slice to original NIfTI slice dimensions.
    Uses nearest-neighbor interpolation to preserve binary values.
    
    Args:
        mask_256: 3D mask with shape (256, 256, num_slices)
        original_shape: Target shape (192, 144, 28)
    
    Returns:
        Resized 3D binary mask matching original_shape
    """
    target_x, target_y, num_slices = original_shape
    resized_mask = np.zeros(original_shape, dtype=np.uint8)
    
    for z in range(num_slices):
        slice_mask = mask_256[:, :, z]
        # Resize using nearest-neighbor (order=0) to preserve binary values
        resized_slice = resize(
            slice_mask,
            (target_x, target_y),
            order=0,  # nearest-neighbor interpolation
            preserve_range=True,
            anti_aliasing=False
        )
        resized_mask[:, :, z] = (resized_slice > 0.5).astype(np.uint8)
    
    return resized_mask


def compute_lesion_volume(
    mask: np.ndarray,
    x_res: float,
    y_res: float,
    z_res: float
) -> tuple:
    """
    Compute lesion volume using the lab formula.
    
    num_voxels = sum(mask(:))
    spatial_res = x_res * y_res * z_res
    lesion_volume = spatial_res * num_voxels
    """
    # Count lesion voxels
    num_voxels = np.sum(mask)
    
    # Compute spatial resolution (voxel volume in mm^3)
    spatial_res = x_res * y_res * z_res
    
    # Compute lesion volume in mm^3
    lesion_volume_mm3 = spatial_res * num_voxels
    
    # Convert to um^3 (1 mm = 1000 um, so 1 mm^3 = 10^9 um^3)
    lesion_volume_um3 = lesion_volume_mm3 * 1e9
    
    return num_voxels, spatial_res, lesion_volume_mm3, lesion_volume_um3


def main():
    # NIfTI file path (relative to project root)
    nii_path = "20250702_1051542T2TurboRAREnex4Rare10s50001a001.nii"
    
    # Extract animal name from filename stem
    animal_name = Path(nii_path).stem
    
    # Step 1: Load NIfTI and extract metadata
    img, x_res, y_res, z_res, original_shape = load_nifti_metadata(nii_path)
    
    print(f"Original shape: {original_shape}")
    print(f"Voxel spacing: x={x_res}, y={y_res}, z={z_res} mm")
    
    # Step 2: Get predicted mask
    # Option A: If mask is already at original resolution
    mask = get_predicted_mask(original_shape)
    
    # Option B: If mask comes from 256x256 model output, resize it
    # mask_256 = load_256x256_mask()  # Your loading function
    # mask = resize_mask_to_original(mask_256, original_shape)
    
    # Ensure mask is binary
    mask = (mask > 0.5).astype(np.uint8)
    
    # Step 3: Compute lesion volume
    num_voxels, spatial_res, lesion_volume_mm3, lesion_volume_um3 = compute_lesion_volume(
        mask, x_res, y_res, z_res
    )
    
    # Debug output
    print(f"Number of lesion voxels: {num_voxels}")
    print(f"Spatial resolution (voxel volume): {spatial_res} mm^3")
    print(f"Lesion volume: {lesion_volume_mm3} mm^3")
    
    # Step 4: Print final output in required format
    print(f"{animal_name}: Lesion Volume is {lesion_volume_um3} in um^3")


if __name__ == "__main__":
    main()