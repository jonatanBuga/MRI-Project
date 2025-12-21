# MRI Segmentation — Mouse Brain Longitudinal Study

## Project Overview
This repository analyzes mouse brain MRI scans collected across multiple timepoints and treatment groups (Control, Combined, GOT, GPT). The goal is to train a segmentation model to detect edema/lesion regions and compute lesion volumes over time for longitudinal analysis.

## Repository Structure
- data/raw, data/interim, data/processed  
  - Raw images, intermediate artifacts, and final preprocessed data / model inputs.
- metadata/  
  - CSVs and tables describing images, subjects, timepoints and labels.
- notebooks/  
  - Exploratory analysis and visualization notebooks (EDA, diagnostics).
- src/data/  
  - Data loading and preprocessing transforms (image pipeline, utils).
- src/models/  
  - Model definitions and training/evaluation scripts.
- src/utils/  
  - Utility functions and configuration (e.g., TARGET_SIZE in src/utils/config.py).
- scripts/  
  - Convenience scripts for running experiments, conversions, or ingestion.

## EDA Summary (high level)
- Image sizes
  - No fixed image size; many different Width×Height pairs exist.
  - All images share the same aspect ratio but at different resolutions.
  - Conclusion: resize all images to a uniform TARGET_SIZE before training.
- Group-level intensity and noise
  - GOT and GPT groups are brighter on average than Control and Combined.
  - GOT shows slightly lower noise (better quality); GPT is a bit noisier.
  - Contrast is very high and consistent across groups.
  - Cohen’s d indicates meaningful differences between Control/Combined vs GOT/GPT.
  - Conclusion: apply intensity normalization to avoid the model learning group identity.
- Per-subject outlier
  - One clear outlier identified: subject "Control 1" (very low mean intensity).
  - Possible reasons: darker acquisition, scanner issue, or different slice position.
  - Conclusion: flag this subject as a potential outlier during training/evaluation.

## Preprocessing Pipeline (src/data/transforms.py)
Implemented pipeline stages (all operate on NumPy arrays until final tensor conversion):
- `load_image(path)`  
  - Loads an image from disk using PIL, converts to RGB (so grayscale → 3 channels), and returns a NumPy array with shape (H, W, C) and dtype float32.
- `resize_image(img, target_size)`  
  - Resizes image to square (TARGET_SIZE × TARGET_SIZE) using OpenCV. Uses cv2.INTER_AREA for downscaling and cv2.INTER_LINEAR for upscaling; preserves dtype float32.
- `normalize_image(img)`  
  - Performs min–max normalization to the range [0, 1]. This is a lightweight normalization step; MRI-specific Z-score standardization will be added later.
- `preprocess_image(path)`  
  - Full pipeline: load → resize → (TODO: orientation correction) → normalize → convert to PyTorch tensor and reorder to (C, H, W). Returns a torch.float32 tensor ready for model input.

Note: The default TARGET_SIZE is configured in `src/utils/config.py`.

## How to Run the Preprocessing Sanity Check
From the project root:
```bash
python -m src.data.transforms
```
What the script does:
- Searches `data/raw/` for PNG/JPG images.
- Randomly selects a few images.
- Runs `preprocess_image` on each sample.
- Prints the tensor shape, dtype, and min/max values for quick verification.

## Dataset & Data Pipeline (Stages 2–3)

This stage establishes a robust, reproducible data pipeline for segmentation training and inference.  
No model assumptions are made at this stage.

### Labeled Metadata

All labeled data is consolidated into a single CSV:

- `metadata/metadata_labeled_roboflow_all.csv`

Each row represents one image and contains:
- `image_path` – path to the image
- `mask_path` – path to the binary segmentation mask
- `split` – one of `{train, val, test}` (fixed, Roboflow-based)
- `has_label`, `mask_valid` – label validity flags

Final dataset sizes:
- Train: 577
- Validation: 169
- Test: 79
- Total: 825

The split is fixed and deterministic, ensuring full reproducibility across runs.

### Segmentation Dataset

Implemented in:
- `src/data/mri_seg_dataset.py`

`MRISegmentationDataset`:
- Loads image–mask pairs from the metadata CSV
- Applies the same preprocessing pipeline used throughout the project
- Converts masks to strict binary format `{0,1}`
- Supports filtering by split (`train`, `val`, `test`)
- Returns a standardized sample:
  ```python
  {
    "image": Tensor[C, H, W],
    "mask":  Tensor[H, W],
    "meta":  dict
  }
  ```

Sanity checks performed:
- Image and mask spatial alignment
- Mask binarization
- Tensor dtype and shape consistency

### DataLoaders

Implemented in:
- `src/data/dataloaders.py`

Key design decisions:
- Custom `collate_fn` to safely batch tensors while keeping metadata intact

Final batch format:
- `image`: `[B, 3, H, W]`, `float32`
- `mask`: `[B, 1, H, W]`, `float32`
- `meta`: `list` of `dict` (one per sample)

Train loader:
- `shuffle=True`
- `drop_last=True`

Validation/Test loaders:
- `shuffle=False`
- `drop_last=False`

### Performance Benchmark

A dedicated benchmark script was used to measure DataLoader throughput:
- `scripts/benchmark_dataloader.py`

Results showed a clear CPU bottleneck with a single worker. Recommended defaults (based on empirical measurement):
- `batch_size = 8`
- `num_workers = 4`
- `pin_memory = True`
- `persistent_workers = True`

These settings are now the project defaults.

### Inference-Only Dataset

Implemented in:
- `src/data/mri_inference_dataset.py`

`MRIInferenceDataset` is a lightweight dataset designed for model inference only:
- Loads images without requiring labels or masks
- Uses the exact same preprocessing pipeline as training
- Can be initialized from:
  - an explicit list of image paths, or
  - a root directory (recursively scanned)

Returns:
```python
{
  "image": Tensor[C, H, W],
  "meta": {
    "image_path": str,
    "index": int
  }
}
```

Key properties:
- Stable, sorted ordering of images (reproducible inference)
- No dependency on metadata CSVs
- Suitable for test-time inference, external datasets, and qualitative evaluation

## Model Evaluation & Benchmark Protocol

### Task Definition
Binary segmentation of edema / lesion regions in mouse brain MRI slices.

All models are evaluated under identical data, preprocessing, and evaluation conditions to ensure a fair and reproducible comparison.

### Input & Output

**Input:**
- MRI slices as 2D images
- Tensor shape: `[B, 3, H, W]`, `float32`
- (Uniform resizing and intensity normalization applied to all models)

**Output:**
- Binary segmentation mask
- Tensor shape: `[B, 1, H, W]`, values in `[0, 1]` after sigmoid

### Training Setup
- Segmentation setting: slice-wise 2D  
  Each MRI slice is treated as an independent sample (no 3D or inter-slice context).

- Training loss:  
  Dice Loss + Binary Cross Entropy (fixed across all models)

### Evaluation Protocol
- Dataset split: fixed and deterministic (train / validation / test)
- Evaluation split: validation set only (test set kept untouched)
- Thresholding:  
  Predicted probabilities are binarized using a fixed threshold of `0.5`
- Metric computation:  
  Metrics are computed at the epoch level, aggregating predictions over the full validation set.

### Evaluation Metrics

**Primary metric:**
- Dice score (overlap between predicted and ground-truth lesion regions)

**Secondary metrics:**
- Intersection over Union (IoU)
- Precision
- Recall

(Volume-based error metrics are planned for later stages of the study.)

### Model Selection Criterion
The best model checkpoint is selected based on the highest validation Dice score.

All models are compared using the same metrics, threshold, and evaluation protocol.

### Reproducibility
- Fixed data split
- Identical preprocessing and normalization
- Identical loss functions and evaluation metrics across models
