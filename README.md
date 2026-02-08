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

## Model Integration & Inference 

This Stage transitions from a finalized data layer (Stages 1–3) to **model integration, training, and evaluation**.

### 1) Purpose 

The objective is to verify that multiple segmentation architectures can be executed end-to-end within the same pipeline:

- **No training or fine-tuning** is performed.
- A strict **sanity check** is conducted for:
  - input/output tensor compatibility,
  - execution on available devices (CPU / GPU / MPS),
  - correct probability generation (sigmoid),
  - correct artifact export (PNG outputs).
- Ensures that all candidate models conform to the project’s **unified data + inference interface**, so subsequent benchmarking reflects model quality rather than integration differences.

### 2) Unified Model Interface

All integrated models follow the same interface (slice-wise 2D segmentation; no 3D context):

- **Input**: MRI slices as 2D tensors  
  `image`: `[B, 3, H, W]`, `float32`, normalized to `[0, 1]`
- **Output**: binary segmentation predictions  
  `logits`: `[B, 1, H, W]` (pre-sigmoid)  
  `probs = sigmoid(logits)`: `[B, 1, H, W]`, `float32`, in `[0, 1]`

This standardization enables consistent downstream post-processing, visualization, and evaluation across architectures.

### 3) Models Integrated 

Three model families were integrated and validated via inference-only runs:

- **UNet + ResNet-50 encoder (CNN baseline)**  
  - Loaded via `segmentation_models_pytorch.Unet`  
  - Encoder: `resnet50`, ImageNet pretrained weights  
  - Inference-only (no task-specific adaptation)

- **TransUNet (R50-ViT-B_16 hybrid CNN–Transformer)**  
  - Loaded from the official TransUNet codebase vendored locally under `third_party/transunet/`  
  - Wrapped in `src/models/transunet_wrapper.py` to enforce the unified output contract  
  - Inference-only (no task-specific adaptation)

- **SegFormer (Transformer-based; Hugging Face Transformers)**  
  - Loaded via `transformers.SegformerForSemanticSegmentation`  
  - Inference-only (no task-specific adaptation)

### 4) Inference Protocol (Shared)

All models use the same inference protocol to ensure comparable outputs:

- **Dataset split**: fixed and deterministic (`train/val/test`) from:  
  `metadata/metadata_labeled_roboflow_all.csv`
- **Inference split**: validation set (`val`)
- **Post-processing**:
  - apply `sigmoid` to logits to obtain probabilities
  - generate binary masks using a configurable threshold (default `0.5`)
- **Saved artifacts** (for qualitative inspection):
  - probability heatmaps: PNG encoded in `0–255`
  - thresholded binary masks: PNG encoded in `{0, 255}`

Outputs are stored per model under:
- `outputs/inference/<model_name>/{timestamp}/`

### 5) Observations (Qualitative)

The goal of Integration and inference is correctness of integration rather than segmentation accuracy. Qualitative behaviors observed during inference-only runs are expected:

- All architectures produce **stable probability maps** prior to training.
- Outputs are often **centered around ~0.5** before fine-tuning, reflecting uncalibrated decision boundaries.
- Threshold sensitivity and probability distribution can differ across model families:
  - CNN baseline vs. hybrid CNN–Transformer vs. transformer-only behavior may vary.
- For **SegFormer**, when configured for binary output (`num_labels=1`), a **new binary segmentation head may be initialized** (depending on how the pretrained checkpoint is adapted). This is expected prior to task-specific training and does not indicate failure.

These observations confirm correct execution and output formatting, not segmentation quality.

### 6) Outcome 

completes the model-integration gate with the following outcomes:

- [x] All three models run inference end-to-end without errors  
- [x] Input/output shapes are consistent across architectures (`[B, 3, H, W]` → `[B, 1, H, W]`)  
- [x] Probability and mask artifacts are generated for qualitative inspection  
- [x] The pipeline is ready for sanity training and quantitative evaluation in subsequent days

## Training Progress
### Sanity Training — UNet + ResNet50 (MPS)

As part of the initial validation phase, we conducted a short **sanity training run** using a **UNet with a ResNet50 encoder**.  
The goal was to verify the full training pipeline before committing to long, full-scale training.

---

### Objectives
- Validate dataset integrity and dataloaders
- Ensure loss, metrics, and backpropagation behave correctly
- Observe **learning trends**, not just isolated metric values
- Inspect qualitative segmentation results via fixed-sample visualizations

---

### Configuration
- **Model:** UNet + ResNet50 encoder
- **Device:** Apple MPS
- **Epochs:** 8
- **Batch size:** 8
- **Train batches:** 10 (sanity subset)
- **Validation batches:** 30
- **Loss:** Dice + Binary Cross-Entropy
- **Metrics:** Dice, IoU, Precision, Recall
- **Visualizations:** Fixed validation samples at `epoch_0` and `epoch_last`

---

### Quantitative Results

- **Training loss** decreases almost monotonically across epochs.
- **Validation loss** shows a clear downward trend with minor fluctuations (expected in sanity runs).
- **Dice score** improves rapidly after epochs 3–4 and stabilizes around **0.6–0.7**.
- **IoU** follows the same trend, reaching ~0.55.
- **Recall** is consistently high, indicating strong sensitivity to target regions.
- **Precision** is more variable, suggesting some over-segmentation early on.

These behaviors are expected given the limited number of batches and confirm meaningful learning rather than noise.

---

### Interpretation
- The model demonstrates **real structural learning** of the segmentation task.
- High recall indicates a preference for detecting relevant regions rather than missing them — a desirable property in medical segmentation.
- Precision is expected to improve with longer training, threshold tuning, or regularization.
- Temporary metric drops (e.g., around epoch 5–6) are attributed to statistical noise from small validation subsets.

---

### Qualitative Evaluation
Fixed-sample visualizations were generated for:
- **epoch_0** (before training)
- **epoch_last** (after sanity training)

A clear qualitative improvement is observed:
- Initial predictions are largely uninformative.
- Final predictions show coherent, anatomically meaningful masks aligned with ground truth.

Visual outputs are saved under: outputs/sanity_viz/unet_r50/
---

### Conclusions
- The full training pipeline (data → model → loss → metrics → visualization) is **stable and reliable**.
- UNet + ResNet50 serves as a **strong baseline** for this project.
- No technical blockers (memory, gradients, device compatibility) were identified.
- The project is ready to proceed to **longer, full-scale training** with confidence.


### Evaluation on Healthy Controls
  To evaluate robustness and false-positive behavior, the baseline model was applied to MRI scans of healthy mice.
  Across all samples, the model exhibited consistently low activation, with mean predicted probabilities in the range of ~0.005-0.008 and fewer than 0.5% of pixels exceeding the segmentation threshold. While isolated pixels occasionally reached high confidence values, no spatially consistent or anatomically plausible lesion patterns were observed.
  Overall, this indicates that the baseline model does not hallucinate edema like regions on healthy scans and demonstrates stable behavior on out of distribution data.

---
## Lesion / Edema Quantification from NIfTI (MRI)

This section describes the **end-to-end inference and volume quantification pipeline** used to compute lesion / edema volumes from 3D MRI scans stored in **NIfTI (`.nii`) format**, using a trained **UNet with ResNet50 encoder**.

The pipeline bridges the gap between **slice-wise segmentation** and **biologically meaningful 3D volume measurements**.

---

### Overview

Given a 3D MRI volume (`.nii`) with multiple axial slices of a single mouse at a specific timepoint, the pipeline:

1. Loads the NIfTI volume and extracts voxel spacing from metadata  
2. Applies volume-level intensity normalization  
3. Runs slice-wise segmentation using a trained UNet-ResNet50 model  
4. Reconstructs a full 3D binary lesion mask  
5. Applies 3D connected-component filtering to remove small false positives  
6. Computes lesion volume in **mm³** and **µm³**  

This allows direct comparison of lesion volumes across:
- Different patiens
- Multiple timepoints 
- Different experimental or treatment groups

---

### Input

- **NIfTI file (`.nii`)**
  - Represents a single animal at a single timepoint
  - Shape: `(H, W, Z)` where `Z` is the number of slices
  - Must include valid voxel spacing in the header (`pixdim`)
- **Trained model checkpoint**
  - UNet with ResNet50 encoder
  - Saved in the following format:
    ```python
    {
      "model": state_dict,
      "epoch": int,
      "best_dice": float
    }
    ```

---

### Preprocessing

1. **Volume-level percentile normalization**
   - Applied across the entire 3D volume (e.g. p1–p99)
   - Improves slice-to-slice consistency and inference stability

2. **Slice-wise preprocessing**
   - Each slice is resized to `256 × 256`
   - Grayscale slices are expanded to 3 channels
   - Intensity values are normalized to `[0, 1]`

---

### Model Inference

- Inference is performed **slice by slice**
- Sigmoid activation is applied to logits
- A fixed threshold (default `0.5`) converts probabilities to a binary mask
- Masks are resized back to the original spatial resolution

Supported devices:
- **Apple Silicon (MPS)** when available
- CPU fallback otherwise

---

### 3D Post-processing

After reconstructing the full `(H, W, Z)` mask:

- **3D connected component analysis** is applied
- Small isolated components are removed
- Only components with at least `min_voxels` (default: `30`) are kept

This step significantly reduces noise-driven false positives.

---

### Volume Quantification

Lesion volume is computed using the voxel-based formula:

num_voxels = sum(mask)
voxel_volume = x_res × y_res × z_res (mm³)
lesion_volume = num_voxels × voxel_volume

Reported metrics:
- Total lesion voxels
- Voxel volume (mm³)
- Lesion volume (mm³)
- Lesion volume (µm³)

---
### Output Structure

For each input NIfTI file, the pipeline creates:
outputs/quantification/<nii_stem>/
├── pred_mask_3d.npy
├── pred_mask_slices_png/
│ ├── slice_000_mask.png
│ ├── slice_001_mask.png
│ └── ...
└── volume_metrics.json


#### Output files
- **`pred_mask_3d.npy`**  
  Binary 3D mask `(H, W, Z)` in original resolution
- **`pred_mask_slices_png/`**  
  One PNG mask per slice for visual inspection
- **`volume_metrics.json`**  
  Contains spacing, voxel counts, volumes, and run metadata

---

### How to Run (Single NIfTI)

Activate environment:
```bash
conda activate YOUR-ENV-NAME
``` 
Run quantification:
python scripts/run_unet_r50_nifti_quantification.py \
  --nii_path PATH/TO/VOLUME.nii \
  --ckpt_path outputs/training/baseline_full_v2/checkpoints/best_by_val_dice.pt \
  --thr 0.5 \
  --out_dir outputs/quantification
