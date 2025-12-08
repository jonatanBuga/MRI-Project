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

