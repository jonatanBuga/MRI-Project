from pathlib import Path
from typing import Optional, Union
import pandas as pd


# attempt to import config constants from the expected module; accept minor naming variations
try:
    import sys
    from pathlib import Path
    sys.path.append(str(Path().resolve().parent / "src"))
    from utils.config import DATA_RAW_DIR, METADATA_DIR
except Exception:
    try:
        from utils.confing import DATA_RAW_DIR, METADATA_DIR  # fallback if name was misspelled
    except Exception:
        # As a last resort, infer locations relative to the project
        project_root = Path(__file__).resolve().parents[2]
        DATA_RAW_DIR = project_root / "data" / "raw"
        METADATA_DIR = project_root / "metadata"


def _normalize_image_path(image_path: Union[str, Path]) -> str:
    """
    Normalize image_path for storage in the metadata CSV.

    - If the provided path is absolute and resides under DATA_RAW_DIR, store it relative to DATA_RAW_DIR.
    - Otherwise, return the original path as a posix string.
    """
    p = Path(image_path)
    try:
        if p.is_absolute():
            try:
                rel = p.relative_to(Path(DATA_RAW_DIR))
                return rel.as_posix()
            except Exception:
                # not under DATA_RAW_DIR; return posix string relative to project if possible
                project_root = Path(__file__).resolve().parents[2]
                try:
                    rel = p.relative_to(project_root)
                    return rel.as_posix()
                except Exception:
                    return p.as_posix()
        else:
            return p.as_posix()
    except Exception:
        return str(image_path)


def _normalize_mask_path(mask_path: Union[str, Path]) -> str:
    """
    Normalize mask_path for storage.

    - If absolute and under project root or DATA_RAW_DIR, store relative where possible.
    - Otherwise return posix string.
    """
    p = Path(mask_path)
    project_root = Path(__file__).resolve().parents[2]
    try:
        if p.is_absolute():
            try:
                return p.relative_to(DATA_RAW_DIR).as_posix()
            except Exception:
                try:
                    return p.relative_to(project_root).as_posix()
                except Exception:
                    return p.as_posix()
        else:
            return p.as_posix()
    except Exception:
        return str(mask_path)


def update_metadata_for_image(
    image_path: Union[str, Path],
    mask_path: Optional[Union[str, Path]],
    metadata_filename: str = "metadata_images.csv",
) -> None:
    """
    Update or append a row in the metadata CSV for a given image, attaching a mask path.

    Behavior:
    - Loads metadata CSV at METADATA_DIR / metadata_filename if it exists, otherwise creates
      a new DataFrame with columns:
        ["image_path", "timepoint", "group", "subject_id", "slice_idx", "has_label"]
    - Ensures columns "mask_path" and "mask_valid" exist (defaults: "" and False).
    - Normalizes image_path (make relative to DATA_RAW_DIR when applicable).
    - Normalizes mask_path (store a consistent relative/posix path when possible).
    - If a row matching image_path exists, update mask_path and has_label.
    - If no row exists, append a new row with sensible defaults and the provided mask info.
    - Writes the updated DataFrame back to METADATA_DIR / metadata_filename (index=False).
    """
    metadata_file = Path(METADATA_DIR) / metadata_filename
    metadata_file.parent.mkdir(parents=True, exist_ok=True)

    # load or create DataFrame
    if metadata_file.exists():
        df = pd.read_csv(metadata_file)
    else:
        cols = ["image_path", "timepoint", "group", "subject_id", "slice_idx", "has_label"]
        df = pd.DataFrame(columns=cols)

    # ensure expected columns exist
    if "image_path" not in df.columns:
        df["image_path"] = ""
    if "timepoint" not in df.columns:
        df["timepoint"] = ""
    if "group" not in df.columns:
        df["group"] = ""
    if "subject_id" not in df.columns:
        df["subject_id"] = ""
    if "slice_idx" not in df.columns:
        df["slice_idx"] = -1
    if "has_label" not in df.columns:
        df["has_label"] = False

    # ensure mask columns
    if "mask_path" not in df.columns:
        df["mask_path"] = ""
    if "mask_valid" not in df.columns:
        df["mask_valid"] = False

    # normalize paths
    img_norm = _normalize_image_path(image_path)
    mask_norm = ""
    if mask_path is not None:
        mask_norm = _normalize_mask_path(mask_path)

    # find matching row
    matches = df.index[df["image_path"].astype(str) == str(img_norm)].tolist()

    if matches:
        # update all matching rows (usually should be unique)
        for idx in matches:
            df.at[idx, "mask_path"] = mask_norm
            if mask_path is not None and mask_norm != "":
                df.at[idx, "has_label"] = True
                df.at[idx, "mask_valid"] = True
            else:
                # clear label info if mask removed
                df.at[idx, "has_label"] = bool(df.at[idx, "has_label"])  # keep existing unless mask is None
                df.at[idx, "mask_valid"] = False if mask_path is None else df.at[idx, "mask_valid"]
    else:
        # append new row with defaults
        new_row = {
            "image_path": img_norm,
            "timepoint": "",
            "group": "",
            "subject_id": "",
            "slice_idx": -1,
            "has_label": bool(mask_path is not None),
            "mask_path": mask_norm,
            "mask_valid": bool(mask_path is not None),
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    # write back
    df.to_csv(metadata_file, index=False)


if __name__ == "__main__":
    # simple debug/demo helper
    example_image = "1 Week/Combined/Combined 1/Slice 19.png"
    example_mask = Path("data/masks_raw/Combined_1_Slice19_mask.png")
    print(f"Updating metadata for image='{example_image}' mask='{example_mask}'\n")
    update_metadata_for_image(example_image, example_mask)
    metadata_file = Path(METADATA_DIR) / "metadata_images.csv"
    if metadata_file.exists():
        print("Updated metadata (last 5 rows):")
        print(pd.read_csv(metadata_file).tail(5).to_string(index=False))
    else:
        print("Metadata file not found after update (unexpected).")
