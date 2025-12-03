#build metadata_images.csv  
import os 
import pandas as pd
from pathlib import Path
from src.confing import DATA_RAW_DIR, METADATA_DIR

def make_metadata():
    rows = []
    for timepoint_dir in sorted(DATA_RAW_DIR.iterdir()):
        if not timepoint_dir.is_dir():
            continue
        timepoint = timepoint_dir.name  # "24h", "48h", ...

        for group_dir in sorted(timepoint_dir.iterdir()):
            if not group_dir.is_dir():
                continue
            group = group_dir.name  # "combined", "control", ...

            for subj_dir in sorted(group_dir.iterdir()):
                if not subj_dir.is_dir():
                    continue
                subject_id = subj_dir.name  # "combined_1"

                for img_path in sorted(subj_dir.iterdir()):
                    if img_path.suffix.lower() not in [".png", ".jpg", ".jpeg", ".tif"]:
                        continue

                    fname = img_path.name

                    # slice_idx: img_3.png -> 3
                    digits = "".join(c for c in fname if c.isdigit())
                    slice_idx = int(digits) if digits else -1

                    rows.append({
                        "image_path": str(img_path.relative_to(DATA_RAW_DIR)),
                        "timepoint": timepoint,
                        "group": group,
                        "subject_id": subject_id,
                        "slice_idx": slice_idx,
                        "has_label": False,
                    })
    df = pd.DataFrame(rows)
    output_csv = METADATA_DIR / "metadata_images.csv"
    df.to_csv(output_csv, index=False)
    print(f"Metadata saved to {output_csv}")

if __name__ == "__main__":
    make_metadata()
