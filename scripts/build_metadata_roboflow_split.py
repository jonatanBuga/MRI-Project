from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _repo_root() -> Path:
    # scripts/ is at <repo_root>/scripts/
    return Path(__file__).resolve().parents[1]


def _load_coco(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _split_for_csv(split: str) -> str:
    # In output CSV, map valid -> val
    return "val" if split == "valid" else split


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Roboflow labeled metadata CSV for a single split.")
    parser.add_argument("--split", required=True, choices=["train", "valid", "test"], help="Roboflow split to process")
    args = parser.parse_args()

    root = _repo_root()
    split = args.split

    coco_path = root / "data" / "roboflow_coco" / split / "_annotations.coco.json"
    images_dir = root / "data" / "roboflow_coco" / split
    masks_dir = root / "data" / "masks_raw" / split
    out_csv = root / "metadata" / f"metadata_labeled_roboflow_{split}.csv"

    if not coco_path.exists():
        raise FileNotFoundError(f"COCO annotation file not found: {coco_path.as_posix()}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir.as_posix()}")
    if not masks_dir.exists():
        # Masks dir might be created later; still allow metadata generation
        masks_dir.mkdir(parents=True, exist_ok=True)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if out_csv.exists():
        print(f"WARNING: Output CSV already exists and will be overwritten: {out_csv.as_posix()}")

    coco = _load_coco(coco_path)
    images = coco.get("images", [])
    if not isinstance(images, list):
        raise ValueError(f"Invalid COCO JSON: 'images' must be a list. File: {coco_path.as_posix()}")

    fieldnames = ["image_path", "mask_path", "split", "has_label", "mask_valid"]

    rows: List[Dict[str, Any]] = []
    masks_found = 0
    masks_missing = 0

    for img in images:
        if not isinstance(img, dict):
            continue

        file_name = img.get("file_name")
        if not isinstance(file_name, str) or not file_name.strip():
            continue
        file_name = file_name.strip()

        image_rel = Path("data") / "roboflow_coco" / split / file_name
        mask_rel = Path("data") / "masks_raw" / split / f"{Path(file_name).stem}_mask.png"

        has_label = (root / mask_rel).exists()
        mask_valid = has_label

        if has_label:
            masks_found += 1
        else:
            masks_missing += 1

        rows.append(
            {
                "image_path": image_rel.as_posix(),
                "mask_path": mask_rel.as_posix(),
                "split": _split_for_csv(split),
                "has_label": bool(has_label),
                "mask_valid": bool(mask_valid),
            }
        )

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("Wrote:", out_csv.as_posix())
    print("Summary:")
    print("  total rows:   ", len(rows))
    print("  masks found:  ", masks_found)
    print("  masks missing:", masks_missing)

    print("\nFirst 5 rows:")
    for r in rows[:5]:
        print(
            {
                "image_path": r["image_path"],
                "mask_path": r["mask_path"],
                "split": r["split"],
                "has_label": r["has_label"],
                "mask_valid": r["mask_valid"],
            }
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())