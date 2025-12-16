from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Iterable, Optional


LOGGER = logging.getLogger(__name__)

# Extensions to check (in order). Add more if your Roboflow export uses them.
ROBOFLOW_EXTS = [".png", ".jpg", ".jpeg"]


def _project_root() -> Path:
    # scripts/ is expected to live directly under the repo root
    return Path(__file__).resolve().parents[1]


def _iter_mask_pngs(masks_dir: Path) -> Iterable[Path]:
    # Per requirements: iterate over all PNG files in masks directory
    return sorted(masks_dir.glob("*.png"))


def _base_name_from_mask(mask_path: Path) -> Optional[str]:
    """
    Given: ".../<base_name>_mask.png" -> returns "<base_name>"
    Returns None if filename doesn't follow the required pattern.
    """
    name = mask_path.name
    suffix = "_mask.png"
    if not name.endswith(suffix):
        return None
    return name[: -len(suffix)]


def _find_roboflow_image(roboflow_dir: Path, base_name: str) -> Optional[Path]:
    """
    Strict matching only: look for roboflow_dir/<base_name><ext>.
    No heuristics beyond checking common extensions.
    """
    for ext in ROBOFLOW_EXTS:
        candidate = roboflow_dir / f"{base_name}{ext}"
        if candidate.exists():
            return candidate
    return None


def _to_repo_relative_posix(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    root = _project_root()
    roboflow_dir = root / "data" / "roboflow_coco" / "train"
    masks_dir = root / "data" / "masks_raw" / "train"
    out_csv = root / "metadata" / "metadata_labeled_roboflow.csv"

    if not roboflow_dir.exists():
        LOGGER.error("Roboflow images directory not found: %s", roboflow_dir)
        return 2
    if not masks_dir.exists():
        LOGGER.error("Masks directory not found: %s", masks_dir)
        return 2

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    total_masks = 0
    linked_pairs = 0
    skipped = 0

    fieldnames = ["image_path", "mask_path", "has_label", "mask_valid"]

    # Clean rebuild each run (overwrite)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for mask_path in _iter_mask_pngs(masks_dir):
            total_masks += 1

            base_name = _base_name_from_mask(mask_path)
            if base_name is None:
                LOGGER.warning("Skipping PNG that does not match '*_mask.png': %s", mask_path.name)
                skipped += 1
                continue

            image_path = _find_roboflow_image(roboflow_dir, base_name)
            if image_path is None:
                LOGGER.warning(
                    "No matching Roboflow image found for mask '%s' (base_name='%s') in %s",
                    mask_path.name,
                    base_name,
                    roboflow_dir.as_posix(),
                )
                skipped += 1
                continue

            writer.writerow(
                {
                    "image_path": _to_repo_relative_posix(image_path, root),
                    "mask_path": _to_repo_relative_posix(mask_path, root),
                    "has_label": True,
                    "mask_valid": True,
                }
            )
            linked_pairs += 1

    LOGGER.info("Wrote labeled pairs CSV: %s", out_csv.as_posix())
    LOGGER.info("Total masks scanned: %d", total_masks)
    LOGGER.info("Successfully linked image–mask pairs: %d", linked_pairs)
    LOGGER.info("Skipped masks: %d", skipped)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())