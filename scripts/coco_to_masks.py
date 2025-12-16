from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)


def _repo_root() -> Path:
    # scripts/ is at <repo_root>/scripts/
    return Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _infer_hw_from_image(image_path: Path) -> Optional[Tuple[int, int]]:
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        return None
    return (h, w)


def _iter_polygons(seg: Any) -> Iterable[List[float]]:
    """
    COCO segmentation can be:
      - list of polygons: [[x1,y1,...], [x1,y1,...], ...]
      - single flat polygon: [x1,y1,...]
    This yields polygons as flat coordinate lists.
    """
    if seg is None:
        return []

    if not isinstance(seg, list):
        return []

    # Case (a): list of polygons
    if len(seg) > 0 and all(isinstance(p, list) for p in seg):
        return seg  # type: ignore[return-value]

    # Case (b): a single flat polygon list
    if len(seg) > 0 and all(isinstance(v, (int, float)) for v in seg):
        return [seg]  # type: ignore[list-item]

    return []


def _poly_to_pts(poly: List[float]) -> Optional[np.ndarray]:
    """
    Convert flat [x1,y1,x2,y2,...] to OpenCV pts with shape (N,1,2) int32.
    Returns None for invalid polygons.
    """
    if len(poly) < 6 or (len(poly) % 2) != 0:
        return None
    arr = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
    pts = np.rint(arr).astype(np.int32).reshape(-1, 1, 2)
    return pts


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert COCO polygon segmentations to binary mask PNGs per split.")
    parser.add_argument("--split", required=True, choices=["train", "valid", "test"], help="Roboflow split to process")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing masks. If not set and mask exists, it will be skipped.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    root = _repo_root()
    split = args.split

    coco_path = root / "data" / "roboflow_coco" / split / "_annotations.coco.json"
    images_dir = root / "data" / "roboflow_coco" / split
    out_dir = root / "data" / "masks_raw" / split
    out_dir.mkdir(parents=True, exist_ok=True)

    if not coco_path.exists():
        LOGGER.error("COCO annotation file not found: %s", coco_path.as_posix())
        return 2
    if not images_dir.exists():
        LOGGER.error("Images directory not found: %s", images_dir.as_posix())
        return 2

    coco = _load_json(coco_path)
    images = coco.get("images", [])
    annotations = coco.get("annotations", [])

    if not isinstance(images, list) or not isinstance(annotations, list):
        LOGGER.error("Invalid COCO JSON format: expected 'images' and 'annotations' to be lists.")
        return 2

    anns_by_image: DefaultDict[int, List[Dict[str, Any]]] = defaultdict(list)
    for ann in annotations:
        if not isinstance(ann, dict):
            continue
        image_id = ann.get("image_id")
        if isinstance(image_id, int):
            anns_by_image[image_id].append(ann)

    total_images = 0
    masks_written = 0
    masks_skipped = 0
    unreadable_images = 0

    for img_info in images:
        if not isinstance(img_info, dict):
            continue

        total_images += 1

        image_id = img_info.get("id")
        file_name = img_info.get("file_name")
        width = img_info.get("width")
        height = img_info.get("height")

        if not isinstance(image_id, int) or not isinstance(file_name, str) or not file_name:
            LOGGER.warning("Skipping invalid image entry (missing id/file_name): %s", img_info)
            continue

        image_path = (images_dir / file_name).resolve()
        mask_name = f"{Path(file_name).stem}_mask.png"
        mask_path = (out_dir / mask_name).resolve()

        if mask_path.exists() and not args.overwrite:
            masks_skipped += 1
            continue

        # Determine H, W
        h: Optional[int] = int(height) if isinstance(height, (int, float)) else None
        w: Optional[int] = int(width) if isinstance(width, (int, float)) else None

        if h is None or w is None or h <= 0 or w <= 0:
            inferred = _infer_hw_from_image(image_path)
            if inferred is None:
                unreadable_images += 1
                LOGGER.warning("Unreadable image (cannot infer size): %s", image_path.as_posix())
                continue
            h, w = inferred

        mask = np.zeros((h, w), dtype=np.uint8)

        for ann in anns_by_image.get(image_id, []):
            seg = ann.get("segmentation")
            for poly in _iter_polygons(seg):
                pts = _poly_to_pts(poly)
                if pts is None:
                    continue
                cv2.fillPoly(mask, [pts], 1)

        # Save as 0/255 PNG
        out = (mask * 255).astype(np.uint8)
        ok = cv2.imwrite(str(mask_path), out)
        if not ok:
            LOGGER.warning("Failed to write mask: %s", mask_path.as_posix())
            continue

        masks_written += 1

    LOGGER.info("Split: %s", split)
    LOGGER.info("Total images: %d", total_images)
    LOGGER.info("Masks written: %d", masks_written)
    LOGGER.info("Masks skipped (exists, overwrite not set): %d", masks_skipped)
    LOGGER.info("Unreadable images: %d", unreadable_images)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())