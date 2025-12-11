import json
from pathlib import Path
from typing import Dict, List
import numpy as np
import cv2

# Try to reuse project config if available; otherwise infer project root
try:
    from src.utils.config import PROJECT_ROOT, MASKS_RAW_DIR  # type: ignore
except Exception:
    try:
        from src.utils.confing import PROJECT_ROOT, MASKS_RAW_DIR  # type: ignore
    except Exception:
        PROJECT_ROOT = Path(__file__).resolve().parents[1]  # repository root (one level above scripts/)
        MASKS_RAW_DIR = PROJECT_ROOT / "data" / "masks_raw"


def main():
    project_root = Path(PROJECT_ROOT)
    coco_path = project_root / "data" / "roboflow_coco" / "train" / "_annotations.coco.json"
    images_dir = project_root / "data" / "roboflow_coco" / "train"
    masks_root = Path(MASKS_RAW_DIR) / "train"
    masks_root.mkdir(parents=True, exist_ok=True)

    if not coco_path.exists():
        print(f"COCO annotations not found: {coco_path}")
        return

    with coco_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    images = data.get("images", [])
    annotations = data.get("annotations", [])

    # Build mapping from image_id -> list of annotations
    ann_by_image: Dict[int, List[dict]] = {}
    for ann in annotations:
        img_id = int(ann.get("image_id"))
        ann_by_image.setdefault(img_id, []).append(ann)

    written = 0
    total = len(images)

    for i, img in enumerate(images):
        img_id = int(img.get("id"))
        file_name = img.get("file_name")
        width = int(img.get("width", 0))
        height = int(img.get("height", 0))

        if width <= 0 or height <= 0:
            # Try to load the image to get size as fallback
            img_path = images_dir / file_name
            if img_path.exists():
                im = cv2.imread(str(img_path))
                if im is None:
                    print(f"Warning: could not read image {img_path}; skipping")
                    continue
                height, width = im.shape[:2]
            else:
                print(f"Warning: missing size and image file {file_name}; skipping")
                continue

        # initialize empty mask (0 background)
        mask = np.zeros((height, width), dtype=np.uint8)

        anns = ann_by_image.get(img_id, [])
        for ann in anns:
            seg = ann.get("segmentation")
            if not seg:
                # no polygon segmentation for this annotation; skip
                continue
            # we expect segmentation to be a list of polygons; take the first polygon
            poly = seg[0]
            if not poly:
                continue
            # poly is a flat list [x1,y1,x2,y2,...]
            coords = np.array(poly, dtype=np.float32).reshape(-1, 2)
            # convert to integer pixel coordinates
            pts = coords.astype(np.int32)
            # cv2.fillPoly expects a list of pts arrays; reshape to (N,1,2) if desired
            pts_cv = pts.reshape((-1, 1, 2))
            # fill polygon with value 1
            cv2.fillPoly(mask, [pts_cv], 1)

        # build output filename
        base_name = Path(file_name).stem
        mask_name = f"{base_name}_mask.png"
        mask_path = masks_root / mask_name

        # write mask as 0/255 PNG
        success = cv2.imwrite(str(mask_path), (mask * 255).astype(np.uint8))
        if success:
            written += 1
        else:
            print(f"Failed to write mask: {mask_path}")

        # progress logging
        if (i + 1) % 50 == 0 or (i + 1) == total:
            print(f"Processed {i+1}/{total} images - masks written: {written}")

    print(f"Done. Total images processed: {total}, masks written: {written}")


if __name__ == "__main__":
    main()