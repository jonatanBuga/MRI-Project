from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
METADATA_DIR = PROJECT_ROOT / "metadata"
MASK_RAW_DIR = PROJECT_ROOT / "data" / "masks_raw"
MASKS_PROCESSED_DIR = PROJECT_ROOT / "data" / "masks_processed" 

METADATA_DIR.mkdir(exist_ok=True)

TARGET_SIZE = 256