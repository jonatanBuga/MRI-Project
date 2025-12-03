from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
METADATA_DIR = PROJECT_ROOT / "metadata"
METADATA_DIR.mkdir(exist_ok=True)

TARGET_SIZE = (256, 256)