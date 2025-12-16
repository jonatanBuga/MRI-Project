from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


FIELDNAMES = ["image_path", "mask_path", "split", "has_label", "mask_valid"]


def _repo_root() -> Path:
    # scripts/ is at <repo_root>/scripts/
    return Path(__file__).resolve().parents[1]


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required input CSV: {path.as_posix()}")

    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path.as_posix()}")
        return [dict(r) for r in reader]


def _write_csv(path: Path, rows: Iterable[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})


def _train_add_split(train_rows: List[Dict[str, str]]) -> List[Dict[str, object]]:
    # Input train-only file has no 'split' column.
    out: List[Dict[str, object]] = []
    for r in train_rows:
        out.append(
            {
                "image_path": (r.get("image_path") or "").strip(),
                "mask_path": (r.get("mask_path") or "").strip(),
                "split": "train",
                "has_label": (r.get("has_label") or "").strip(),
                "mask_valid": (r.get("mask_valid") or "").strip(),
            }
        )
    return out


def _normalize_rows(rows: List[Dict[str, str]]) -> List[Dict[str, object]]:
    # Ensure columns exist (missing -> ""), strip strings
    out: List[Dict[str, object]] = []
    for r in rows:
        out.append(
            {
                "image_path": (r.get("image_path") or "").strip(),
                "mask_path": (r.get("mask_path") or "").strip(),
                "split": (r.get("split") or "").strip(),
                "has_label": (r.get("has_label") or "").strip(),
                "mask_valid": (r.get("mask_valid") or "").strip(),
            }
        )
    return out


def _dedupe_by_image_path(rows: List[Dict[str, object]]) -> Tuple[List[Dict[str, object]], List[str]]:
    seen = set()
    dupes: List[str] = []
    out: List[Dict[str, object]] = []
    for r in rows:
        image_path = str(r.get("image_path") or "")
        if not image_path:
            # skip empty image path rows entirely
            continue
        if image_path in seen:
            dupes.append(image_path)
            continue
        seen.add(image_path)
        out.append(r)
    return out, dupes


def _split_counts(rows: List[Dict[str, object]]) -> Counter:
    c: Counter = Counter()
    for r in rows:
        split = str(r.get("split") or "")
        c[split] += 1
    return c


def main() -> int:
    root = _repo_root()

    in_train = root / "metadata" / "metadata_labeled_roboflow.csv"
    in_valid = root / "metadata" / "metadata_labeled_roboflow_valid.csv"
    in_test = root / "metadata" / "metadata_labeled_roboflow_test.csv"

    out_train = root / "metadata" / "metadata_labeled_roboflow_train.csv"
    out_all = root / "metadata" / "metadata_labeled_roboflow_all.csv"
    out_splits = root / "metadata" / "splits_roboflow.csv"

    # 1) Build train_with_split (do NOT modify existing train-only file)
    train_rows_raw = _read_csv_rows(in_train)
    train_rows = _train_add_split(train_rows_raw)
    _write_csv(out_train, train_rows, FIELDNAMES)

    # 2) Read valid/test (already have split col)
    valid_rows = _normalize_rows(_read_csv_rows(in_valid))
    test_rows = _normalize_rows(_read_csv_rows(in_test))

    # 3) Merge + dedupe by image_path (keep first occurrence)
    merged = list(train_rows) + valid_rows + test_rows
    merged_deduped, dupes = _dedupe_by_image_path(merged)

    if dupes:
        print("DUPLICATE image_path rows detected (keeping first occurrence):")
        for p in sorted(set(dupes)):
            print(" -", p)

    _write_csv(out_all, merged_deduped, FIELDNAMES)

    # 4) Write splits file
    splits_rows = [{"image_path": r["image_path"], "split": r["split"]} for r in merged_deduped]
    _write_csv(out_splits, splits_rows, ["image_path", "split"])

    # 5) Summary + expected totals validation
    counts = _split_counts(merged_deduped)
    total = len(merged_deduped)

    expected = {"train": 577, "val": 169, "test": 79}
    expected_total = 825

    # Some files may use "valid" instead of "val"; we keep what's in CSV.
    # The requirement expects val=169, so warn if key mismatch too.
    if counts.get("train", 0) != expected["train"] or counts.get("val", 0) != expected["val"] or counts.get("test", 0) != expected["test"] or total != expected_total:
        print("WARNING: Split counts differ from expected totals.")
        print("  Expected:", expected, "total=", expected_total)
        print("  Actual:  ", dict(counts), "total=", total)

    print("\nWrote:")
    print(" -", out_train.as_posix())
    print(" -", out_all.as_posix())
    print(" -", out_splits.as_posix())

    print("\nSummary:")
    print("  total rows:", total)
    print("  counts per split:", dict(counts))

    print("\nFirst 5 rows of merged all:")
    for r in merged_deduped[:5]:
        print({k: r.get(k, "") for k in FIELDNAMES})

    return 0


if __name__ == "__main__":
    raise SystemExit(main())