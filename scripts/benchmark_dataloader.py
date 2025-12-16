from __future__ import annotations

import sys
import time
from pathlib import Path

# --- Parameters (edit as needed) ---
BATCH_SIZE = 8
NUM_WORKERS_TO_TEST = [0, 2, 4]
NUM_BATCHES_TO_TIME = 100
WARMUP_BATCHES = 10
# -----------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.dataloaders import build_dataloaders  # noqa: E402


def _consume_n_batches(loader, n: int) -> int:
    """Iterate over at most n batches. Returns number of batches actually consumed."""
    it = iter(loader)
    count = 0
    for _ in range(n):
        try:
            _ = next(it)
        except StopIteration:
            break
        count += 1
    return count


def main() -> int:
    print("=== DataLoader Throughput Benchmark (train_loader) ===")
    print(f"batch_size={BATCH_SIZE} warmup_batches={WARMUP_BATCHES} timed_batches={NUM_BATCHES_TO_TIME}")

    for nw in NUM_WORKERS_TO_TEST:
        pin_memory = True
        persistent_workers = nw > 0

        train_loader, _, _ = build_dataloaders(
            batch_size=BATCH_SIZE,
            num_workers=nw,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )

        # Warmup (not timed)
        warmup_done = _consume_n_batches(train_loader, WARMUP_BATCHES)

        # Timed
        t0 = time.perf_counter()
        timed_done = _consume_n_batches(train_loader, NUM_BATCHES_TO_TIME)
        t1 = time.perf_counter()

        elapsed = max(t1 - t0, 1e-12)
        batches_per_sec = timed_done / elapsed
        samples_per_sec = (timed_done * BATCH_SIZE) / elapsed

        print(f"\nnum_workers={nw} pin_memory={pin_memory} persistent_workers={persistent_workers}")
        print(f"  warmup batches: {warmup_done}")
        print(f"  timed batches:  {timed_done}")
        print(f"  elapsed (s):    {elapsed:.4f}")
        print(f"  batches/sec:   {batches_per_sec:.2f}")
        print(f"  samples/sec:   {samples_per_sec:.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())