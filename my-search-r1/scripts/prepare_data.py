"""Download and prepare Search-R1 MiniLab train/dev/test JSONL data."""

from __future__ import annotations

import argparse
from pathlib import Path

from search_r1_minilab.prepare_data import (
    DATASET_ID,
    DATASET_REVISION,
    format_counts,
    prepare_dataset,
)


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    """Parse dataset preparation arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "datasets" / "raw")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "datasets")
    parser.add_argument("--dev-per-source", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--dataset-revision", default=DATASET_REVISION)
    return parser.parse_args()


def main() -> None:
    """Prepare all Search-R1 dataset splits."""
    args = parse_args()
    counts_by_split = prepare_dataset(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        dev_per_source=args.dev_per_source,
        seed=args.seed,
        dataset_id=args.dataset_id,
        revision=args.dataset_revision,
    )
    for split in ["train", "test", "dev"]:
        print(format_counts(split, counts_by_split[split]))


if __name__ == "__main__":
    main()
