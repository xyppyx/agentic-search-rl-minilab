"""Generate a Markdown report from trajectory JSONL."""

from __future__ import annotations

import argparse
from pathlib import Path

from search_r1_minilab.trajectories import build_markdown_report, load_trajectory_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a trajectory Markdown report")
    parser.add_argument("--input", required=True, help="Input trajectory JSONL path")
    parser.add_argument("--output", required=True, help="Output Markdown path")
    parser.add_argument("--title", default="Trajectory Report", help="Report title")
    parser.add_argument(
        "--max-examples",
        type=int,
        default=3,
        help="Maximum examples per report section",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_trajectory_jsonl(args.input)
    report = build_markdown_report(
        records,
        title=args.title,
        max_examples_per_section=args.max_examples,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(f"wrote {len(records)} trajectories to {output_path}")


if __name__ == "__main__":
    main()
