"""Plot MiniLab checkpoint EM and format-rate comparisons from eval JSONL files."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from search_r1_minilab.analysis import (
    CheckpointSpec,
    default_or_existing_specs,
    load_metric_series,
    make_figure,
    parse_checkpoint_spec,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT_DIR = ROOT / "eval_results"
DEFAULT_OUTPUT = DEFAULT_RESULT_DIR / "checkpoint_em_format.png"


def parse_args() -> argparse.Namespace:
    """Parse checkpoint analysis arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--show", action="store_true")
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        metavar="LABEL=FILENAME",
        help=(
            "Checkpoint result to plot. May be repeated. Defaults to the original "
            "Search-R1 checkpoint filenames, or Current=trajectories.jsonl when present."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Load checkpoint metrics and write a comparison figure."""
    args = parse_args()
    checkpoints = _checkpoint_specs(args)
    labels, macro_em, format_rate = load_metric_series(args.result_dir, checkpoints)
    figure = make_figure(labels, macro_em, format_rate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved figure: {args.output}")
    for label, em_value, format_value in zip(labels, macro_em, format_rate, strict=True):
        print(f"{label}: em/macro={em_value:.4f}, format/rate={format_value:.4f}")
    if args.show:
        plt.show()
    else:
        plt.close(figure)


def _checkpoint_specs(args: argparse.Namespace) -> tuple[CheckpointSpec, ...]:
    if args.checkpoint:
        return tuple(parse_checkpoint_spec(value) for value in args.checkpoint)
    return default_or_existing_specs(args.result_dir)


if __name__ == "__main__":
    main()
