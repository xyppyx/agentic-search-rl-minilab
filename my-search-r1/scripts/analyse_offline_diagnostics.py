"""Run offline diagnostics over persisted Search-R1 eval JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from search_r1_minilab.offline_diagnostics import (
    build_markdown_report,
    diagnose_records,
    load_records,
    summarize_diagnostics,
    write_diagnostic_jsonl,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "eval_results" / "trajectories.jsonl"
DEFAULT_JSONL_OUTPUT = ROOT / "eval_results" / "offline_diagnostics.jsonl"
DEFAULT_REPORT_OUTPUT = ROOT / "eval_results" / "offline_diagnostics.md"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--jsonl-output", type=Path, default=DEFAULT_JSONL_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    parser.add_argument("--title", default="Offline Diagnostic Report")
    parser.add_argument("--max-cases-per-section", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    """Load eval trajectories, write diagnostic labels, and print summary."""
    args = parse_args()
    diagnostics = diagnose_records(load_records(args.input))
    write_diagnostic_jsonl(diagnostics, args.jsonl_output)
    report = build_markdown_report(
        diagnostics,
        title=args.title,
        max_cases_per_section=args.max_cases_per_section,
    )
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(report, encoding="utf-8")
    print(json.dumps(summarize_diagnostics(diagnostics).to_dict(), indent=2, sort_keys=True))
    print(f"wrote {len(diagnostics)} diagnostics to {args.jsonl_output}")
    print(f"wrote report to {args.report_output}")


if __name__ == "__main__":
    main()
