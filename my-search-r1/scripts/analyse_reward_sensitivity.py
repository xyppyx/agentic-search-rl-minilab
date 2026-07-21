"""Run offline reward sensitivity analysis over persisted eval JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from search_r1_minilab.reward_sensitivity import (
    analyze_jsonl,
    build_markdown_report,
    build_summary_payload,
    load_configs,
    write_results_jsonl,
    write_summary_json,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "eval_results" / "trajectories.jsonl"
DEFAULT_SUMMARY_OUTPUT = ROOT / "eval_results" / "reward_sensitivity_summary.json"
DEFAULT_JSONL_OUTPUT = ROOT / "eval_results" / "reward_sensitivity.jsonl"
DEFAULT_REPORT_OUTPUT = ROOT / "eval_results" / "reward_sensitivity.md"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_OUTPUT)
    parser.add_argument("--jsonl-output", type=Path, default=DEFAULT_JSONL_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    parser.add_argument("--title", default="Reward Sensitivity Report")
    parser.add_argument("--max-cases-per-config", type=int, default=5)
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        help=(
            "Custom config, e.g. "
            "'name:duplicate=0.03,empty=0.01,max_search=0,verbose=0,verbose_threshold=0'"
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Run sensitivity analysis and write JSON/Markdown artifacts."""
    args = parse_args()
    configs = load_configs(args.config)
    results = analyze_jsonl(args.input, configs)
    payload = build_summary_payload(configs, results)
    write_summary_json(payload, args.summary_output)
    write_results_jsonl(results, args.jsonl_output)
    report = build_markdown_report(
        configs,
        results,
        title=args.title,
        max_cases_per_config=args.max_cases_per_config,
    )
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(report, encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"wrote {len(results)} sensitivity rows to {args.jsonl_output}")
    print(f"wrote summary to {args.summary_output}")
    print(f"wrote report to {args.report_output}")


if __name__ == "__main__":
    main()
