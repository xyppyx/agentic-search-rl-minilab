"""Run a no-model tool smoke and write trajectory JSONL plus Markdown report."""

from __future__ import annotations

import argparse
from pathlib import Path

from search_r1_minilab.tools import LocalBM25Backend, MockSearchBackend, SearchItem, ToolRegistry
from search_r1_minilab.tools.smoke import build_tool_smoke_records
from search_r1_minilab.trajectories import build_markdown_report, write_trajectory_jsonl


DEFAULT_QUERIES = [
    "little prince",
    "novella french writer",
    "unknown query",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run mock/local BM25 tool smoke")
    parser.add_argument(
        "--backend",
        choices=["mock_search", "local_bm25"],
        default="mock_search",
        help="Backend to call",
    )
    parser.add_argument(
        "--bm25-corpus",
        default="my-search-r1/tests/fixtures/bm25_corpus.jsonl",
        help="JSONL corpus for local_bm25",
    )
    parser.add_argument(
        "--query",
        action="append",
        dest="queries",
        help="Query to run; repeat for multiple queries",
    )
    parser.add_argument(
        "--jsonl-output",
        default="my-search-r1/outputs/tool_smoke/trajectories.jsonl",
        help="Trajectory JSONL output path",
    )
    parser.add_argument(
        "--report-output",
        default="my-search-r1/outputs/tool_smoke/report.md",
        help="Markdown report output path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    registry = ToolRegistry()
    if args.backend == "mock_search":
        registry.register(_default_mock_backend())
    else:
        registry.register(LocalBM25Backend.from_jsonl(args.bm25_corpus))
    queries = args.queries or DEFAULT_QUERIES
    records = build_tool_smoke_records(registry, args.backend, queries)
    write_trajectory_jsonl(records, args.jsonl_output)
    report = build_markdown_report(
        records,
        title=f"Tool Smoke Report: {args.backend}",
    )
    report_path = Path(args.report_output)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print(f"wrote {len(records)} trajectories to {args.jsonl_output}")
    print(f"wrote report to {args.report_output}")


def _default_mock_backend() -> MockSearchBackend:
    return MockSearchBackend.from_pairs(
        {
            "little prince": [
                SearchItem(
                    id="mock-little-prince",
                    title="The Little Prince",
                    content="The Little Prince is by Antoine de Saint-Exupery.",
                    url="https://example.test/little-prince",
                    source="mock",
                )
            ],
            "novella french writer": [
                SearchItem(
                    id="mock-saint-exupery",
                    title="Antoine de Saint-Exupery",
                    content="Antoine de Saint-Exupery was a French writer.",
                    url="https://example.test/saint-exupery",
                    source="mock",
                )
            ],
        }
    )


if __name__ == "__main__":
    main()
