"""Tests for trajectory JSONL and Markdown reports."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from search_r1_minilab.tools import MockSearchBackend, SearchItem, ToolRegistry
from search_r1_minilab.tools.smoke import build_tool_smoke_records
from search_r1_minilab.trajectories import (
    build_markdown_report,
    classify_trajectory,
    load_trajectory_jsonl,
    summarize_trajectories,
    write_trajectory_jsonl,
)


class TrajectoryJsonlTest(unittest.TestCase):
    def test_write_and_load_normalized_jsonl(self) -> None:
        records = [
            {
                "question": "Who wrote The Little Prince?",
                "answers": "Antoine de Saint-Exupery",
                "turns": [{"role": "assistant", "text": "Answer: Antoine de Saint-Exupery"}],
                "reward": 1.0,
                "exact_match": True,
                "valid_format": True,
            }
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "trajectories.jsonl"

            count = write_trajectory_jsonl(records, path)
            loaded = load_trajectory_jsonl(path)

        self.assertEqual(count, 1)
        self.assertEqual(loaded[0]["answers"], ["Antoine de Saint-Exupery"])
        self.assertEqual(loaded[0]["search_calls"], 0)
        self.assertEqual(loaded[0]["tool_failures"], 0)


class TrajectoryReportTest(unittest.TestCase):
    def test_classification_and_summary_counts(self) -> None:
        records = _sample_records()

        labels = [classify_trajectory(record) for record in records]
        summary = summarize_trajectories(records)

        self.assertIn("correct", labels[0])
        self.assertIn("wrong", labels[1])
        self.assertIn("invalid_format", labels[2])
        self.assertIn("tool_failure", labels[3])
        self.assertIn("repeated_search", labels[4])
        self.assertEqual(summary.total, 5)
        self.assertEqual(summary.correct, 1)
        self.assertEqual(summary.wrong, 3)
        self.assertEqual(summary.invalid_format, 1)
        self.assertEqual(summary.tool_failure, 1)
        self.assertEqual(summary.repeated_search, 1)

    def test_markdown_report_contains_expected_sections(self) -> None:
        report = build_markdown_report(_sample_records(), title="Smoke Trajectories")

        self.assertIn("# Smoke Trajectories", report)
        self.assertIn("## Summary", report)
        self.assertIn("## Correct Cases", report)
        self.assertIn("## Wrong Cases", report)
        self.assertIn("## Invalid Format Cases", report)
        self.assertIn("## Tool Failure Cases", report)
        self.assertIn("## Repeated Search Cases", report)
        self.assertIn("## Group Comparisons", report)

    def test_tool_smoke_records_round_trip_into_report(self) -> None:
        registry = ToolRegistry()
        registry.register(
            MockSearchBackend.from_pairs(
                {"little prince": [SearchItem(title="The Little Prince", content="fixture")]}
            )
        )
        records = build_tool_smoke_records(registry, "mock_search", ["little prince"])

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "smoke.jsonl"
            write_trajectory_jsonl(records, path)
            loaded = load_trajectory_jsonl(path)

        report = build_markdown_report(loaded, title="Tool Smoke")

        self.assertEqual(loaded[0]["data_source"], "tool_smoke")
        self.assertIn("# Tool Smoke", report)
        self.assertIn("| Trajectories | 1 |", report)


def _sample_records() -> list[dict]:
    return [
        {
            "question": "Q1",
            "answers": ["A1"],
            "turns": [{"role": "assistant", "text": "Answer: A1"}],
            "reward": 1.0,
            "advantage": 0.5,
            "valid_format": True,
            "exact_match": True,
            "search_calls": 0,
        },
        {
            "question": "Q1",
            "answers": ["A1"],
            "turns": [{"role": "assistant", "text": "Answer: wrong"}],
            "reward": 0.0,
            "advantage": -0.5,
            "valid_format": True,
            "exact_match": False,
            "search_calls": 0,
        },
        {
            "question": "Q2",
            "answers": ["A2"],
            "turns": [{"role": "assistant", "text": "no answer marker"}],
            "reward": -0.1,
            "valid_format": False,
            "exact_match": False,
            "search_calls": 0,
        },
        {
            "question": "Q3",
            "answers": ["A3"],
            "turns": [
                {"role": "assistant", "text": "", "tool_call": {"name": "search", "query": "q3"}},
                {"role": "tool", "tool_name": "search", "ok": False, "error_type": "timeout"},
            ],
            "reward": 0.0,
            "valid_format": True,
            "exact_match": False,
            "search_calls": 1,
            "tool_failures": 1,
        },
        {
            "question": "Q4",
            "answers": ["A4"],
            "turns": [
                {"role": "assistant", "text": "", "tool_call": {"name": "search", "query": "same"}},
                {"role": "tool", "tool_name": "search", "ok": True},
                {"role": "assistant", "text": "", "tool_call": {"name": "search", "query": "same"}},
                {"role": "tool", "tool_name": "search", "ok": True},
            ],
            "reward": 0.0,
            "valid_format": True,
            "exact_match": False,
            "search_calls": 2,
        },
    ]


if __name__ == "__main__":
    unittest.main()
