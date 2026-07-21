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
    diagnose_record,
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
        self.assertIn("direct_correct", labels[0])
        self.assertIn("direct_wrong", labels[1])
        self.assertIn("searched_invalid_format", labels[5])
        self.assertIn("empty_search", labels[5])
        self.assertIn("max_search_no_answer", labels[5])
        self.assertIn("too_many_search_no_gain", labels[5])
        self.assertEqual(summary.total, 6)
        self.assertEqual(summary.correct, 1)
        self.assertEqual(summary.wrong, 3)
        self.assertEqual(summary.invalid_format, 2)
        self.assertEqual(summary.tool_failure, 1)
        self.assertEqual(summary.repeated_search, 2)
        self.assertEqual(summary.behavior.direct_correct, 1)
        self.assertEqual(summary.behavior.searched_invalid_format, 1)
        self.assertEqual(summary.behavior.empty_observation_count, 1)
        self.assertEqual(summary.behavior.pending_tool_call_count, 1)

    def test_markdown_report_contains_expected_sections(self) -> None:
        report = build_markdown_report(_sample_records(), title="Smoke Trajectories")

        self.assertIn("# Smoke Trajectories", report)
        self.assertIn("## Summary", report)
        self.assertIn("## Correct Cases", report)
        self.assertIn("## Wrong Cases", report)
        self.assertIn("## Invalid Format Cases", report)
        self.assertIn("## Tool Failure Cases", report)
        self.assertIn("## Repeated Search Cases", report)
        self.assertIn("## Direct Correct Cases", report)
        self.assertIn("## Searched Correct Cases", report)
        self.assertIn("## Searched Wrong Cases", report)
        self.assertIn("## Empty Search Cases", report)
        self.assertIn("## Max-Search No-Answer Cases", report)
        self.assertIn("| Direct correct | 1 |", report)
        self.assertIn("| Direct correct rate | 0.1667 |", report)
        self.assertIn("- Gold answers: `['A1']`", report)
        self.assertIn("- Extracted answer: `A1`", report)
        self.assertIn("## Group Comparisons", report)

    def test_diagnose_record_counts_empty_duplicate_and_pending_calls(self) -> None:
        diagnostics = diagnose_record(_sample_records()[5])

        self.assertEqual(diagnostics.empty_observation_count, 1)
        self.assertEqual(diagnostics.duplicate_query_count, 1)
        self.assertEqual(diagnostics.pending_tool_call_count, 1)
        self.assertTrue(diagnostics.max_search_no_answer)

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
        {
            "question": "Q5",
            "answers": ["A5"],
            "turns": [
                {
                    "role": "assistant",
                    "text": "",
                    "tool_call": {"name": "search", "query": "same"},
                },
                {
                    "role": "tool",
                    "tool_name": "search",
                    "ok": True,
                    "items": [],
                },
                {
                    "role": "assistant",
                    "text": "",
                    "tool_call": {"name": "search", "query": "same"},
                },
                {
                    "role": "tool",
                    "tool_name": "search",
                    "ok": True,
                    "items": [{"title": "weak"}],
                },
                {
                    "role": "assistant",
                    "text": "",
                    "tool_call": {"name": "search", "query": "pending"},
                },
            ],
            "reward": -0.1,
            "valid_format": False,
            "exact_match": False,
            "search_calls": 3,
            "metadata": {"stop_reason": "max_search_calls"},
        },
    ]


if __name__ == "__main__":
    unittest.main()
