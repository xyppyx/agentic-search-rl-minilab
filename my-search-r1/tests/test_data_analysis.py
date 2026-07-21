"""Tests for dataset preparation and checkpoint analysis helpers."""

from __future__ import annotations

import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from search_r1_minilab.analysis import (
    CheckpointSpec,
    load_metric_series,
    load_metrics_from_jsonl,
    parse_checkpoint_spec,
    save_figure,
)
from search_r1_minilab.prepare_data import (
    clean_answers,
    extract_answers,
    format_counts,
    normalize_row,
    select_dev,
    write_jsonl,
)


class PrepareDataTest(unittest.TestCase):
    def test_clean_answers_preserves_order_and_deduplicates(self) -> None:
        self.assertEqual(clean_answers([" A ", "B", "A", "", " B "]), ["A", "B"])
        self.assertEqual(clean_answers(" single "), ["single"])
        self.assertEqual(clean_answers(None), [])

    def test_extract_answers_falls_back_to_reward_model(self) -> None:
        row = {"reward_model": {"ground_truth": {"target": ["Paris", " Paris "]}}}

        self.assertEqual(extract_answers(row), ["Paris"])

    def test_normalize_row_drops_missing_question_or_answers(self) -> None:
        self.assertIsNone(normalize_row({"question": "", "golden_answers": ["A"]}))
        self.assertIsNone(normalize_row({"question": "Q?", "golden_answers": []}))

        record = normalize_row(
            {
                "id": 123,
                "question": " Q? ",
                "golden_answers": ["A"],
                "data_source": "nq",
            }
        )

        self.assertEqual(record["id"], "123")
        self.assertEqual(record["question"], "Q?")
        self.assertEqual(record["answers"], ["A"])
        self.assertEqual(record["data_source"], "nq")

    def test_select_dev_is_balanced_and_deterministic(self) -> None:
        records = [
            {"id": f"a-{index}", "data_source": "a"}
            for index in range(4)
        ] + [{"id": f"b-{index}", "data_source": "b"} for index in range(4)]

        first = select_dev(records, per_source=2, seed=7)
        second = select_dev(records, per_source=2, seed=7)

        self.assertEqual(first, second)
        self.assertEqual(Counter(record["data_source"] for record in first), {"a": 2, "b": 2})

    def test_write_jsonl_and_format_counts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "data" / "dev.jsonl"
            write_jsonl([{"id": "x", "question": "Q?", "answers": ["A"]}], path)

            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(rows[0]["id"], "x")
        self.assertEqual(format_counts("dev", Counter({"nq": 2})), "dev: total=2; nq=2")


class AnalysisTest(unittest.TestCase):
    def test_parse_checkpoint_spec(self) -> None:
        spec = parse_checkpoint_spec("Step 20=eval_results_rl_step_20.jsonl")

        self.assertEqual(spec.label, "Step 20")
        self.assertEqual(spec.filename, "eval_results_rl_step_20.jsonl")

    def test_load_metrics_from_summary_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "eval_results.jsonl"
            path.write_text(
                json.dumps({"type": "trajectory", "question": "Q"}) + "\n"
                + json.dumps(
                    {
                        "type": "summary",
                        "metrics": {"em/macro": 0.25, "format/rate": 0.75},
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            metrics = load_metrics_from_jsonl(path)

        self.assertEqual(metrics, {"em/macro": 0.25, "format/rate": 0.75})

    def test_load_metrics_from_trajectory_jsonl(self) -> None:
        records = [
            {"data_source": "nq", "exact_match": True, "valid_format": True},
            {"data_source": "nq", "exact_match": False, "valid_format": True},
            {"data_source": "hotpotqa", "exact_match": False, "valid_format": False},
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "trajectories.jsonl"
            path.write_text(
                "\n".join(json.dumps(record) for record in records) + "\n",
                encoding="utf-8",
            )

            metrics = load_metrics_from_jsonl(path)

        self.assertEqual(metrics["em/macro"], 0.25)
        self.assertAlmostEqual(metrics["format/rate"], 2 / 3)

    def test_load_metric_series_and_save_figure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result_dir = Path(temp_dir)
            (result_dir / "base.jsonl").write_text(
                json.dumps(
                    {
                        "type": "summary",
                        "metrics": {"em/macro": 0.5, "format/rate": 1.0},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            labels, macro_em, format_rate = load_metric_series(
                result_dir,
                (CheckpointSpec("Base", "base.jsonl"),),
            )
            output = result_dir / "figure.png"

            save_figure(labels, macro_em, format_rate, output, dpi=80)

            self.assertEqual(labels, ["Base"])
            self.assertEqual(macro_em, [0.5])
            self.assertEqual(format_rate, [1.0])
            self.assertTrue(output.is_file())


if __name__ == "__main__":
    unittest.main()
