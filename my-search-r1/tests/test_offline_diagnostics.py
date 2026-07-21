"""Tests for offline eval diagnostics."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from search_r1_minilab.offline_diagnostics import (
    build_markdown_report,
    diagnose_record,
    diagnose_records,
    load_records,
    summarize_diagnostics,
    write_diagnostic_jsonl,
)


def make_record(
    *,
    question: str,
    answers: list[str],
    answer: str,
    exact_match: bool = False,
    search_calls: int = 1,
    queries: list[str] | None = None,
    observation: str = "Alice Smith was the writer. Bob Jones recorded it.",
) -> dict:
    """Build a minimal persisted trajectory record."""
    turns = []
    for query in queries or ["test query"]:
        turns.append(
            {
                "role": "assistant",
                "parsed_kind": "tool",
                "tool_call": {"name": "search", "query": query},
                "text": f"<tool_call>{query}</tool_call>",
            }
        )
        turns.append(
            {
                "role": "tool",
                "ok": True,
                "items": [{"title": "Evidence", "content": observation}],
                "observation": observation,
            }
        )
    turns.append(
        {
            "role": "assistant",
            "parsed_kind": "answer",
            "text": f"Reasoning.\n\nAnswer: {answer}",
        }
    )
    return {
        "metadata": {"id": "case-1", "stop_reason": "answer"},
        "question": question,
        "answers": answers,
        "exact_match": exact_match,
        "valid_format": True,
        "search_calls": search_calls,
        "turns": turns,
    }


class OfflineDiagnosticsTest(unittest.TestCase):
    def test_possible_alias_match_flags_name_variant(self) -> None:
        record = make_record(
            question="Who is younger, Martin Luther King III or his brother Dexter?",
            answers=["Dexter"],
            answer="Dexter King",
            search_calls=2,
            queries=["Martin Luther King III birth date", "Dexter King birth date"],
        )

        diagnostic = diagnose_record(record)

        self.assertTrue(diagnostic.possible_alias_match)
        self.assertFalse(diagnostic.answer_granularity_miss)
        self.assertFalse(diagnostic.missing_followup_query)

    def test_answer_granularity_miss_flags_year_only_date_answer(self) -> None:
        record = make_record(
            question='When was the person who delivered the "Quit India" speech born?',
            answers=["October 2, 1869"],
            answer="1869",
            observation="Mahatma Gandhi delivered the Quit India speech.",
        )

        diagnostic = diagnose_record(record)

        self.assertTrue(diagnostic.answer_granularity_miss)
        self.assertIn("final_answer_is_less_specific_than_gold", diagnostic.reasons)

    def test_missing_followup_query_flags_single_search_multihop_risk(self) -> None:
        record = make_record(
            question="Who is the maternal grandfather of Claudia Antonia?",
            answers=["Sextus Aelius Catus"],
            answer="Gaius Silius",
            queries=["Claudia Antonia maternal grandfather"],
            observation="Claudia Antonia was the daughter of Claudius and Aelia Paetina.",
        )

        diagnostic = diagnose_record(record)

        self.assertTrue(diagnostic.missing_followup_query)
        self.assertIn("single_search_multihop_or_role_binding_risk", diagnostic.reasons)

    def test_multi_candidate_answer_flags_overwide_final_answer(self) -> None:
        record = make_record(
            question="Who was the director of My Last Day?",
            answers=["Barry Cook"],
            answer="Barry Cook (for the 2011 film) or Liu Bicheng (for the 2020 film)",
            queries=["My Last Day film director"],
        )

        diagnostic = diagnose_record(record)

        self.assertTrue(diagnostic.multi_candidate_answer)
        self.assertIn("final_answer_contains_multiple_candidates", diagnostic.reasons)

    def test_bad_max_search_loop_preserves_helpful_followup(self) -> None:
        repeated = make_record(
            question="Who is Thomas Lloyd-Mostyn's paternal grandfather?",
            answers=["Edward Lloyd, 1st Baron Mostyn"],
            answer="Edward Lloyd",
            search_calls=4,
            queries=[
                "Thomas Lloyd-Mostyn paternal grandfather",
                "Thomas Lloyd-Mostyn paternal grandfather",
                "Thomas Lloyd-Mostyn paternal grandfather",
            ],
        )
        helpful = make_record(
            question="Who is the maternal grandfather of Claudia Antonia?",
            answers=["Sextus Aelius Catus"],
            answer="Aelius Sejanus",
            search_calls=3,
            queries=[
                "Claudia Antonia maternal grandfather",
                "Aelia Paetina father",
                "Sextus Aelius Catus father",
            ],
        )

        repeated_diagnostic = diagnose_record(repeated)
        helpful_diagnostic = diagnose_record(helpful)

        self.assertTrue(repeated_diagnostic.bad_max_search_loop)
        self.assertFalse(repeated_diagnostic.helpful_followup_query)
        self.assertTrue(helpful_diagnostic.helpful_followup_query)
        self.assertFalse(helpful_diagnostic.bad_max_search_loop)

    def test_summary_jsonl_and_markdown_outputs(self) -> None:
        records = [
            make_record(
                question="Who was the director of Spring Waltz?",
                answers=["Yoon Seok-Ho"],
                answer="Yun Seok-ho",
                observation="Spring Waltz was directed by Yoon Seok-ho.",
            ),
            make_record(
                question='When was the person who delivered the "Quit India" speech born?',
                answers=["October 2, 1869"],
                answer="1869",
                observation="Mahatma Gandhi delivered the Quit India speech.",
            ),
        ]
        diagnostics = diagnose_records(records)
        summary = summarize_diagnostics(diagnostics)
        report = build_markdown_report(diagnostics, title="Test Report")

        self.assertEqual(summary.total, 2)
        self.assertEqual(summary.wrong_valid, 2)
        self.assertEqual(summary.possible_alias_match, 1)
        self.assertEqual(summary.answer_granularity_miss, 1)
        self.assertEqual(summary.multi_candidate_answer, 0)
        self.assertIn("# Test Report", report)
        self.assertIn("bad_max_search_loop", report)

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "diagnostics.jsonl"
            write_diagnostic_jsonl(diagnostics, path)
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(len(rows), 2)
        self.assertTrue(rows[0]["possible_alias_match"])

    def test_load_records_skips_summary_records(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "eval.jsonl"
            path.write_text(
                json.dumps(make_record(question="Q?", answers=["A"], answer="B"))
                + "\n"
                + json.dumps({"type": "summary", "metrics": {}})
                + "\n",
                encoding="utf-8",
            )

            records = load_records(path)

        self.assertEqual(len(records), 1)


if __name__ == "__main__":
    unittest.main()
