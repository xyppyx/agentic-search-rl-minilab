"""Tests for offline reward sensitivity analysis."""

from __future__ import annotations

import unittest

from search_r1_minilab.reward_sensitivity import (
    build_markdown_report,
    default_sensitivity_configs,
    load_configs,
    parse_sensitivity_config,
    rescore_record,
    summarize_results,
)


def make_record(
    *,
    answer: str = "Wrong answer with enough words to trigger verbose penalty",
    exact_match: bool = False,
    valid_format: bool = True,
    search_calls: int = 1,
    stop_reason: str = "answer",
    duplicate_query: bool = False,
    empty_observation: bool = False,
    question: str = "Who is the maternal grandfather of Claudia Antonia?",
    answers: list[str] | None = None,
) -> dict:
    """Build a minimal trajectory record for sensitivity tests."""
    queries = ["Claudia Antonia maternal grandfather"]
    if duplicate_query:
        queries.append("Claudia Antonia maternal grandfather")
    turns = []
    for query in queries:
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
                "items": [] if empty_observation else [{"title": "Evidence"}],
                "observation": (
                    ""
                    if empty_observation
                    else "Claudia Antonia was the daughter of Claudius and Aelia Paetina."
                ),
            }
        )
    final_text = "No final answer" if not valid_format else f"Reasoning.\nAnswer: {answer}"
    turns.append({"role": "assistant", "parsed_kind": "answer", "text": final_text})
    return {
        "metadata": {"id": "case-1", "stop_reason": stop_reason},
        "question": question,
        "answers": answers or ["Sextus Aelius Catus"],
        "exact_match": exact_match,
        "valid_format": valid_format,
        "search_calls": search_calls,
        "turns": turns,
    }


class RewardSensitivityTest(unittest.TestCase):
    def test_base_reward_reconstruction(self) -> None:
        base_config = default_sensitivity_configs()[0]

        correct = rescore_record(make_record(answer="A", answers=["A"], exact_match=True), base_config)
        wrong = rescore_record(make_record(), base_config)
        invalid = rescore_record(make_record(valid_format=False), base_config)

        self.assertEqual(correct.base_reward, 1.0)
        self.assertEqual(wrong.base_reward, 0.0)
        self.assertEqual(invalid.base_reward, -0.1)

    def test_v1_is_stronger_than_v2_for_same_bad_trajectory(self) -> None:
        record = make_record(
            duplicate_query=True,
            empty_observation=True,
            search_calls=2,
            answer="This verbose wrong answer has many tokens in it",
        )
        configs = {config.name: config for config in default_sensitivity_configs()}

        v1 = rescore_record(record, configs["penalty_v1"])
        v2 = rescore_record(record, configs["penalty_v2_candidate"])

        self.assertLess(v1.final_reward, v2.final_reward)
        self.assertEqual(v1.duplicate_query_penalty, 0.05)
        self.assertEqual(v1.empty_result_penalty, 0.03)
        self.assertEqual(v1.verbose_answer_penalty, 0.02)
        self.assertEqual(v2.duplicate_query_penalty, 0.03)
        self.assertEqual(v2.empty_result_penalty, 0.01)
        self.assertEqual(v2.verbose_answer_penalty, 0.0)

    def test_correct_trajectory_is_not_penalized(self) -> None:
        record = make_record(
            answer="Sextus Aelius Catus",
            exact_match=True,
            duplicate_query=True,
            empty_observation=True,
        )
        v1 = default_sensitivity_configs()[1]

        result = rescore_record(record, v1)

        self.assertEqual(result.base_reward, 1.0)
        self.assertEqual(result.final_reward, 1.0)
        self.assertFalse(result.penalized)

    def test_missing_followup_is_not_penalized_by_v2_for_search_count(self) -> None:
        record = make_record(
            answer="Gaius Silius",
            search_calls=1,
            question="Who is the maternal grandfather of Claudia Antonia?",
        )
        configs = {config.name: config for config in default_sensitivity_configs()}

        result = rescore_record(record, configs["penalty_v2_candidate"])

        self.assertTrue(result.missing_followup_query)
        self.assertEqual(result.final_reward, result.base_reward)
        self.assertFalse(result.penalized)

    def test_v3_penalizes_granularity_and_multi_candidate_answers(self) -> None:
        configs = {config.name: config for config in default_sensitivity_configs()}
        date_record = make_record(
            question='When was the person who delivered the "Quit India" speech born?',
            answers=["October 2, 1869"],
            answer="1869",
        )
        multi_record = make_record(
            question="Who was the director of My Last Day?",
            answers=["Barry Cook"],
            answer="Barry Cook (for the 2011 film) or Liu Bicheng (for the 2020 film)",
        )

        date_result = rescore_record(date_record, configs["penalty_v3_followup_aware"])
        multi_result = rescore_record(multi_record, configs["penalty_v3_followup_aware"])

        self.assertEqual(date_result.date_granularity_penalty, 0.05)
        self.assertEqual(multi_result.multi_candidate_answer_penalty, 0.02)
        self.assertTrue(date_result.penalized)
        self.assertTrue(multi_result.penalized)

    def test_v3_bad_max_search_penalty_is_followup_aware(self) -> None:
        configs = {config.name: config for config in default_sensitivity_configs()}
        repeated = make_record(
            valid_format=False,
            search_calls=4,
            stop_reason="max_search_calls",
            duplicate_query=True,
        )
        helpful = make_record(
            answer="Aelius Sejanus",
            search_calls=3,
            question="Who is the maternal grandfather of Claudia Antonia?",
        )
        helpful["turns"] = make_record(
            answer="Aelius Sejanus",
            search_calls=3,
            question="Who is the maternal grandfather of Claudia Antonia?",
            duplicate_query=True,
        )["turns"][:-1]
        helpful["turns"][2]["tool_call"]["query"] = "Aelia Paetina father"
        helpful["turns"][2]["text"] = "<tool_call>Aelia Paetina father</tool_call>"
        helpful["turns"].extend(
            [
                {
                    "role": "assistant",
                    "parsed_kind": "tool",
                    "tool_call": {"name": "search", "query": "Sextus Aelius Catus father"},
                    "text": "<tool_call>Sextus Aelius Catus father</tool_call>",
                },
                {
                    "role": "tool",
                    "ok": True,
                    "items": [{"title": "Evidence"}],
                    "observation": "Aelia Paetina was the daughter of Sextus Aelius Catus.",
                },
                {
                    "role": "assistant",
                    "parsed_kind": "answer",
                    "text": "Reasoning.\nAnswer: Aelius Sejanus",
                },
            ]
        )

        repeated_result = rescore_record(repeated, configs["penalty_v3_followup_aware"])
        helpful_result = rescore_record(helpful, configs["penalty_v3_followup_aware"])

        self.assertEqual(repeated_result.bad_max_search_penalty, 0.01)
        self.assertTrue(repeated_result.bad_max_search_loop)
        self.assertEqual(helpful_result.bad_max_search_penalty, 0.0)
        self.assertTrue(helpful_result.helpful_followup_query)

    def test_v4_boosts_helpful_wrong_valid_followup_only(self) -> None:
        configs = {config.name: config for config in default_sensitivity_configs()}
        helpful_wrong = make_record(
            answer="Aelia Paetina",
            search_calls=2,
            duplicate_query=True,
            question="Who is the maternal grandfather of Claudia Antonia?",
        )
        helpful_wrong["turns"][2]["tool_call"]["query"] = "Aelia Paetina father"
        helpful_wrong["turns"][2]["text"] = "<tool_call>Aelia Paetina father</tool_call>"
        helpful_correct = make_record(
            answer="Sextus Aelius Catus",
            answers=["Sextus Aelius Catus"],
            exact_match=True,
            search_calls=2,
            duplicate_query=True,
            question="Who is the maternal grandfather of Claudia Antonia?",
        )
        helpful_correct["turns"][2]["tool_call"]["query"] = "Aelia Paetina father"
        helpful_correct["turns"][2]["text"] = "<tool_call>Aelia Paetina father</tool_call>"

        wrong_result = rescore_record(helpful_wrong, configs["reward_v4_followup_bonus"])
        correct_result = rescore_record(helpful_correct, configs["reward_v4_followup_bonus"])

        self.assertTrue(wrong_result.helpful_followup_query)
        self.assertEqual(wrong_result.helpful_followup_bonus, 0.02)
        self.assertAlmostEqual(wrong_result.final_reward, 0.02)
        self.assertTrue(wrong_result.boosted)
        self.assertEqual(correct_result.helpful_followup_bonus, 0.0)
        self.assertEqual(correct_result.final_reward, 1.0)
        self.assertFalse(correct_result.boosted)

    def test_max_search_penalty_applies_to_invalid_max_search_case(self) -> None:
        record = make_record(
            valid_format=False,
            search_calls=4,
            stop_reason="max_search_calls",
        )
        configs = {config.name: config for config in default_sensitivity_configs()}

        result = rescore_record(record, configs["penalty_v1"])

        self.assertEqual(result.base_reward, -0.1)
        self.assertEqual(result.max_search_no_answer_penalty, 0.05)
        self.assertEqual(result.final_reward, -0.15000000000000002)

    def test_parse_custom_config(self) -> None:
        config = parse_sensitivity_config(
            "custom:duplicate=0.03,empty=0.01,max_search=0,verbose=0,verbose_threshold=0"
        )

        self.assertEqual(config.name, "custom")
        self.assertEqual(config.reward_shaping.duplicate_query_penalty, 0.03)
        self.assertEqual(config.reward_shaping.empty_result_penalty, 0.01)

        v3 = parse_sensitivity_config(
            "v3:duplicate=0.03,empty=0.01,bad_max_search=0.01,"
            "date_granularity=0.05,multi_candidate=0.02,verbose=0,verbose_threshold=0"
        )
        self.assertEqual(v3.reward_shaping.bad_max_search_penalty, 0.01)
        self.assertEqual(v3.reward_shaping.date_granularity_penalty, 0.05)
        self.assertEqual(v3.reward_shaping.multi_candidate_answer_penalty, 0.02)

        v4 = parse_sensitivity_config("v4:helpful_followup=0.02")
        self.assertEqual(v4.reward_shaping.helpful_followup_bonus, 0.02)

        with self.assertRaises(ValueError):
            parse_sensitivity_config("bad:unknown=1")
        with self.assertRaises(ValueError):
            parse_sensitivity_config("bad:duplicate=-0.1")

    def test_summary_and_report_include_audit_fields(self) -> None:
        configs = load_configs([])
        records = [
            make_record(duplicate_query=True),
            make_record(answer="Sextus Aelius Catus", exact_match=True),
        ]
        results = [
            rescore_record(record, config)
            for config in configs
            for record in records
        ]

        summaries = summarize_results(results)
        report = build_markdown_report(configs, results, title="Sensitivity")

        v1_summary = next(item for item in summaries if item.config_name == "penalty_v1")
        self.assertEqual(v1_summary.penalized_count, 1)
        self.assertEqual(v1_summary.correct_penalized_count, 0)
        self.assertIn("reward_v4_followup_bonus", report)
        self.assertIn("helpful_followup_bonus", report)
        self.assertIn("missing_followup", report)
        self.assertIn("# Sensitivity", report)


if __name__ == "__main__":
    unittest.main()
