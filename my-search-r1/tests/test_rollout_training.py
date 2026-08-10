"""Tests for training-grade rollout and PyTRIO training helpers."""

from __future__ import annotations

import unittest
from dataclasses import dataclass
from pathlib import Path

from search_r1_minilab.data import SearchExample
from search_r1_minilab.rollout import (
    AssistantTurn,
    RolloutConfig,
    Trajectory,
    assign_group_advantages,
    rollout_batch,
    score_trajectory,
    trajectory_to_record,
)
from search_r1_minilab.rewards import RewardShapingConfig
from search_r1_minilab.tooling import BackendConfig, build_registry
from search_r1_minilab.tools import LocalBM25Backend, ToolRegistry
from search_r1_minilab.training import (
    add_reference_logprobs,
    build_custom_forward_datums,
    build_datum,
    build_training_datums,
    compute_reference_logprobs,
    datum_loss_token_count,
    evaluation_metrics,
    pack_micro_batches,
    TurnCreditConfig,
    weight_micro_batch_for_global_mean,
    weight_micro_batch_items_for_global_mean,
)
from search_r1_minilab.trajectories import build_markdown_report


FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


class TrainingRolloutTest(unittest.TestCase):
    def test_group_rollout_scores_and_assigns_advantages(self) -> None:
        registry = ToolRegistry()
        registry.register(LocalBM25Backend.from_jsonl(FIXTURES_DIR / "bm25_corpus.jsonl"))

        trajectories = rollout_batch(
            FakeSamplingClient([["Answer: A", "Answer: B"]]),
            FakeTokenizer(),
            registry,
            "local_bm25",
            [
                SearchExample(
                    id="q1",
                    question="Pick A",
                    answers=["A"],
                    data_source="test",
                )
            ],
            RolloutConfig(group_size=2),
        )

        self.assertEqual([item.reward for item in trajectories], [1.0, 0.0])
        self.assertEqual([item.advantage for item in trajectories], [0.5, -0.5])
        self.assertEqual(len(build_training_datums(trajectories)), 2)

    def test_standardized_group_advantages_can_be_clipped(self) -> None:
        trajectories = [
            Trajectory(
                example=SearchExample(f"q{index}", "Question?", ["A"], "test"),
                group_index=index,
                messages=[],
                question_index=0,
                reward=reward,
            )
            for index, reward in enumerate([1.0, 0.0])
        ]

        assign_group_advantages(
            trajectories,
            normalization="standardize",
            clip=0.75,
        )

        self.assertEqual([item.advantage for item in trajectories], [0.75, -0.75])

    def test_tool_rollout_records_failure_for_shared_schema(self) -> None:
        registry = ToolRegistry()
        registry.register(LocalBM25Backend.from_jsonl(FIXTURES_DIR / "bm25_corpus.jsonl"))

        trajectories = rollout_batch(
            FakeSamplingClient([[_tool_call("???")], "Answer: unknown"]),
            FakeTokenizer(),
            registry,
            "local_bm25",
            [
                SearchExample(
                    id="q2",
                    question="Who wrote The Little Prince?",
                    answers=["Antoine de Saint-Exupery"],
                    data_source="test",
                )
            ],
            RolloutConfig(group_size=1),
        )
        record = trajectory_to_record(trajectories[0], run_type="eval")
        report = build_markdown_report([record], title="Failure")

        self.assertEqual(record["search_calls"], 1)
        self.assertEqual(record["tool_failures"], 1)
        self.assertEqual(record["turns"][0]["tool_call"]["name"], "search")
        self.assertEqual(record["turns"][1]["error_type"], "empty_query")
        self.assertIn("## Tool Failure Cases", report)

    def test_max_search_calls_stops_before_registry_call(self) -> None:
        registry = ToolRegistry()
        registry.register(LocalBM25Backend.from_jsonl(FIXTURES_DIR / "bm25_corpus.jsonl"))

        trajectories = rollout_batch(
            FakeSamplingClient([[_tool_call("little prince")]]),
            FakeTokenizer(),
            registry,
            "local_bm25",
            [
                SearchExample(
                    id="q3",
                    question="Who wrote The Little Prince?",
                    answers=["Antoine de Saint-Exupery"],
                    data_source="test",
                )
            ],
            RolloutConfig(group_size=1, max_search_calls=0),
        )

        self.assertEqual(trajectories[0].search_calls, 0)
        self.assertEqual(trajectories[0].stop_reason, "max_search_calls")
        self.assertEqual(registry.metrics()["tool/local_bm25/requests"], 0.0)

    def test_prompt_reconstruction_failure_stops_single_trajectory(self) -> None:
        registry = ToolRegistry()
        registry.register(LocalBM25Backend.from_jsonl(FIXTURES_DIR / "bm25_corpus.jsonl"))

        trajectories = rollout_batch(
            FakeSamplingClient([[_tool_call("little prince")]]),
            BrokenAssistantBoundaryTokenizer(),
            registry,
            "local_bm25",
            [
                SearchExample(
                    id="q-boundary",
                    question="Who wrote The Little Prince?",
                    answers=["Antoine de Saint-Exupery"],
                    data_source="test",
                )
            ],
            RolloutConfig(group_size=1),
        )

        self.assertEqual(trajectories[0].search_calls, 1)
        self.assertEqual(trajectories[0].stop_reason, "prompt_reconstruction_failed")
        self.assertEqual(trajectory_to_record(trajectories[0], run_type="eval")["tool_failures"], 0)

    def test_build_datum_aligns_shifted_advantages(self) -> None:
        trajectory = Trajectory(
            example=SearchExample("q4", "Question?", ["A"], "test"),
            group_index=0,
            messages=[],
            advantage=2.0,
            turns=[
                AssistantTurn([1, 2], [3, 4], [0.1, 0.2], "ab"),
                AssistantTurn([1, 2, 3, 4, 5], [6], [0.3], "c"),
            ],
        )

        datum = build_datum(trajectory)
        advantages = datum.datum.loss_fn_inputs["advantages"].to_numpy().tolist()

        self.assertEqual(datum.num_tokens, 5)
        self.assertEqual(advantages, [0.0, 2.0, 2.0, 0.0, 2.0])
        self.assertEqual(datum_loss_token_count(datum), 3)

    def test_turn_credit_none_preserves_trajectory_advantage(self) -> None:
        trajectory = _turn_credit_candidate(advantage=-0.5)

        datums = build_training_datums(
            [trajectory],
            TurnCreditConfig(policy="none", helpful_search_turn_bonus=0.10),
        )
        advantages = datums[0].datum.loss_fn_inputs["advantages"].to_numpy().tolist()

        self.assertEqual(advantages, [0.0, -0.5, 0.0, -0.5])
        self.assertEqual(trajectory.turns[1].credit_label, "")

    def test_helpful_bridge_turn_credit_overrides_wrong_valid_search_turn(self) -> None:
        trajectory = _turn_credit_candidate(advantage=-0.5)

        datums = build_training_datums(
            [trajectory],
            TurnCreditConfig(
                policy="helpful_bridge",
                helpful_search_turn_bonus=0.10,
            ),
        )
        advantages = datums[0].datum.loss_fn_inputs["advantages"].to_numpy().tolist()
        record = trajectory_to_record(trajectory, run_type="train")

        self.assertEqual(advantages[:3], [0.0, -0.5, 0.0])
        self.assertAlmostEqual(advantages[3], 0.1)
        self.assertEqual(trajectory.turns[1].credit_label, "helpful_bridge_search")
        self.assertEqual(
            record["metadata"]["turn_credits"][0]["query"],
            "Aelia Paetina father",
        )

    def test_turn_credit_rejects_repeated_first_empty_invalid_and_correct(self) -> None:
        config = TurnCreditConfig(
            policy="helpful_bridge",
            helpful_search_turn_bonus=0.10,
        )
        candidates = [
            _turn_credit_candidate(advantage=-0.5, second_query="Claudia Antonia mother"),
            _turn_credit_candidate(advantage=-0.5, empty_previous_observation=True),
            _turn_credit_candidate(advantage=-0.5, valid_format=False),
            _turn_credit_candidate(advantage=-0.5, exact_match=True),
            _turn_credit_candidate(advantage=-0.5, search_calls=1),
        ]

        datums = build_training_datums(candidates, config)

        self.assertEqual(len(datums), len(candidates))
        for trajectory in candidates:
            self.assertTrue(all(not turn.credit_label for turn in trajectory.turns))

    def test_turn_credit_builds_datum_when_group_advantage_is_zero(self) -> None:
        trajectory = _turn_credit_candidate(advantage=0.0)

        datums = build_training_datums(
            [trajectory],
            TurnCreditConfig(
                policy="helpful_bridge",
                helpful_search_turn_bonus=0.10,
            ),
        )

        self.assertEqual(len(datums), 1)
        self.assertEqual(datum_loss_token_count(datums[0]), 1)

    def test_evidence_bridge_rewards_entity_backed_second_search(self) -> None:
        trajectory = _turn_credit_candidate(
            advantage=-0.5,
            second_query="Aelia Paetina parents",
        )

        datums = build_training_datums(
            [trajectory],
            TurnCreditConfig(
                policy="evidence_bridge",
                evidence_search_turn_bonus=0.10,
                early_answer_turn_penalty=0.05,
            ),
        )
        advantages = datums[0].datum.loss_fn_inputs["advantages"].to_numpy().tolist()
        record = trajectory_to_record(trajectory, run_type="train")

        self.assertEqual(advantages[:3], [0.0, -0.5, 0.0])
        self.assertAlmostEqual(advantages[3], 0.1)
        self.assertEqual(trajectory.turns[1].credit_label, "evidence_bridge_search")
        self.assertEqual(
            record["metadata"]["turn_credits"][0]["label"],
            "evidence_bridge_search",
        )
        self.assertEqual(
            record["metadata"]["turn_credits"][0]["query"],
            "Aelia Paetina parents",
        )

    def test_evidence_bridge_rejects_weak_or_ineligible_search_turns(self) -> None:
        config = TurnCreditConfig(
            policy="evidence_bridge",
            evidence_search_turn_bonus=0.10,
            early_answer_turn_penalty=0.05,
        )
        candidates = [
            _turn_credit_candidate(advantage=-0.5, search_calls=1),
            _turn_credit_candidate(advantage=-0.5, second_query="Claudia Antonia mother"),
            _turn_credit_candidate(advantage=-0.5, empty_previous_observation=True),
            _turn_credit_candidate(advantage=-0.5, empty_current_observation=True),
            _turn_credit_candidate(advantage=-0.5, valid_format=False),
            _turn_credit_candidate(advantage=-0.5, exact_match=True),
        ]

        build_training_datums(candidates, config)

        for trajectory in candidates:
            self.assertNotIn(
                "evidence_bridge_search",
                [turn.credit_label for turn in trajectory.turns],
            )

    def test_early_answer_penalty_marks_missing_followup_final_turn(self) -> None:
        trajectory = _turn_credit_candidate(
            advantage=-0.5,
            search_calls=1,
            include_final_answer_turn=True,
            final_text="Answer: Tiberius",
        )

        datums = build_training_datums(
            [trajectory],
            TurnCreditConfig(
                policy="evidence_bridge",
                evidence_search_turn_bonus=0.10,
                early_answer_turn_penalty=0.05,
            ),
        )
        advantages = datums[0].datum.loss_fn_inputs["advantages"].to_numpy().tolist()
        record = trajectory_to_record(trajectory, run_type="train")

        self.assertAlmostEqual(advantages[-1], -0.55)
        self.assertEqual(
            trajectory.turns[-1].credit_label,
            "early_answer_missing_followup",
        )
        self.assertEqual(
            record["metadata"]["turn_credits"][0]["label"],
            "early_answer_missing_followup",
        )

    def test_early_answer_penalty_rejects_correct_invalid_simple_and_empty(self) -> None:
        config = TurnCreditConfig(
            policy="evidence_bridge",
            evidence_search_turn_bonus=0.10,
            early_answer_turn_penalty=0.05,
        )
        candidates = [
            _turn_credit_candidate(
                advantage=-0.5,
                search_calls=1,
                include_final_answer_turn=True,
                exact_match=True,
            ),
            _turn_credit_candidate(
                advantage=-0.5,
                search_calls=1,
                include_final_answer_turn=True,
                valid_format=False,
            ),
            _turn_credit_candidate(
                advantage=-0.5,
                search_calls=1,
                include_final_answer_turn=True,
                empty_previous_observation=True,
            ),
            _turn_credit_candidate(
                advantage=-0.5,
                search_calls=1,
                include_final_answer_turn=True,
                question="Who wrote Hamlet?",
            ),
        ]

        build_training_datums(candidates, config)

        for trajectory in candidates:
            self.assertNotIn(
                "early_answer_missing_followup",
                [turn.credit_label for turn in trajectory.turns],
            )

    def test_final_hop_bridge_rewards_attribute_search(self) -> None:
        trajectory = _final_hop_candidate(
            advantage=-0.5,
            question="When did the president who succeeded Jimmy Carter die?",
            second_query="Ronald Reagan death date",
            current_observation="Ronald Reagan died on June 5, 2004.",
            answers=["June 5, 2004"],
        )

        datums = build_training_datums(
            [trajectory],
            TurnCreditConfig(
                policy="final_hop_bridge",
                evidence_search_turn_bonus=0.05,
                final_hop_search_turn_bonus=0.10,
                early_answer_turn_penalty=0.05,
                missing_final_hop_turn_penalty=0.08,
            ),
        )
        advantages = datums[0].datum.loss_fn_inputs["advantages"].to_numpy().tolist()
        record = trajectory_to_record(trajectory, run_type="train")

        self.assertAlmostEqual(advantages[3], 0.1)
        self.assertEqual(
            trajectory.turns[1].credit_label,
            "final_hop_attribute_search",
        )
        self.assertEqual(
            record["metadata"]["turn_credits"][0]["label"],
            "final_hop_attribute_search",
        )

    def test_final_hop_bridge_does_not_reward_plain_evidence_search(self) -> None:
        trajectory = _turn_credit_candidate(
            advantage=-0.5,
            question="Who is connected to Claudia Antonia?",
            second_query="Aelia Paetina parents",
        )

        build_training_datums(
            [trajectory],
            TurnCreditConfig(
                policy="final_hop_bridge",
                final_hop_search_turn_bonus=0.10,
                missing_final_hop_turn_penalty=0.08,
            ),
        )

        self.assertTrue(all(not turn.credit_label for turn in trajectory.turns))

    def test_missing_final_hop_penalty_marks_multi_search_early_answer(self) -> None:
        trajectory = _final_hop_candidate(
            advantage=-0.5,
            question="Which film has the director who is older, Film A or Film B?",
            first_query="Film A director",
            first_observation="Film A was directed by John Smith.",
            second_query="John Smith movies",
            current_observation="John Smith directed Film A and other films.",
            final_text="Answer: Film A",
            answers=["Film B"],
            include_final_answer_turn=True,
        )

        datums = build_training_datums(
            [trajectory],
            TurnCreditConfig(
                policy="final_hop_bridge",
                final_hop_search_turn_bonus=0.10,
                missing_final_hop_turn_penalty=0.08,
            ),
        )
        advantages = datums[0].datum.loss_fn_inputs["advantages"].to_numpy().tolist()
        record = trajectory_to_record(trajectory, run_type="train")

        self.assertAlmostEqual(advantages[-1], -0.58)
        self.assertEqual(
            trajectory.turns[-1].credit_label,
            "missing_final_hop_attribute",
        )
        self.assertEqual(
            record["metadata"]["turn_credits"][0]["label"],
            "missing_final_hop_attribute",
        )

    def test_missing_final_hop_penalty_rejects_correct_invalid_empty_and_covered(self) -> None:
        config = TurnCreditConfig(
            policy="final_hop_bridge",
            missing_final_hop_turn_penalty=0.08,
        )
        candidates = [
            _final_hop_candidate(
                advantage=-0.5,
                exact_match=True,
                include_final_answer_turn=True,
            ),
            _final_hop_candidate(
                advantage=-0.5,
                valid_format=False,
                include_final_answer_turn=True,
            ),
            _final_hop_candidate(
                advantage=-0.5,
                empty_first_observation=True,
                empty_current_observation=True,
                include_final_answer_turn=True,
            ),
            _final_hop_candidate(
                advantage=-0.5,
                second_query="Ronald Reagan death date",
                current_observation="Ronald Reagan died on June 5, 2004.",
                include_final_answer_turn=True,
            ),
            _final_hop_candidate(
                advantage=-0.5,
                second_query="Ronald Reagan biography",
                current_observation="Ronald Reagan died on June 5, 2004.",
                final_text="Answer: June 2004",
                include_final_answer_turn=True,
            ),
        ]

        build_training_datums(candidates, config)

        for trajectory in candidates:
            self.assertNotIn(
                "missing_final_hop_attribute",
                [turn.credit_label for turn in trajectory.turns],
            )

    def test_missing_final_hop_penalty_requires_explicit_attribute_query(self) -> None:
        trajectory = _final_hop_candidate(
            advantage=-0.5,
            question="When did the president who succeeded Jimmy Carter die?",
            second_query="Ronald Reagan biography",
            current_observation="Ronald Reagan was president from 1981 to 1989.",
            include_final_answer_turn=True,
        )

        build_training_datums(
            [trajectory],
            TurnCreditConfig(
                policy="final_hop_bridge",
                missing_final_hop_turn_penalty=0.08,
            ),
        )

        self.assertEqual(
            trajectory.turns[-1].credit_label,
            "missing_final_hop_attribute",
        )

    def test_final_answer_guard_penalizes_max_search_and_invalid_format(self) -> None:
        max_search = _final_hop_candidate(
            advantage=-0.5,
            valid_format=False,
            final_text=_tool_call("Ronald Reagan death date"),
            stop_reason="max_search_calls",
        )
        invalid_answer = _final_hop_candidate(
            advantage=-0.5,
            valid_format=False,
            include_final_answer_turn=True,
            final_text="June 2004",
            stop_reason="invalid_format",
        )

        datums = build_training_datums(
            [max_search, invalid_answer],
            TurnCreditConfig(
                policy="final_hop_bridge",
                final_answer_guard_turn_penalty=0.06,
            ),
        )

        self.assertAlmostEqual(
            datums[0].datum.loss_fn_inputs["advantages"].to_numpy().tolist()[-1],
            -0.56,
        )
        self.assertEqual(max_search.turns[-1].credit_label, "final_answer_guard")
        self.assertEqual(invalid_answer.turns[-1].credit_label, "final_answer_guard")

    def test_final_answer_guard_rejects_correct_and_unsearched_invalid(self) -> None:
        correct = _final_hop_candidate(
            advantage=0.5,
            exact_match=True,
            valid_format=True,
            include_final_answer_turn=True,
            final_text="Answer: June 5, 2004",
        )
        unsearched_invalid = Trajectory(
            example=SearchExample("q-invalid", "Question?", ["A"], "test"),
            group_index=0,
            messages=[],
            search_calls=0,
            final_text="A",
            reward=-0.1,
            advantage=-0.5,
            valid_format=False,
            exact_match=False,
            stop_reason="invalid_format",
            events=[{"role": "assistant", "text": "A", "parsed_kind": "invalid"}],
            turns=[AssistantTurn([1], [2], [0.1], "A")],
        )

        build_training_datums(
            [correct, unsearched_invalid],
            TurnCreditConfig(
                policy="final_hop_bridge",
                final_answer_guard_turn_penalty=0.06,
            ),
        )

        self.assertFalse(any(turn.credit_label for turn in correct.turns))
        self.assertFalse(any(turn.credit_label for turn in unsearched_invalid.turns))

    def test_weight_micro_batch_scales_advantages(self) -> None:
        trajectories = [
            Trajectory(
                example=SearchExample(f"q{index}", "Question?", ["A"], "test"),
                group_index=index,
                messages=[],
                advantage=1.0,
                turns=[AssistantTurn([1], [2, 3], [0.1, 0.2], "ab")],
            )
            for index in range(2)
        ]
        micro_batches = pack_micro_batches([build_datum(item) for item in trajectories])

        weighted = weight_micro_batch_for_global_mean(micro_batches[0], total_samples=4)
        advantages = weighted[0].loss_fn_inputs["advantages"].to_numpy().tolist()

        self.assertEqual(advantages, [0.5, 0.5])

    def test_weighted_micro_batch_items_preserve_reference_logprobs(self) -> None:
        trajectory = Trajectory(
            example=SearchExample("q-ref", "Question?", ["A"], "test"),
            group_index=0,
            messages=[],
            advantage=1.0,
            turns=[AssistantTurn([1], [2, 3], [0.1, 0.2], "ab")],
        )
        datum = build_datum(trajectory)
        datum.reference_logprobs = [-0.4, -0.5]

        weighted = weight_micro_batch_items_for_global_mean([datum], total_samples=2)

        self.assertEqual(weighted[0].reference_logprobs, [-0.4, -0.5])
        self.assertEqual(
            weighted[0].datum.loss_fn_inputs["advantages"].to_numpy().tolist(),
            [0.5, 0.5],
        )

    def test_reference_logprobs_align_to_shifted_targets(self) -> None:
        trajectory = Trajectory(
            example=SearchExample("q-kl", "Question?", ["A"], "test"),
            group_index=0,
            messages=[],
            advantage=1.0,
            turns=[AssistantTurn([10, 11], [12, 13], [0.1, 0.2], "ab")],
        )
        datum = build_datum(trajectory)
        reference = FakeReferenceClient()

        ref_logprobs = compute_reference_logprobs(datum, reference)

        self.assertEqual(reference.requests, [[10, 11, 12, 13]])
        self.assertEqual(ref_logprobs, [-0.11, -0.12, -0.13])

    def test_add_reference_logprobs_and_custom_forward_datums(self) -> None:
        trajectory = Trajectory(
            example=SearchExample("q-custom", "Question?", ["A"], "test"),
            group_index=0,
            messages=[],
            advantage=1.0,
            turns=[AssistantTurn([1], [2], [0.1], "a")],
        )
        datums = add_reference_logprobs([build_datum(trajectory)], FakeReferenceClient())

        custom = build_custom_forward_datums(datums)

        self.assertEqual(datums[0].reference_logprobs, [-0.02])
        self.assertEqual(set(custom[0].loss_fn_inputs.keys()), {"target_tokens"})

    def test_failure_wrapper_preserves_dispatch_backend_name(self) -> None:
        registry = build_registry(
            BackendConfig(
                backend="local_bm25",
                bm25_corpus=FIXTURES_DIR / "bm25_corpus.jsonl",
                p_timeout=1.0,
                failure_seed=123,
            )
        )

        result = registry.call("local_bm25", {"query": "little prince"})

        self.assertFalse(result.ok)
        self.assertEqual(result.backend, "local_bm25")
        self.assertEqual(result.error_type, "timeout")

    def test_reward_shaping_default_preserves_base_reward(self) -> None:
        trajectory = _penalty_candidate(final_text="not an answer")

        score_trajectory(trajectory, FakeTokenizer(), RewardShapingConfig())
        record = trajectory_to_record(trajectory, run_type="eval")

        self.assertEqual(trajectory.reward, -0.1)
        self.assertFalse(trajectory.valid_format)
        self.assertEqual(record["metadata"]["reward_components"]["base_reward"], -0.1)
        self.assertEqual(
            record["metadata"]["reward_components"]["duplicate_query_penalty"],
            0.0,
        )
        self.assertEqual(record["metadata"]["reward_components"]["final_reward"], -0.1)

    def test_reward_shaping_applies_enabled_behavior_penalties(self) -> None:
        trajectory = _penalty_candidate(final_text="not an answer")

        score_trajectory(
            trajectory,
            FakeTokenizer(),
            RewardShapingConfig(
                duplicate_query_penalty=0.05,
                empty_result_penalty=0.03,
                max_search_no_answer_penalty=0.05,
            ),
        )

        self.assertAlmostEqual(trajectory.reward, -0.23)
        self.assertEqual(
            trajectory.reward_components["duplicate_query_penalty"],
            0.05,
        )
        self.assertEqual(trajectory.reward_components["empty_result_penalty"], 0.03)
        self.assertEqual(
            trajectory.reward_components["max_search_no_answer_penalty"],
            0.05,
        )

    def test_verbose_answer_penalty_only_applies_to_long_wrong_answer(self) -> None:
        long_wrong = Trajectory(
            example=SearchExample("q6", "Question?", ["to"], "test"),
            group_index=0,
            messages=[],
            final_text="Answer: Veins carry blood to the heart.",
            stop_reason="answer",
        )
        short_correct = Trajectory(
            example=SearchExample("q7", "Question?", ["to"], "test"),
            group_index=0,
            messages=[],
            final_text="Answer: to",
            stop_reason="answer",
        )
        config = RewardShapingConfig(
            verbose_answer_penalty=0.02,
            verbose_answer_token_threshold=8,
        )

        score_trajectory(long_wrong, FakeTokenizer(), config)
        score_trajectory(short_correct, FakeTokenizer(), config)

        self.assertEqual(long_wrong.reward_components["verbose_answer_penalty"], 0.02)
        self.assertAlmostEqual(long_wrong.reward, -0.02)
        self.assertEqual(short_correct.reward_components["verbose_answer_penalty"], 0.0)
        self.assertEqual(short_correct.reward, 1.0)

    def test_v3_answer_penalties_apply_to_wrong_final_answers(self) -> None:
        date = Trajectory(
            example=SearchExample("q-date", "When was the person born?", ["October 2, 1869"], "test"),
            group_index=0,
            messages=[],
            final_text="Answer: 1869",
            stop_reason="answer",
        )
        multi = Trajectory(
            example=SearchExample("q-multi", "Who was the director?", ["Barry Cook"], "test"),
            group_index=0,
            messages=[],
            final_text="Answer: Barry Cook or Liu Bicheng",
            stop_reason="answer",
        )
        config = RewardShapingConfig(
            date_granularity_penalty=0.05,
            multi_candidate_answer_penalty=0.02,
        )

        score_trajectory(date, FakeTokenizer(), config)
        score_trajectory(multi, FakeTokenizer(), config)

        self.assertEqual(date.reward_components["date_granularity_penalty"], 0.05)
        self.assertEqual(multi.reward_components["multi_candidate_answer_penalty"], 0.02)
        self.assertAlmostEqual(date.reward, -0.05)
        self.assertAlmostEqual(multi.reward, -0.02)

    def test_helpful_followup_bonus_applies_to_wrong_valid_answer(self) -> None:
        trajectory = _helpful_followup_candidate(
            final_text="Reasoning.\nAnswer: Aelia Paetina",
            exact_answer="Sextus Aelius Catus",
        )

        score_trajectory(
            trajectory,
            FakeTokenizer(),
            RewardShapingConfig(helpful_followup_bonus=0.02),
        )

        self.assertAlmostEqual(trajectory.reward, 0.02)
        self.assertEqual(trajectory.reward_components["helpful_followup_bonus"], 0.02)

    def test_helpful_followup_bonus_does_not_apply_to_correct_or_invalid(self) -> None:
        correct = _helpful_followup_candidate(
            final_text="Reasoning.\nAnswer: Sextus Aelius Catus",
            exact_answer="Sextus Aelius Catus",
        )
        invalid = _helpful_followup_candidate(
            final_text="No final answer",
            exact_answer="Sextus Aelius Catus",
        )
        config = RewardShapingConfig(helpful_followup_bonus=0.02)

        score_trajectory(correct, FakeTokenizer(), config)
        score_trajectory(invalid, FakeTokenizer(), config)

        self.assertEqual(correct.reward, 1.0)
        self.assertEqual(correct.reward_components["helpful_followup_bonus"], 0.0)
        self.assertEqual(invalid.reward, -0.1)
        self.assertEqual(invalid.reward_components["helpful_followup_bonus"], 0.0)

    def test_no_search_penalty_applies_to_wrong_unsearched_answers_only(self) -> None:
        wrong = Trajectory(
            example=SearchExample("q-nosrch-wrong", "Question?", ["A"], "test"),
            group_index=0,
            messages=[],
            final_text="Answer: B",
            stop_reason="answer",
        )
        correct = Trajectory(
            example=SearchExample("q-nosrch-correct", "Question?", ["A"], "test"),
            group_index=0,
            messages=[],
            final_text="Answer: A",
            stop_reason="answer",
        )
        searched_wrong = Trajectory(
            example=SearchExample("q-searched-wrong", "Question?", ["A"], "test"),
            group_index=0,
            messages=[],
            search_calls=1,
            final_text="Answer: B",
            stop_reason="answer",
            events=[
                {
                    "role": "assistant",
                    "text": "",
                    "tool_call": {"name": "search", "query": "Question"},
                },
                {"role": "tool", "tool_name": "search", "ok": True, "items": []},
            ],
        )
        config = RewardShapingConfig(no_search_penalty=0.03)

        score_trajectory(wrong, FakeTokenizer(), config)
        score_trajectory(correct, FakeTokenizer(), config)
        score_trajectory(searched_wrong, FakeTokenizer(), config)

        self.assertEqual(wrong.reward_components["no_search_penalty"], 0.03)
        self.assertAlmostEqual(wrong.reward, -0.03)
        self.assertEqual(correct.reward_components["no_search_penalty"], 0.0)
        self.assertEqual(correct.reward, 1.0)
        self.assertEqual(searched_wrong.reward_components["no_search_penalty"], 0.0)
        self.assertEqual(searched_wrong.reward, 0.0)

    def test_evaluation_metrics_include_behavior_rates(self) -> None:
        direct = Trajectory(
            example=SearchExample("q8", "Question?", ["A"], "test"),
            group_index=0,
            messages=[],
            reward=1.0,
            valid_format=True,
            exact_match=True,
            stop_reason="answer",
        )
        searched_wrong = Trajectory(
            example=SearchExample("q9", "Question?", ["B"], "test"),
            group_index=0,
            messages=[],
            reward=0.0,
            valid_format=True,
            exact_match=False,
            search_calls=1,
            stop_reason="answer",
            events=[
                {
                    "role": "assistant",
                    "text": "",
                    "tool_call": {"name": "search", "query": "q"},
                },
                {"role": "tool", "tool_name": "search", "ok": True, "items": []},
            ],
        )

        metrics = evaluation_metrics([direct, searched_wrong])

        self.assertEqual(metrics["behavior/direct_correct_rate"], 0.5)
        self.assertEqual(metrics["behavior/searched_wrong_rate"], 0.5)
        self.assertEqual(metrics["behavior/empty_observation_rate"], 1.0)


def _tool_call(query: str) -> str:
    return (
        "<tool_call><function=search><parameter=query>"
        f"{query}"
        "</parameter></function></tool_call>"
    )


def _turn_credit_candidate(
    *,
    advantage: float,
    second_query: str = "Aelia Paetina father",
    empty_previous_observation: bool = False,
    empty_current_observation: bool = False,
    valid_format: bool = True,
    exact_match: bool = False,
    search_calls: int = 2,
    include_final_answer_turn: bool = False,
    final_text: str = "Answer: Aelia Paetina",
    question: str = "Who is the maternal grandfather of Claudia Antonia?",
) -> Trajectory:
    previous_items = (
        []
        if empty_previous_observation
        else [
            {
                "title": "Aelia Paetina",
                "content": "Claudia Antonia was the daughter of Aelia Paetina.",
            }
        ]
    )
    current_items = (
        []
        if empty_current_observation
        else [
            {
                "title": "Sextus Aelius Catus",
                "content": "Aelia Paetina was the daughter of Sextus Aelius Catus.",
            }
        ]
    )
    events = [
        {
            "role": "assistant",
            "text": _tool_call("Claudia Antonia mother"),
            "tool_call": {"name": "search", "query": "Claudia Antonia mother"},
        },
        {
            "role": "tool",
            "tool_name": "search",
            "ok": True,
            "items": previous_items,
            "observation": "Claudia Antonia was the daughter of Aelia Paetina.",
        },
        {
            "role": "assistant",
            "text": _tool_call(second_query),
            "tool_call": {"name": "search", "query": second_query},
        },
        {
            "role": "tool",
            "tool_name": "search",
            "ok": True,
            "items": current_items,
            "observation": "Aelia Paetina was the daughter of Sextus Aelius Catus.",
        },
    ]
    if search_calls == 1:
        events = events[:2]
    turns = [
        AssistantTurn([1, 2], [3], [0.1], _tool_call("Claudia Antonia mother")),
    ]
    if search_calls != 1:
        turns.append(
            AssistantTurn([1, 2, 3, 4], [5], [0.2], _tool_call(second_query))
        )
    if include_final_answer_turn:
        events.append({"role": "assistant", "text": final_text, "parsed_kind": "answer"})
        turns.append(AssistantTurn([1, 2, 3, 4, 5], [6], [0.3], final_text))
    return Trajectory(
        example=SearchExample(
            "q-turn-credit",
            question,
            ["Sextus Aelius Catus"],
            "test",
        ),
        group_index=0,
        messages=[],
        search_calls=search_calls,
        final_text=final_text,
        reward=float(exact_match),
        advantage=advantage,
        valid_format=valid_format,
        exact_match=exact_match,
        stop_reason="answer",
        events=events,
        turns=turns,
    )


def _final_hop_candidate(
    *,
    advantage: float,
    question: str = "When did the president who succeeded Jimmy Carter die?",
    first_query: str = "president after Jimmy Carter",
    first_observation: str = "Ronald Reagan succeeded Jimmy Carter.",
    second_query: str = "Ronald Reagan biography",
    current_observation: str = "Ronald Reagan was the 40th president.",
    answers: list[str] | None = None,
    empty_first_observation: bool = False,
    empty_current_observation: bool = False,
    valid_format: bool = True,
    exact_match: bool = False,
    include_final_answer_turn: bool = False,
    final_text: str = "Answer: March 2004",
    stop_reason: str = "answer",
) -> Trajectory:
    answers = answers or ["June 5, 2004"]
    first_items = (
        []
        if empty_first_observation
        else [{"title": "Ronald Reagan", "content": first_observation}]
    )
    current_items = (
        []
        if empty_current_observation
        else [{"title": "Ronald Reagan", "content": current_observation}]
    )
    events = [
        {
            "role": "assistant",
            "text": _tool_call(first_query),
            "tool_call": {"name": "search", "query": first_query},
        },
        {
            "role": "tool",
            "tool_name": "search",
            "ok": True,
            "items": first_items,
            "observation": first_observation if first_items else "",
        },
        {
            "role": "assistant",
            "text": _tool_call(second_query),
            "tool_call": {"name": "search", "query": second_query},
        },
        {
            "role": "tool",
            "tool_name": "search",
            "ok": True,
            "items": current_items,
            "observation": current_observation if current_items else "",
        },
    ]
    turns = [
        AssistantTurn([1, 2], [3], [0.1], _tool_call(first_query)),
        AssistantTurn([1, 2, 3, 4], [5], [0.2], _tool_call(second_query)),
    ]
    if include_final_answer_turn:
        events.append({"role": "assistant", "text": final_text, "parsed_kind": "answer"})
        turns.append(AssistantTurn([1, 2, 3, 4, 5], [6], [0.3], final_text))
    return Trajectory(
        example=SearchExample("q-final-hop", question, answers, "test"),
        group_index=0,
        messages=[],
        search_calls=2,
        final_text=final_text,
        reward=float(exact_match),
        advantage=advantage,
        valid_format=valid_format,
        exact_match=exact_match,
        stop_reason=stop_reason,
        events=events,
        turns=turns,
    )


def _penalty_candidate(final_text: str) -> Trajectory:
    return Trajectory(
        example=SearchExample("q5", "Question?", ["A"], "test"),
        group_index=0,
        messages=[],
        search_calls=2,
        final_text=final_text,
        stop_reason="max_search_calls",
        events=[
            {
                "role": "assistant",
                "text": "",
                "tool_call": {"name": "search", "query": "same"},
            },
            {"role": "tool", "tool_name": "search", "ok": True, "items": []},
            {
                "role": "assistant",
                "text": "",
                "tool_call": {"name": "search", "query": "same"},
            },
            {"role": "tool", "tool_name": "search", "ok": True, "items": []},
        ],
    )


def _helpful_followup_candidate(final_text: str, exact_answer: str) -> Trajectory:
    return Trajectory(
        example=SearchExample(
            "q-follow",
            "Who is the maternal grandfather of Claudia Antonia?",
            [exact_answer],
            "test",
        ),
        group_index=0,
        messages=[],
        search_calls=2,
        final_text=final_text,
        stop_reason="answer",
        events=[
            {
                "role": "assistant",
                "text": "",
                "tool_call": {"name": "search", "query": "Claudia Antonia mother"},
            },
            {
                "role": "tool",
                "tool_name": "search",
                "ok": True,
                "items": [{"title": "Evidence"}],
            },
            {
                "role": "assistant",
                "text": "",
                "tool_call": {"name": "search", "query": "Aelia Paetina father"},
            },
            {
                "role": "tool",
                "tool_name": "search",
                "ok": True,
                "items": [{"title": "Evidence"}],
            },
        ],
    )


@dataclass
class FakeSequence:
    text: str

    @property
    def tokens(self) -> list[int]:
        return [ord(char) for char in self.text]

    @property
    def logprobs(self) -> list[float]:
        return [-0.5] * len(self.text)


@dataclass
class FakeResponse:
    sequences: list[FakeSequence]


class FakeSamplingClient:
    def __init__(self, responses: list[str | list[str]]) -> None:
        self._responses = list(responses)

    async def sample_async(self, **kwargs: object) -> FakeResponse:
        if not self._responses:
            raise AssertionError("fake sampler exhausted")
        response = self._responses.pop(0)
        num_samples = int(kwargs["num_samples"])
        texts = response if isinstance(response, list) else [response]
        if len(texts) != num_samples:
            raise AssertionError("fake sample count does not match num_samples")
        return FakeResponse([FakeSequence(text) for text in texts])


class FakeFuture:
    def __init__(self, value: list[float | None]) -> None:
        self._value = value

    def result(self) -> list[float | None]:
        return self._value


class FakeReferenceClient:
    def __init__(self) -> None:
        self.requests: list[list[int]] = []

    def compute_logprobs(self, prompt: object) -> FakeFuture:
        tokens = [int(token) for token in prompt.tolist()]
        self.requests.append(tokens)
        return FakeFuture([None, *[-float(token) / 100.0 for token in tokens[1:]]])


class FakeTokenizer:
    eos_token = None

    def apply_chat_template(
        self,
        messages: list[dict],
        *,
        tools: list[dict],
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> list[int]:
        del tools, tokenize, enable_thinking
        text = ""
        for message in messages:
            role = message["role"]
            if role == "assistant":
                text += "assistant:" + message.get("content", "") + "\n"
            elif role == "tool":
                text += "tool:" + message.get("content", "") + "\n"
            else:
                text += role + ":" + message.get("content", "") + "\n"
        if add_generation_prompt:
            text += "assistant:"
        return self.encode(text, add_special_tokens=False)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(char) for char in text]

    def decode(self, tokens: list[int], skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return "".join(chr(token) for token in tokens)


class BrokenAssistantBoundaryTokenizer(FakeTokenizer):
    def apply_chat_template(
        self,
        messages: list[dict],
        *,
        tools: list[dict],
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> list[int]:
        del tools, tokenize, enable_thinking
        text = ""
        for message in messages:
            role = message["role"]
            if role == "assistant":
                text += "assistant:[rewritten]\n"
            elif role == "tool":
                text += "tool:" + message.get("content", "") + "\n"
            else:
                text += role + ":" + message.get("content", "") + "\n"
        if add_generation_prompt:
            text += "assistant:"
        return self.encode(text, add_special_tokens=False)


if __name__ == "__main__":
    unittest.main()
