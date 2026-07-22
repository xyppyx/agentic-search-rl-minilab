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
