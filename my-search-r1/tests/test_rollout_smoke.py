"""Tests for the MiniLab rollout smoke/eval loop."""

from __future__ import annotations

import unittest
from dataclasses import dataclass
from pathlib import Path

from search_r1_minilab.rollout_smoke import (
    RolloutSmokeConfig,
    SearchExample,
    rollout_one,
)
from search_r1_minilab.tools import LocalBM25Backend, ToolRegistry
from search_r1_minilab.trajectories import build_markdown_report


FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


class RolloutSmokeTest(unittest.TestCase):
    def test_two_turn_tool_rollout_scores_exact_match(self) -> None:
        registry = ToolRegistry()
        registry.register(LocalBM25Backend.from_jsonl(FIXTURES_DIR / "bm25_corpus.jsonl"))
        sampler = FakeSamplingClient(
            [
                _tool_call("little prince"),
                "Answer: Antoine de Saint-Exupery",
            ]
        )

        record = rollout_one(
            sampler,
            FakeTokenizer(),
            registry,
            "local_bm25",
            SearchExample(
                id="q1",
                question="Who wrote The Little Prince?",
                answers=["Antoine de Saint-Exupery"],
                data_source="test",
            ),
            RolloutSmokeConfig(),
        )

        self.assertEqual(record["reward"], 1.0)
        self.assertTrue(record["exact_match"])
        self.assertTrue(record["valid_format"])
        self.assertEqual(record["search_calls"], 1)
        self.assertEqual(record["tool_failures"], 0)
        self.assertEqual(record["turns"][0]["tool_call"]["name"], "search")
        self.assertEqual(record["turns"][1]["backend"], "local_bm25")
        self.assertGreaterEqual(len(record["turns"][1]["items"]), 1)

    def test_invalid_final_answer_is_reported(self) -> None:
        registry = ToolRegistry()
        registry.register(LocalBM25Backend.from_jsonl(FIXTURES_DIR / "bm25_corpus.jsonl"))

        record = rollout_one(
            FakeSamplingClient(["I think it was a French writer."]),
            FakeTokenizer(),
            registry,
            "local_bm25",
            SearchExample(
                id="q2",
                question="Who wrote The Little Prince?",
                answers=["Antoine de Saint-Exupery"],
                data_source="test",
            ),
            RolloutSmokeConfig(),
        )
        report = build_markdown_report([record], title="Invalid Smoke")

        self.assertEqual(record["reward"], -0.1)
        self.assertFalse(record["valid_format"])
        self.assertIn("## Invalid Format Cases", report)
        self.assertIn("I think it was a French writer.", report)

    def test_tool_failure_is_reported(self) -> None:
        registry = ToolRegistry()
        registry.register(LocalBM25Backend.from_jsonl(FIXTURES_DIR / "bm25_corpus.jsonl"))

        record = rollout_one(
            FakeSamplingClient([_tool_call("???"), "Answer: unknown"]),
            FakeTokenizer(),
            registry,
            "local_bm25",
            SearchExample(
                id="q3",
                question="Who wrote The Little Prince?",
                answers=["Antoine de Saint-Exupery"],
                data_source="test",
            ),
            RolloutSmokeConfig(),
        )
        report = build_markdown_report([record], title="Failure Smoke")

        self.assertEqual(record["search_calls"], 1)
        self.assertEqual(record["tool_failures"], 1)
        self.assertEqual(record["turns"][1]["error_type"], "empty_query")
        self.assertIn("## Tool Failure Cases", report)
        self.assertIn("Tool failures: `1`", report)


def _tool_call(query: str) -> str:
    return (
        "<tool_call><function=search><parameter=query>"
        f"{query}"
        "</parameter></function></tool_call>"
    )


@dataclass
class FakeSequence:
    text: str

    @property
    def tokens(self) -> list[int]:
        return [ord(char) for char in self.text]


@dataclass
class FakeResponse:
    sequences: list[FakeSequence]


class FakeSamplingClient:
    def __init__(self, texts: list[str]) -> None:
        self._texts = list(texts)

    async def sample_async(self, **_: object) -> FakeResponse:
        if not self._texts:
            raise AssertionError("fake sampler exhausted")
        return FakeResponse([FakeSequence(self._texts.pop(0))])


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
        del tools, tokenize, add_generation_prompt, enable_thinking
        text = "\n".join(f"{message['role']}:{message.get('content', '')}" for message in messages)
        return self.encode(text, add_special_tokens=False)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(char) for char in text]

    def decode(self, tokens: list[int], skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return "".join(chr(token) for token in tokens)


if __name__ == "__main__":
    unittest.main()
