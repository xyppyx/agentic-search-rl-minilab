"""Tests for prompt and tool-observation protocol constraints."""

from __future__ import annotations

import unittest

from search_r1_minilab.protocol import (
    TOOL_OBSERVATION_REMINDER,
    build_prompt,
    initial_messages,
    token_count,
    tool_message,
    tool_message_content,
)
from search_r1_minilab.rollout import RolloutConfig, fit_tool_content
from search_r1_minilab.tools.base import SearchResult


class ProtocolPromptConstraintTest(unittest.TestCase):
    def test_initial_prompt_includes_multihop_and_short_answer_constraints(self) -> None:
        system_prompt = initial_messages("Who wrote it?")[0]["content"]

        self.assertIn("bridge entity", system_prompt)
        self.assertIn("Do not stop after a search result", system_prompt)
        self.assertIn("output exactly one line", system_prompt)
        self.assertIn("shortest single answer span", system_prompt)
        self.assertIn("Do not include reasoning", system_prompt)

    def test_tool_message_appends_followup_reminder(self) -> None:
        message = tool_message("search-1", "Evidence about a bridge entity.")

        self.assertEqual(message["role"], "tool")
        self.assertIn("Evidence about a bridge entity.", message["content"])
        self.assertIn(TOOL_OBSERVATION_REMINDER, message["content"])

    def test_training_tool_budget_counts_prompt_reminder(self) -> None:
        tokenizer = FakeTokenizer()
        messages = initial_messages("Question?")
        assistant_text = (
            "<tool_call><function=search><parameter=query>"
            "query"
            "</parameter></function></tool_call>"
        )
        result = SearchResult(ok=True, items=[], latency=0.0, backend="test")
        empty_observation = "Search returned no results."
        reminder_budget = token_count(
            tokenizer,
            tool_message_content(empty_observation),
        )

        fitted = fit_tool_content(
            tokenizer,
            messages,
            assistant_text,
            build_prompt(tokenizer, messages),
            tokenizer.encode(assistant_text, add_special_tokens=False),
            "search-1",
            result,
            RolloutConfig(
                max_tool_response_tokens=reminder_budget - 1,
                max_trajectory_tokens=100_000,
            ),
        )

        self.assertIsNone(fitted)


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
            content = message.get("content", "")
            if role == "assistant":
                text += f"assistant:{content}\n"
            elif role == "tool":
                text += f"tool:{content}\n"
            else:
                text += f"{role}:{content}\n"
        if add_generation_prompt:
            text += "assistant:"
        return self.encode(text, add_special_tokens=False)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(char) for char in text]


if __name__ == "__main__":
    unittest.main()
