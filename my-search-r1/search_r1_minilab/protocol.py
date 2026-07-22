"""Search-R1 chat protocol helpers used by MiniLab rollout smoke runs."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


MODEL_TOOL_NAME = "search"

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": MODEL_TOOL_NAME,
        "description": "Search for evidence. Use a concise English query.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "A concise English search query."}
            },
            "required": ["query"],
        },
    },
}

SYSTEM_PROMPT = """You answer factual questions with help from a search tool.
Search before giving the final answer. Use concise English queries.
Do not answer from memory before seeing at least one search result.
For multi-hop or relation questions, first identify the bridge entity, then search that entity or relation before answering.
Call search exactly once per assistant turn. Wait for the tool result before making another search call.
Do not stop after a search result that only identifies an intermediate person, work, place, date, role, or organization.
When ready, output exactly one line and nothing else:
Answer: <shortest single answer span>
Do not include reasoning, markdown, citations, parentheses, alternatives, or words such as "or" after Answer:.
Do not call a tool and give the final answer in the same turn."""

TOOL_OBSERVATION_REMINDER = (
    "Reminder: if the result only identifies a bridge entity, search that entity "
    "or relation before answering. Final output must be exactly one line: "
    "Answer: <shortest single answer span>."
)

TOOL_CALL_PATTERN = re.compile(
    r"<tool_call>\s*<function=search>\s*<parameter=query>\s*(.*?)\s*"
    r"</parameter>\s*</function>\s*</tool_call>",
    re.DOTALL,
)


@dataclass(frozen=True)
class ParsedAssistant:
    """One parsed assistant response."""

    kind: str
    content: str
    query: str | None = None


def initial_messages(question: str) -> list[dict[str, Any]]:
    """Create the initial system/user messages for one question."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


def build_prompt(tokenizer: Any, messages: list[dict[str, Any]]) -> list[int]:
    """Render messages with the model chat template and search tool definition."""
    return _render_chat(tokenizer, messages, add_generation_prompt=True)


def _render_chat(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    *,
    add_generation_prompt: bool,
) -> list[int]:
    """Render messages and normalize tokenizer outputs to a flat token list."""
    rendered = tokenizer.apply_chat_template(
        messages,
        tools=[SEARCH_TOOL],
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=False,
    )
    if isinstance(rendered, Mapping):
        rendered = rendered["input_ids"]
    if hasattr(rendered, "tolist"):
        rendered = rendered.tolist()
    if rendered and isinstance(rendered[0], list):
        rendered = rendered[0]
    return [int(token) for token in rendered]


def _encoded_text_tokens(tokenizer: Any, text: str) -> list[int]:
    """Encode plain text and normalize tokenizer outputs to a flat token list."""
    encoded = tokenizer.encode(text, add_special_tokens=False)
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    return [int(token) for token in encoded]


def _suffix_prefix_overlap(tokens: list[int], suffix: list[int]) -> int:
    """Return the longest overlap between tokens suffix and suffix prefix."""
    for length in range(min(len(tokens), len(suffix)), 0, -1):
        if tokens[-length:] == suffix[:length]:
            return length
    return 0


def build_next_prompt(
    tokenizer: Any,
    messages_before_assistant: list[dict[str, Any]],
    assistant_text: str,
    previous_prompt_tokens: list[int],
    completion_tokens: list[int],
    next_tool_message: dict[str, Any],
) -> list[int]:
    """Append assistant output and a tool observation without re-tokenizing sampled tokens."""
    canonical_prompt = build_prompt(tokenizer, messages_before_assistant)
    assistant_message = {"role": "assistant", "content": assistant_text}
    messages_with_assistant = [*messages_before_assistant, assistant_message]
    canonical_assistant_end = _render_chat(
        tokenizer,
        messages_with_assistant,
        add_generation_prompt=False,
    )
    canonical_text_tokens = _encoded_text_tokens(tokenizer, assistant_text)
    canonical_action = [*canonical_prompt, *canonical_text_tokens]
    if canonical_assistant_end[: len(canonical_action)] != canonical_action:
        raise ValueError("chat template cannot recover assistant message boundary")

    assistant_closing_tokens = canonical_assistant_end[len(canonical_action) :]
    canonical_next_prompt = build_prompt(
        tokenizer,
        [*messages_with_assistant, next_tool_message],
    )
    if canonical_next_prompt[: len(canonical_assistant_end)] != canonical_assistant_end:
        raise ValueError("chat template rewrote history after tool observation")

    observation_tokens = canonical_next_prompt[len(canonical_assistant_end) :]
    overlap = _suffix_prefix_overlap(completion_tokens, assistant_closing_tokens)
    return [
        *previous_prompt_tokens,
        *completion_tokens,
        *assistant_closing_tokens[overlap:],
        *observation_tokens,
    ]


def parse_assistant(text: str) -> ParsedAssistant:
    """Classify an assistant response as a tool call, final answer, or invalid output."""
    matches = list(TOOL_CALL_PATTERN.finditer(text))
    if not matches:
        kind = "invalid" if "<tool_call>" in text else "answer"
        return ParsedAssistant(kind=kind, content=text.strip())
    if len(matches) != 1 or text[matches[0].end() :].strip():
        return ParsedAssistant(kind="invalid", content=text.strip())
    query = matches[0].group(1).strip()
    if not query or "<" in query or ">" in query:
        return ParsedAssistant(kind="invalid", content=text.strip())
    content = text[: matches[0].start()].strip()
    return ParsedAssistant(kind="tool", content=content, query=query)


def tool_message_content(content: str) -> str:
    """Append rollout guidance to one tool observation."""
    return f"{content}\n\n{TOOL_OBSERVATION_REMINDER}"


def tool_message(call_id: str, content: str) -> dict[str, Any]:
    """Build a chat message containing a tool observation."""
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "name": MODEL_TOOL_NAME,
        "content": tool_message_content(content),
    }


def stop_sequences(tokenizer: Any) -> list[str]:
    """Return stop strings for ending one assistant turn."""
    eos_token = getattr(tokenizer, "eos_token", None)
    return [eos_token] if eos_token else []


def token_count(tokenizer: Any, text: str) -> int:
    """Count tokenizer tokens in plain text."""
    return len(_encoded_text_tokens(tokenizer, text))
