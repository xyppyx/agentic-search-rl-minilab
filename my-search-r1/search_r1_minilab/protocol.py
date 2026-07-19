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
Search when you need evidence. You may call search several times with concise English queries.
Call search exactly once per assistant turn. Wait for the tool result before making another search call.
When ready, end with exactly one non-empty line in this format:
Answer: <your short answer>
Do not call a tool and give the final answer in the same turn."""

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
    rendered = tokenizer.apply_chat_template(
        messages,
        tools=[SEARCH_TOOL],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if isinstance(rendered, Mapping):
        rendered = rendered["input_ids"]
    if hasattr(rendered, "tolist"):
        rendered = rendered.tolist()
    if rendered and isinstance(rendered[0], list):
        rendered = rendered[0]
    return [int(token) for token in rendered]


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


def tool_message(call_id: str, content: str) -> dict[str, Any]:
    """Build a chat message containing a tool observation."""
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "name": MODEL_TOOL_NAME,
        "content": content,
    }


def stop_sequences(tokenizer: Any) -> list[str]:
    """Return stop strings for ending one assistant turn."""
    eos_token = getattr(tokenizer, "eos_token", None)
    return [eos_token] if eos_token else []


def token_count(tokenizer: Any, text: str) -> int:
    """Count tokenizer tokens in plain text."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if hasattr(tokens, "tolist"):
        tokens = tokens.tolist()
    if tokens and isinstance(tokens[0], list):
        tokens = tokens[0]
    return len(tokens)
