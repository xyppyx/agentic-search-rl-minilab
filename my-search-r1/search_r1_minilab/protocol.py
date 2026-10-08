"""Search-R1 对话协议辅助函数。

本文件只处理“模型应该按什么格式和搜索工具交互”：system prompt、tool
schema、assistant 输出解析、tool observation 拼接和 tokenizer chat template
渲染。训练、reward 和搜索 backend 都不放在这里。
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


MODEL_TOOL_NAME = "search"

# 暴露给 tokenizer chat template 的工具定义；模型输出必须匹配
# TOOL_CALL_PATTERN，rollout 才会真正调用后端搜索。
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

# 这是 prompt-only search budget guard 的核心约束：要求先搜、桥接实体后继续搜、
# 最终答案只输出一行短 span。这里影响 rollout 行为，但不直接参与 loss。
SYSTEM_PROMPT = """You answer factual questions with help from a search tool.
Search before giving the final answer. Use concise English queries.
Do not answer from memory before seeing at least one search result.
For multi-hop or relation questions, first identify the bridge entity, then search that entity or relation before answering.
Call search exactly once per assistant turn. Wait for the tool result before making another search call.
Do not stop after a search result that only identifies an intermediate person, work, place, date, role, or organization.
Use at most three searches when possible. After three searches, answer with the best supported short span instead of asking for another search.
When ready, output exactly one line and nothing else:
Answer: <shortest single answer span>
Do not include reasoning, markdown, citations, parentheses, alternatives, or words such as "or" after Answer:.
Do not call a tool and give the final answer in the same turn."""

TOOL_OBSERVATION_REMINDER = (
    "Reminder: if the result only identifies a bridge entity, search that entity "
    "or relation before answering. Final output must be exactly one line: "
    "Answer: <shortest single answer span>. If you have searched three times, "
    "answer with the best supported span instead of searching again."
)

# 训练 rollout 使用 XML-like tool call 文本协议，而不是直接依赖平台级 function
# calling；这样 sampled text、token 和 logprob 能完整进入 trajectory。
TOOL_CALL_PATTERN = re.compile(
    r"<tool_call>\s*<function=search>\s*<parameter=query>\s*(.*?)\s*"
    r"</parameter>\s*</function>\s*</tool_call>",
    re.DOTALL,
)


@dataclass(frozen=True)
class ParsedAssistant:
    """一次 assistant 输出的解析结果。"""

    kind: str
    content: str
    query: str | None = None


def initial_messages(question: str) -> list[dict[str, Any]]:
    """为单个问题构造初始 system/user messages。"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


def teacher_messages_with_skill(
    messages: list[dict[str, Any]], skill_context: str
) -> list[dict[str, Any]]:
    """Add a teacher-only rule to the system message without changing rollout history."""
    if not skill_context.strip():
        raise ValueError("teacher skill must be nonempty")
    if len(messages) < 2 or messages[0].get("role") != "system" or messages[1].get("role") != "user":
        raise ValueError("teacher history must start with system and user messages")
    return [
        {
            **messages[0],
            "content": (
                f"{messages[0]['content']}\n\n"
                f"Teacher-only search skill:\n{skill_context.strip()}"
            ),
        },
        *messages[1:],
    ]


def build_prompt(tokenizer: Any, messages: list[dict[str, Any]]) -> list[int]:
    """用模型 chat template 渲染 messages，并注入 search tool 定义。"""
    return _render_chat(tokenizer, messages, add_generation_prompt=True)


def _render_chat(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    *,
    add_generation_prompt: bool,
) -> list[int]:
    """渲染 chat template，并把不同 tokenizer 返回格式统一为一维 token 列表。"""
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
    """编码普通文本，并统一为一维 token 列表。"""
    encoded = tokenizer.encode(text, add_special_tokens=False)
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    return [int(token) for token in encoded]


def _suffix_prefix_overlap(tokens: list[int], suffix: list[int]) -> int:
    """返回 tokens 后缀与 suffix 前缀的最长重叠长度。"""
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
    """在不重 tokenize 采样 completion 的前提下，追加 tool observation prompt。"""
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
    # 先用标准 chat template 验证 assistant 边界，防止手工拼接后的 token
    # 与模型实际上下文不一致，进而破坏后续 logprob/target 对齐。
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
    # sampled completion 可能已经包含一部分 assistant closing token；用 overlap
    # 去重，避免把同一段 template token 写入两次。
    overlap = _suffix_prefix_overlap(completion_tokens, assistant_closing_tokens)
    return [
        *previous_prompt_tokens,
        *completion_tokens,
        *assistant_closing_tokens[overlap:],
        *observation_tokens,
    ]


def parse_assistant(text: str) -> ParsedAssistant:
    """把 assistant 文本分类为 tool call、final answer 或格式错误。"""
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
    """给 tool observation 追加 follow-up/最终答案格式提醒。"""
    return f"{content}\n\n{TOOL_OBSERVATION_REMINDER}"


def tool_message(call_id: str, content: str) -> dict[str, Any]:
    """构造一条 role=tool 的 chat message。"""
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "name": MODEL_TOOL_NAME,
        "content": tool_message_content(content),
    }


def stop_sequences(tokenizer: Any) -> list[str]:
    """返回单轮 assistant 生成的 stop strings。"""
    eos_token = getattr(tokenizer, "eos_token", None)
    return [eos_token] if eos_token else []


def token_count(tokenizer: Any, text: str) -> int:
    """计算普通文本 token 数，用于工具响应预算控制。"""
    return len(_encoded_text_tokens(tokenizer, text))
