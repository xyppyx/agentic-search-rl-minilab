"""Minimal PyTRIO-backed rollout/eval smoke loop."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import pytrio as trio

from search_r1_minilab.data import SearchExample, load_examples
from search_r1_minilab.protocol import (
    MODEL_TOOL_NAME,
    build_prompt,
    initial_messages,
    parse_assistant,
    stop_sequences,
    token_count,
    tool_message,
)
from search_r1_minilab.rewards import score_answer
from search_r1_minilab.tools.base import SearchResult, format_item
from search_r1_minilab.tools.registry import ToolRegistry


@dataclass(frozen=True)
class RolloutSmokeConfig:
    """Sampling and rollout limits for smoke/eval."""

    max_search_calls: int = 2
    max_assistant_turns: int = 4
    max_trajectory_tokens: int = 4096
    max_assistant_tokens: int = 512
    max_tool_response_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int = 42


def rollout_examples(
    sampling_client: Any,
    tokenizer: Any,
    registry: ToolRegistry,
    backend_name: str,
    examples: list[SearchExample],
    config: RolloutSmokeConfig,
    *,
    start_index: int = 0,
) -> list[dict[str, Any]]:
    """Run one eval trajectory per example and return trajectory-compatible records."""
    records: list[dict[str, Any]] = []
    for index, example in enumerate(examples):
        records.append(
            rollout_one(
                sampling_client,
                tokenizer,
                registry,
                backend_name,
                example,
                config,
                question_index=start_index + index,
            )
        )
    return records


def rollout_one(
    sampling_client: Any,
    tokenizer: Any,
    registry: ToolRegistry,
    backend_name: str,
    example: SearchExample,
    config: RolloutSmokeConfig,
    *,
    question_index: int = 0,
) -> dict[str, Any]:
    """Run a bounded Search-R1 style interaction for one question."""
    messages = initial_messages(example.question)
    turns: list[dict[str, Any]] = []
    final_text = ""
    tool_call_attempts = 0
    executed_tool_calls = 0
    tool_failures = 0
    stop_reason = "max_assistant_turns"

    for assistant_turn in range(config.max_assistant_turns):
        prompt_tokens = build_prompt(tokenizer, messages)
        max_tokens = min(
            config.max_assistant_tokens,
            config.max_trajectory_tokens - len(prompt_tokens),
        )
        if max_tokens <= 0:
            stop_reason = "max_trajectory_tokens"
            break

        text = _sample_text(
            sampling_client,
            prompt_tokens,
            tokenizer,
            config,
            max_tokens=max_tokens,
            seed=config.seed + question_index * 10_000 + assistant_turn,
        )
        parsed = parse_assistant(text)
        assistant_record: dict[str, Any] = {
            "role": "assistant",
            "text": text,
            "parsed_kind": parsed.kind,
        }
        messages.append({"role": "assistant", "content": text})

        if parsed.kind != "tool":
            turns.append(assistant_record)
            final_text = text
            stop_reason = "answer" if parsed.kind == "answer" else "invalid_format"
            break

        tool_call_attempts += 1
        query = parsed.query or ""
        assistant_record["tool_call"] = {"name": MODEL_TOOL_NAME, "query": query}
        turns.append(assistant_record)

        if executed_tool_calls >= config.max_search_calls:
            final_text = text
            stop_reason = "max_search_calls"
            break

        call_id = f"search-{question_index}-{executed_tool_calls + 1}"
        result = registry.call(backend_name, {"query": query})
        executed_tool_calls += 1
        if not result.ok:
            tool_failures += 1

        fitted = _fit_tool_observation(
            tokenizer,
            messages,
            call_id,
            result,
            config,
        )
        if fitted is None:
            final_text = text
            stop_reason = "tool_observation_budget"
            break

        observation = fitted
        messages.append(tool_message(call_id, observation))
        turns.append(_tool_turn_record(result, observation))
    else:
        if turns and turns[-1].get("role") == "assistant":
            final_text = str(turns[-1].get("text") or "")

    reward = score_answer(final_text, example.answers)
    return {
        "question": example.question,
        "answers": example.answers,
        "data_source": example.data_source,
        "turns": turns,
        "reward": reward.reward,
        "advantage": None,
        "valid_format": reward.valid_format,
        "exact_match": reward.exact_match,
        "search_calls": tool_call_attempts,
        "tool_failures": tool_failures,
        "metadata": {
            "id": example.id,
            "answer": reward.answer,
            "stop_reason": stop_reason,
            "model_tool_name": MODEL_TOOL_NAME,
            "dispatch_backend": backend_name,
            "executed_tool_calls": executed_tool_calls,
            "max_search_calls": config.max_search_calls,
            "max_assistant_turns": config.max_assistant_turns,
        },
    }


def build_metrics(
    records: list[dict[str, Any]],
    registry: ToolRegistry,
) -> dict[str, float]:
    """Build compact smoke/eval metrics."""
    total = max(len(records), 1)
    metrics = {
        "eval/examples": float(len(records)),
        "em": sum(float(record.get("exact_match") is True) for record in records) / total,
        "format/rate": sum(float(record.get("valid_format") is True) for record in records)
        / total,
        "rollout/search_calls": sum(
            float(record.get("search_calls") or 0) for record in records
        )
        / total,
        "rollout/tool_failure_rate": sum(
            float((record.get("tool_failures") or 0) > 0) for record in records
        )
        / total,
    }
    metrics.update(registry.metrics())
    return metrics


def _sample_text(
    sampling_client: Any,
    prompt_tokens: list[int],
    tokenizer: Any,
    config: RolloutSmokeConfig,
    *,
    max_tokens: int,
    seed: int,
) -> str:
    response = asyncio.run(
        _sample_text_async(
            sampling_client,
            prompt_tokens,
            tokenizer,
            config,
            max_tokens=max_tokens,
            seed=seed,
        )
    )
    return response


async def _sample_text_async(
    sampling_client: Any,
    prompt_tokens: list[int],
    tokenizer: Any,
    config: RolloutSmokeConfig,
    *,
    max_tokens: int,
    seed: int,
) -> str:
    params = trio.SamplingParams(
        max_tokens=max_tokens,
        seed=seed,
        stop=stop_sequences(tokenizer),
        temperature=config.temperature,
        top_p=config.top_p,
    )
    response = await sampling_client.sample_async(
        prompt=trio.ModelInput.from_ints(prompt_tokens),
        num_samples=1,
        sampling_params=params,
        return_text=True,
    )
    if len(response.sequences) != 1:
        raise ValueError("rollout smoke expects exactly one sampled sequence")
    sequence = response.sequences[0]
    text = getattr(sequence, "text", None)
    if text is not None:
        return str(text)
    tokens = [int(token) for token in getattr(sequence, "tokens")]
    return str(tokenizer.decode(tokens, skip_special_tokens=True))


def _fit_tool_observation(
    tokenizer: Any,
    messages_with_assistant: list[dict[str, Any]],
    call_id: str,
    result: SearchResult,
    config: RolloutSmokeConfig,
) -> str | None:
    candidates = _observation_candidates(result)
    accepted: list[str] = []
    accepted_content: str | None = None
    for candidate in candidates:
        content = "\n\n".join([*accepted, candidate])
        if token_count(tokenizer, content) > config.max_tool_response_tokens:
            break
        next_messages = [*messages_with_assistant, tool_message(call_id, content)]
        if len(build_prompt(tokenizer, next_messages)) > config.max_trajectory_tokens:
            break
        accepted.append(candidate)
        accepted_content = content
    return accepted_content


def _observation_candidates(result: SearchResult) -> list[str]:
    if not result.ok:
        return [f"Search error: {result.error_type or result.error or 'unknown error'}"]
    if not result.items:
        return ["Search returned no results."]
    return [format_item(item, index) for index, item in enumerate(result.items, start=1)]


def _tool_turn_record(result: SearchResult, observation: str) -> dict[str, Any]:
    return {
        "role": "tool",
        "tool_name": MODEL_TOOL_NAME,
        "backend": result.backend,
        "ok": result.ok,
        "items": [item.to_dict() for item in result.items],
        "error_type": result.error_type,
        "error": result.error,
        "observation": observation,
        "latency": result.latency,
        "status": result.status,
        "metadata": result.metadata,
    }
