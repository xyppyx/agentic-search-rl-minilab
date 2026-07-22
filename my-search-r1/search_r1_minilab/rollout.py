"""Training-grade Search-R1 rollout with pluggable search backends."""

from __future__ import annotations

import asyncio
import copy
from dataclasses import dataclass, field
from typing import Any, Callable

import pytrio as trio

from search_r1_minilab.data import SearchExample
from search_r1_minilab.diagnostics import diagnose_fields
from search_r1_minilab.protocol import (
    MODEL_TOOL_NAME,
    build_next_prompt,
    build_prompt,
    initial_messages,
    parse_assistant,
    stop_sequences,
    tool_message,
    tool_message_content,
    token_count,
)
from search_r1_minilab.rewards import (
    RewardShapingConfig,
    apply_reward_shaping,
    score_answer,
)
from search_r1_minilab.tools.base import SearchResult, format_item
from search_r1_minilab.tools.registry import ToolRegistry


@dataclass(frozen=True)
class RolloutConfig:
    """Sampling and trajectory limits."""

    group_size: int = 8
    max_search_calls: int = 4
    max_assistant_turns: int = 6
    max_trajectory_tokens: int = 8192
    max_assistant_tokens: int = 1024
    max_tool_response_tokens: int = 1024
    temperature: float = 1.0
    top_p: float = 1.0
    seed: int = 42
    reward_shaping: RewardShapingConfig = field(default_factory=RewardShapingConfig)


@dataclass
class AssistantTurn:
    """One trainable assistant generation."""

    prompt_tokens: list[int]
    completion_tokens: list[int]
    logprobs: list[float]
    text: str


@dataclass
class Trajectory:
    """One complete multi-turn rollout and its training signal."""

    example: SearchExample
    group_index: int
    messages: list[dict[str, Any]]
    next_prompt_tokens: list[int] | None = None
    question_index: int = 0
    turns: list[AssistantTurn] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    search_calls: int = 0
    final_text: str = ""
    reward: float = -0.1
    advantage: float = 0.0
    valid_format: bool = False
    exact_match: bool = False
    reward_components: dict[str, float] = field(default_factory=dict)
    done: bool = False
    stop_reason: str = "max_assistant_turns"


@dataclass(frozen=True)
class SampleRequest:
    """One PyTRIO sampling request for a trajectory state."""

    trajectory_index: int
    prompt_tokens: list[int]
    num_samples: int
    max_tokens: int
    seed: int


async def sample_requests_async(
    sampling_client: Any,
    requests: list[SampleRequest],
    config: RolloutConfig,
    tokenizer: Any,
) -> list[Any]:
    """Run PyTRIO sample_async requests concurrently."""
    tasks = []
    for request in requests:
        params = trio.SamplingParams(
            max_tokens=request.max_tokens,
            seed=request.seed,
            stop=stop_sequences(tokenizer),
            temperature=config.temperature,
            top_p=config.top_p,
        )
        tasks.append(
            sampling_client.sample_async(
                prompt=trio.ModelInput.from_ints(request.prompt_tokens),
                num_samples=request.num_samples,
                sampling_params=params,
                return_text=True,
            )
        )
    return list(await asyncio.gather(*tasks))


def fit_tool_content(
    tokenizer: Any,
    messages_before_assistant: list[dict[str, Any]],
    assistant_text: str,
    previous_prompt_tokens: list[int],
    completion_tokens: list[int],
    call_id: str,
    result: SearchResult,
    config: RolloutConfig,
) -> tuple[str, list[int]] | None:
    """Fit complete search-result items inside tool and trajectory token budgets."""
    if not result.ok:
        candidates = [f"Search error: {result.error_type or result.error or 'unknown error'}"]
    elif not result.items:
        candidates = ["Search returned no results."]
    else:
        candidates = [format_item(item, index) for index, item in enumerate(result.items, 1)]

    accepted: list[str] = []
    accepted_prompt: list[int] | None = None
    for candidate in candidates:
        content = "\n\n".join([*accepted, candidate])
        tool_content = tool_message_content(content)
        if token_count(tokenizer, tool_content) > config.max_tool_response_tokens:
            break
        next_tool_message = tool_message(call_id, content)
        next_prompt = build_next_prompt(
            tokenizer,
            messages_before_assistant,
            assistant_text,
            previous_prompt_tokens,
            completion_tokens,
            next_tool_message,
        )
        if len(next_prompt) > config.max_trajectory_tokens:
            break
        accepted.append(candidate)
        accepted_prompt = next_prompt
    if not accepted or accepted_prompt is None:
        return None
    return "\n\n".join(accepted), accepted_prompt


def make_request(
    tokenizer: Any,
    trajectory: Trajectory,
    trajectory_index: int,
    num_samples: int,
    seed: int,
    config: RolloutConfig,
) -> SampleRequest | None:
    """Create a sampling request from the current trajectory state."""
    prompt_tokens = (
        trajectory.next_prompt_tokens
        if trajectory.next_prompt_tokens is not None
        else build_prompt(tokenizer, trajectory.messages)
    )
    max_tokens = min(
        config.max_assistant_tokens,
        config.max_trajectory_tokens - len(prompt_tokens),
    )
    if max_tokens <= 0:
        trajectory.done = True
        trajectory.stop_reason = "max_trajectory_tokens"
        return None
    return SampleRequest(trajectory_index, prompt_tokens, num_samples, max_tokens, seed)


def read_sequence(sequence: Any, tokenizer: Any) -> tuple[list[int], list[float], str]:
    """Read sampled tokens, old logprobs, and text from one PyTRIO sequence."""
    tokens = [int(token) for token in sequence.tokens]
    logprobs = [float(value) for value in sequence.logprobs]
    if len(tokens) != len(logprobs):
        raise ValueError("sample token and logprob lengths differ")
    text = sequence.text
    if text is None:
        text = tokenizer.decode(tokens, skip_special_tokens=True)
    return tokens, logprobs, str(text)


def advance_trajectory(
    trajectory: Trajectory,
    prompt_tokens: list[int],
    sequence: Any,
    tokenizer: Any,
    registry: ToolRegistry,
    backend_name: str,
    config: RolloutConfig,
) -> None:
    """Consume one assistant output and either call search or finish the trajectory."""
    tokens, logprobs, text = read_sequence(sequence, tokenizer)
    trajectory.turns.append(AssistantTurn(prompt_tokens, tokens, logprobs, text))
    parsed = parse_assistant(text)
    assistant_event: dict[str, Any] = {
        "role": "assistant",
        "text": text,
        "parsed_kind": parsed.kind,
    }
    if parsed.kind == "tool":
        assistant_event["tool_call"] = {
            "name": MODEL_TOOL_NAME,
            "query": parsed.query or "",
        }
    trajectory.events.append(assistant_event)

    can_search = (
        parsed.kind == "tool"
        and trajectory.search_calls < config.max_search_calls
        and len(trajectory.turns) < config.max_assistant_turns
    )
    if not can_search:
        trajectory.messages.append({"role": "assistant", "content": text})
        trajectory.final_text = text
        trajectory.done = True
        if parsed.kind == "answer":
            trajectory.stop_reason = "answer"
        elif parsed.kind == "tool" and trajectory.search_calls >= config.max_search_calls:
            trajectory.stop_reason = "max_search_calls"
        elif parsed.kind == "tool":
            trajectory.stop_reason = "max_assistant_turns"
        else:
            trajectory.stop_reason = "invalid_format"
        return

    call_id = (
        f"search-{trajectory.question_index}-{trajectory.group_index}-"
        f"{trajectory.search_calls + 1}"
    )
    messages_before_assistant = list(trajectory.messages)
    trajectory.messages.append({"role": "assistant", "content": text})
    result = registry.call(backend_name, {"query": parsed.query or ""})
    trajectory.search_calls += 1
    prompt_reconstruction_failed = False
    try:
        fitted = fit_tool_content(
            tokenizer,
            messages_before_assistant,
            text,
            prompt_tokens,
            tokens,
            call_id,
            result,
            config,
        )
    except ValueError:
        fitted = None
        prompt_reconstruction_failed = True
    observation = fitted[0] if fitted is not None else ""
    trajectory.events.append(_tool_event(result, observation))
    if fitted is None:
        trajectory.final_text = text
        trajectory.done = True
        trajectory.stop_reason = (
            "prompt_reconstruction_failed"
            if prompt_reconstruction_failed
            else "tool_observation_budget"
        )
        return

    content, next_prompt_tokens = fitted
    trajectory.messages.append(tool_message(call_id, content))
    trajectory.next_prompt_tokens = next_prompt_tokens


def score_trajectory(
    trajectory: Trajectory,
    tokenizer: Any,
    reward_shaping: RewardShapingConfig,
) -> None:
    """Score a completed trajectory with answer format and exact match reward."""
    result = score_answer(trajectory.final_text, trajectory.example.answers)
    diagnostics = diagnose_fields(
        turns=trajectory.events,
        search_calls=trajectory.search_calls,
        exact_match=result.exact_match,
        valid_format=result.valid_format,
        stop_reason=trajectory.stop_reason,
        question=trajectory.example.question,
    )
    answer_token_count = (
        token_count(tokenizer, result.answer) if result.answer is not None else 0
    )
    components = apply_reward_shaping(
        result,
        diagnostics,
        reward_shaping,
        answer_token_count=answer_token_count,
        references=trajectory.example.answers,
    )
    trajectory.reward = components.final_reward
    trajectory.valid_format = result.valid_format
    trajectory.exact_match = result.exact_match
    trajectory.reward_components = components.to_dict()


def assign_group_advantages(trajectories: list[Trajectory]) -> int:
    """Assign centered reward advantages within each question group."""
    groups: dict[int, list[Trajectory]] = {}
    for trajectory in trajectories:
        groups.setdefault(trajectory.question_index, []).append(trajectory)
    degenerate = 0
    for group in groups.values():
        mean_reward = sum(item.reward for item in group) / len(group)
        for item in group:
            item.advantage = item.reward - mean_reward
        if all(item.advantage == 0.0 for item in group):
            degenerate += 1
    return degenerate


def rollout_batch(
    sampling_client: Any,
    tokenizer: Any,
    registry: ToolRegistry,
    backend_name: str,
    examples: list[SearchExample],
    config: RolloutConfig,
    progress_callback: Callable[[int], None] | None = None,
) -> list[Trajectory]:
    """Run grouped multi-turn rollouts and return scored trajectories."""
    roots = [
        Trajectory(
            example=example,
            group_index=0,
            messages=initial_messages(example.question),
            question_index=question_index,
        )
        for question_index, example in enumerate(examples)
    ]

    first_requests: list[SampleRequest] = []
    for index, trajectory in enumerate(roots):
        request = make_request(
            tokenizer,
            trajectory,
            index,
            config.group_size,
            config.seed + index,
            config,
        )
        if request:
            first_requests.append(request)

    trajectories: list[Trajectory] = []
    if first_requests:
        responses = asyncio.run(
            sample_requests_async(sampling_client, first_requests, config, tokenizer)
        )
        for request, response in zip(first_requests, responses, strict=True):
            root = roots[request.trajectory_index]
            if len(response.sequences) != config.group_size:
                raise ValueError("first-turn sample count differs from group_size")
            for group_index, sequence in enumerate(response.sequences):
                branch = copy.deepcopy(root)
                branch.group_index = group_index
                advance_trajectory(
                    branch,
                    request.prompt_tokens,
                    sequence,
                    tokenizer,
                    registry,
                    backend_name,
                    config,
                )
                trajectories.append(branch)
                if branch.done and progress_callback is not None:
                    progress_callback(1)

    while any(not trajectory.done for trajectory in trajectories):
        requests: list[SampleRequest] = []
        for index, trajectory in enumerate(trajectories):
            if trajectory.done:
                continue
            request = make_request(
                tokenizer,
                trajectory,
                index,
                1,
                config.seed + index + len(trajectory.turns) * 10_000,
                config,
            )
            if request:
                requests.append(request)
            elif trajectory.done and progress_callback is not None:
                progress_callback(1)
        if not requests:
            break
        responses = asyncio.run(sample_requests_async(sampling_client, requests, config, tokenizer))
        for request, response in zip(requests, responses, strict=True):
            if len(response.sequences) != 1:
                raise ValueError("later rollout turns must return exactly one sequence")
            trajectory = trajectories[request.trajectory_index]
            advance_trajectory(
                trajectory,
                request.prompt_tokens,
                response.sequences[0],
                tokenizer,
                registry,
                backend_name,
                config,
            )
            if trajectory.done and progress_callback is not None:
                progress_callback(1)

    for trajectory in trajectories:
        score_trajectory(trajectory, tokenizer, config.reward_shaping)
    assign_group_advantages(trajectories)
    return trajectories


def trajectory_to_record(trajectory: Trajectory, *, run_type: str = "rollout") -> dict[str, Any]:
    """Convert a training-grade trajectory to the shared JSONL/report schema."""
    tool_failures = sum(
        event.get("role") == "tool" and event.get("ok") is False
        for event in trajectory.events
    )
    return {
        "question": trajectory.example.question,
        "answers": trajectory.example.answers,
        "data_source": trajectory.example.data_source,
        "turns": trajectory.events,
        "reward": trajectory.reward,
        "advantage": trajectory.advantage,
        "valid_format": trajectory.valid_format,
        "exact_match": trajectory.exact_match,
        "search_calls": trajectory.search_calls,
        "tool_failures": int(tool_failures),
        "metadata": {
            "id": trajectory.example.id,
            "run_type": run_type,
            "stop_reason": trajectory.stop_reason,
            "model_tool_name": MODEL_TOOL_NAME,
            "question_index": trajectory.question_index,
            "group_index": trajectory.group_index,
            "assistant_turns": len(trajectory.turns),
            "reward_components": trajectory.reward_components,
        },
    }


def _tool_event(result: SearchResult, observation: str) -> dict[str, Any]:
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
