"""训练级 Search-R1 rollout 状态机。

本文件负责把一个问题展开成多轮 assistant/tool trajectory：采样模型输出、
解析搜索调用、调用 ToolRegistry、拼接 observation、最终打 reward 并计算组内
advantage。这里不做参数更新；更新逻辑在 training.py。
"""

from __future__ import annotations

import asyncio
import copy
import math
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
    """采样参数和 trajectory 预算。

    group_size 控制 GRPO 每个问题采样多少条候选轨迹；max_search_calls 和
    max_assistant_turns 限制 agent 交互轮数；advantage_* 控制组内 advantage
    的中心化/标准化/裁剪方式。
    """

    group_size: int = 8
    max_search_calls: int = 4
    max_assistant_turns: int = 6
    max_trajectory_tokens: int = 8192
    max_assistant_tokens: int = 1024
    max_tool_response_tokens: int = 1024
    temperature: float = 1.0
    top_p: float = 1.0
    seed: int = 42
    advantage_normalization: str = "center"
    advantage_epsilon: float = 1e-6
    advantage_clip: float = 0.0
    reward_shaping: RewardShapingConfig = field(default_factory=RewardShapingConfig)

    def __post_init__(self) -> None:
        if self.advantage_normalization not in {"center", "standardize"}:
            raise ValueError("advantage_normalization must be 'center' or 'standardize'")
        if self.advantage_epsilon <= 0.0:
            raise ValueError("advantage_epsilon must be positive")
        if self.advantage_clip < 0.0:
            raise ValueError("advantage_clip must be non-negative")


@dataclass
class AssistantTurn:
    """一次可训练的 assistant 生成。

    prompt_tokens 是该轮生成前的完整上下文，completion_tokens/logprobs 来自
    PyTRIO 采样。turn-level credit 会写入 effective_advantage 和 credit_* 字段。
    """

    prompt_tokens: list[int]
    completion_tokens: list[int]
    logprobs: list[float]
    text: str
    effective_advantage: float | None = None
    credit_label: str = ""
    credit_bonus: float = 0.0
    credit_query: str | None = None


@dataclass
class Trajectory:
    """一条完整多轮 rollout 及其训练信号。"""

    example: SearchExample
    group_index: int
    messages: list[dict[str, Any]]
    next_prompt_tokens: list[int] | None = None
    question_index: int = 0
    turns: list[AssistantTurn] = field(default_factory=list)
    # events 是落盘 JSONL/report 的可观测轨迹；turns 是训练需要的 token/logprob。
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
    """一个 trajectory 当前状态对应的 PyTRIO 采样请求。"""

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
    """并发执行一批 PyTRIO sample_async 请求。"""
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
    """在工具响应和总上下文预算内，尽量保留完整搜索结果条目。"""
    if not result.ok:
        candidates = [f"Search error: {result.error_type or result.error or 'unknown error'}"]
    elif not result.items:
        candidates = ["Search returned no results."]
    else:
        candidates = [format_item(item, index) for index, item in enumerate(result.items, 1)]

    accepted: list[str] = []
    accepted_prompt: list[int] | None = None
    for candidate in candidates:
        # 逐条尝试加入搜索结果；只接受“完整 item”，避免截断证据文本后污染模型输入。
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
    """从 trajectory 当前状态构造下一次采样请求。"""
    prompt_tokens = (
        trajectory.next_prompt_tokens
        if trajectory.next_prompt_tokens is not None
        else build_prompt(tokenizer, trajectory.messages)
    )
    max_tokens = min(
        config.max_assistant_tokens,
        config.max_trajectory_tokens - len(prompt_tokens),
    )
    # 如果剩余上下文预算不足，直接结束该 trajectory，避免发送无效采样请求。
    if max_tokens <= 0:
        trajectory.done = True
        trajectory.stop_reason = "max_trajectory_tokens"
        return None
    return SampleRequest(trajectory_index, prompt_tokens, num_samples, max_tokens, seed)


def read_sequence(sequence: Any, tokenizer: Any) -> tuple[list[int], list[float], str]:
    """读取采样 token、旧策略 logprob 和文本。"""
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
    """消费一次 assistant 输出：要么调用搜索，要么结束 trajectory。"""
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
    # Search-R1 约定一轮 assistant 只能做一次 search；不能继续搜索时，该输出就成为
    # final_text，后续由 reward 判断它是否是有效 Answer。
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
        # 搜索结果必须能被重新拼进 token 上下文，否则训练 datum 无法保证前缀一致。
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
    # 缓存下一轮 prompt tokens，避免重复渲染时 sampled token 边界发生漂移。
    trajectory.next_prompt_tokens = next_prompt_tokens


def score_trajectory(
    trajectory: Trajectory,
    tokenizer: Any,
    reward_shaping: RewardShapingConfig,
) -> None:
    """对已完成 trajectory 计算答案 reward 和行为诊断组件。"""
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


def assign_group_advantages(
    trajectories: list[Trajectory],
    *,
    normalization: str = "center",
    epsilon: float = 1e-6,
    clip: float = 0.0,
) -> int:
    """按 question group 计算 GRPO advantage。"""
    if normalization not in {"center", "standardize"}:
        raise ValueError("normalization must be 'center' or 'standardize'")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    if clip < 0.0:
        raise ValueError("clip must be non-negative")
    groups: dict[int, list[Trajectory]] = {}
    for trajectory in trajectories:
        groups.setdefault(trajectory.question_index, []).append(trajectory)
    degenerate = 0
    for group in groups.values():
        mean_reward = sum(item.reward for item in group) / len(group)
        centered = [item.reward - mean_reward for item in group]
        scale = 1.0
        if normalization == "standardize":
            # 标准化能减少不同问题 reward 方差差异；epsilon 防止全同 reward 时除零。
            variance = sum(value * value for value in centered) / len(centered)
            scale = max(math.sqrt(variance), epsilon)
        for item in group:
            advantage = (item.reward - mean_reward) / scale
            if clip > 0.0:
                # 裁剪极端 advantage，降低小 batch on-policy 更新的不稳定性。
                advantage = max(-clip, min(clip, advantage))
            item.advantage = advantage
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
    """运行 grouped multi-turn rollout，并返回已打分、已分配 advantage 的轨迹。"""
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
                # 第一轮一次采 group_size 条，随后每条分支独立继续搜索/回答。
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
    # GRPO 的相对优势必须在同一问题的多条采样轨迹内计算。
    assign_group_advantages(
        trajectories,
        normalization=config.advantage_normalization,
        epsilon=config.advantage_epsilon,
        clip=config.advantage_clip,
    )
    return trajectories


def trajectory_to_record(trajectory: Trajectory, *, run_type: str = "rollout") -> dict[str, Any]:
    """转换为统一 JSONL/report schema，供评测、复盘和离线诊断读取。"""
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
            "turn_credits": _turn_credit_records(trajectory),
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


def _turn_credit_records(trajectory: Trajectory) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, turn in enumerate(trajectory.turns):
        if not turn.credit_label:
            continue
        records.append(
            {
                "turn_index": index,
                "label": turn.credit_label,
                "query": turn.credit_query,
                "bonus": turn.credit_bonus,
                "effective_advantage": turn.effective_advantage,
            }
        )
    return records
