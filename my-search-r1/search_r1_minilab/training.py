"""PyTRIO GRPO 训练辅助逻辑。

本文件承接 rollout 产出的 Trajectory，并完成训练前的关键转换：turn-level
credit、token-level datum 构造、micro-batch 装箱、reference/teacher logprob
对齐，以及 GRPO + KL + Skill OPSD 的 custom loss；旧门控目标显式保留。
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable

import numpy as np
import pytrio as trio

from search_r1_minilab.diagnostics import behavior_metrics, diagnose_fields
from search_r1_minilab.protocol import build_next_prompt, build_prompt, teacher_messages_with_skill
from search_r1_minilab.rollout import Trajectory
from search_r1_minilab.rewards import extract_answer
from search_r1_minilab.skills import SkillBank
from search_r1_minilab.turn_credit import (
    detect_early_answer_risk,
    detect_final_answer_guard_risk,
    detect_missing_final_hop_risk,
    find_evidence_bridge_turns,
    find_final_hop_attribute_turns,
    find_helpful_bridge_shape_turns,
)


# 训练 datum 和 micro-batch 的安全上限；超过上限时宁可跳过/拆分，也不让 PyTRIO
# 后端收到过长上下文或 padding 爆炸的 batch。
MAX_TRAIN_CONTEXT_TOKENS = 8192
MAX_MICRO_BATCH_ITEMS = 32
MAX_MICRO_BATCH_PADDED_TOKENS = 64_000

# turn credit policy 决定是否把 trajectory-level advantage 改写为 turn-level
# effective_advantage；当前最终路线使用 final_hop_bridge。
TURN_CREDIT_POLICIES = {
    "none",
    "helpful_bridge",
    "evidence_bridge",
    "final_hop_bridge",
}
# same_context 仅用于复现旧门控自模仿；新训练默认使用 skill_context。
OPSD_CONTEXT_POLICIES = {"same_context", "skill_context"}
OPSD_GATE_POLICIES = {"none", "sigmoid_gap"}

# mask policy 决定哪些 assistant turn 可以被 OPSD 蒸馏。
OPSD_MASK_POLICIES = {
    "none",
    "assistant_all",
    "final_answer",
    "credited_turns",
    "final_and_credited",
}
# positive policy 是 OPSD 的第二道 gate，避免蒸馏负优势或错误轨迹。
OPSD_POSITIVE_POLICIES = {
    "all",
    "positive_advantage",
    "positive_reward",
    "exact_match",
}


@dataclass(frozen=True)
class TurnCreditConfig:
    """turn-level credit 配置。

    bonus 用于奖励有价值搜索 turn，penalty 用于惩罚过早回答、缺 final-hop 或搜索后
    格式不干净的 final turn。
    """

    policy: str = "none"
    helpful_search_turn_bonus: float = 0.0
    evidence_search_turn_bonus: float = 0.0
    final_hop_search_turn_bonus: float = 0.0
    early_answer_turn_penalty: float = 0.0
    missing_final_hop_turn_penalty: float = 0.0
    final_answer_guard_turn_penalty: float = 0.0

    def __post_init__(self) -> None:
        if self.policy not in TURN_CREDIT_POLICIES:
            raise ValueError(
                "turn credit policy must be 'none', 'helpful_bridge', "
                "'evidence_bridge', or 'final_hop_bridge'"
            )
        if self.helpful_search_turn_bonus < 0.0:
            raise ValueError("helpful search turn bonus must be non-negative")
        if self.evidence_search_turn_bonus < 0.0:
            raise ValueError("evidence search turn bonus must be non-negative")
        if self.final_hop_search_turn_bonus < 0.0:
            raise ValueError("final-hop search turn bonus must be non-negative")
        if self.early_answer_turn_penalty < 0.0:
            raise ValueError("early answer turn penalty must be non-negative")
        if self.missing_final_hop_turn_penalty < 0.0:
            raise ValueError("missing final-hop turn penalty must be non-negative")
        if self.final_answer_guard_turn_penalty < 0.0:
            raise ValueError("final answer guard turn penalty must be non-negative")


@dataclass(frozen=True)
class OPSDConfig:
    """OPSD 辅助目标配置；旧 same-context 路径需要显式指定。"""

    coef: float = 0.0
    context_policy: str = "skill_context"
    mask_policy: str = "assistant_all"
    positive_policy: str = "all"
    min_teacher_logprob: float | None = None
    gate_policy: str = "sigmoid_gap"
    gate_beta: float = 5.0

    def __post_init__(self) -> None:
        if self.coef < 0.0:
            raise ValueError("OPSD coef must be non-negative")
        if self.context_policy not in OPSD_CONTEXT_POLICIES:
            raise ValueError("unsupported OPSD context policy")
        if self.gate_policy not in OPSD_GATE_POLICIES:
            raise ValueError("unsupported OPSD gate policy")
        if self.gate_beta <= 0.0:
            raise ValueError("OPSD gate beta must be positive")
        if self.coef > 0.0 and self.context_policy == "skill_context" and self.gate_policy != "sigmoid_gap":
            raise ValueError("skill_context requires sigmoid_gap gate")
        if self.coef > 0.0 and self.context_policy == "same_context" and self.gate_policy != "none":
            raise ValueError("same_context requires the legacy none gate")
        if self.mask_policy not in OPSD_MASK_POLICIES:
            raise ValueError(
                "unsupported OPSD mask policy"
            )
        if self.positive_policy not in OPSD_POSITIVE_POLICIES:
            raise ValueError(
                "OPSD positive policy must be 'all', 'positive_advantage', "
                "'positive_reward', or 'exact_match'"
            )


class TrainingDatum:
    """PyTRIO datum 及其额外训练元数据。

    reference_logprobs 用于 KL drift penalty；opsd_logprobs 和 opsd_mask 用于 gated
    OPSD。num_tokens 是未 padding 的长度，供 micro-batch 装箱使用。
    """

    def __init__(
        self,
        datum: trio.Datum,
        num_tokens: int,
        *,
        reference_logprobs: list[float] | None = None,
        opsd_logprobs: list[float] | None = None,
        opsd_mask: list[float] | None = None,
        trajectory: Trajectory | None = None,
        skill_id: str | None = None,
        teacher_calls: int = 0,
        teacher_seconds: float = 0.0,
        opsd_skip_reason: str | None = None,
    ) -> None:
        self.datum = datum
        self.num_tokens = num_tokens
        self.reference_logprobs = reference_logprobs
        self.opsd_logprobs = opsd_logprobs
        self.opsd_mask = opsd_mask
        self.trajectory = trajectory
        self.skill_id = skill_id
        self.teacher_calls = teacher_calls
        self.teacher_seconds = teacher_seconds
        self.opsd_skip_reason = opsd_skip_reason


def build_datum(
    trajectory: Trajectory,
    *,
    opsd_mask_policy: str = "none",
    opsd_positive_policy: str = "all",
) -> TrainingDatum:
    """把一条 rollout trajectory 转成 PyTRIO training datum。"""
    if opsd_mask_policy not in OPSD_MASK_POLICIES:
        raise ValueError(
            "unsupported OPSD mask policy"
        )
    if opsd_positive_policy not in OPSD_POSITIVE_POLICIES:
        raise ValueError(
            "OPSD positive policy must be 'all', 'positive_advantage', "
            "'positive_reward', or 'exact_match'"
        )
    if not trajectory.turns:
        raise ValueError("cannot build a training datum without assistant turns")

    full_tokens: list[int] = []
    old_logprobs_by_token: list[float] = []
    advantages_by_token: list[float] = []
    opsd_mask_by_token: list[float] = []
    assistant_token_count = 0

    for turn_index, turn in enumerate(trajectory.turns):
        if len(turn.completion_tokens) != len(turn.logprobs):
            raise ValueError(
                f"assistant turn {turn_index + 1} token/logprob lengths differ"
            )

        if turn_index == 0:
            delta_observation = turn.prompt_tokens
        elif turn.prompt_tokens[: len(full_tokens)] == full_tokens:
            delta_observation = turn.prompt_tokens[len(full_tokens) :]
        else:
            raise ValueError(
                f"assistant turn {turn_index + 1} prompt is not a trajectory prefix"
            )

        # 默认所有 assistant token 使用 trajectory-level advantage；turn credit 命中后
        # 会把单个 turn 的 effective_advantage 改成更细粒度的奖励/惩罚。
        turn_advantage = (
            trajectory.advantage
            if turn.effective_advantage is None
            else turn.effective_advantage
        )
        full_tokens.extend(delta_observation)
        full_tokens.extend(turn.completion_tokens)
        # OPSD mask 只可能覆盖 assistant completion token，observation/prompt token
        # 始终为 0，避免把工具返回文本也当成策略输出蒸馏。
        turn_opsd_mask = (
            1.0
            if _opsd_turn_selected(
                trajectory,
                turn_index,
                opsd_mask_policy,
                positive_policy=opsd_positive_policy,
                turn_advantage=turn_advantage,
            )
            else 0.0
        )
        old_logprobs_by_token.extend([0.0] * len(delta_observation))
        old_logprobs_by_token.extend(turn.logprobs)
        advantages_by_token.extend([0.0] * len(delta_observation))
        advantages_by_token.extend([turn_advantage] * len(turn.completion_tokens))
        opsd_mask_by_token.extend([0.0] * len(delta_observation))
        opsd_mask_by_token.extend([turn_opsd_mask] * len(turn.completion_tokens))
        assistant_token_count += len(turn.completion_tokens)

    if assistant_token_count == 0:
        raise ValueError("cannot build a training datum without assistant tokens")
    if not (
        len(full_tokens)
        == len(old_logprobs_by_token)
        == len(advantages_by_token)
        == len(opsd_mask_by_token)
    ):
        raise ValueError("trajectory token/logprob/advantage/OPSD mask lengths differ")

    # PyTRIO loss 使用 next-token prediction，因此所有 token-level 字段都要右移：
    # input_tokens 预测 target_tokens，对应的 old_logprobs/advantages/mask 也对齐 target。
    input_tokens = full_tokens[:-1]
    target_tokens = full_tokens[1:]
    old_logprobs = old_logprobs_by_token[1:]
    advantages = advantages_by_token[1:]
    opsd_mask = opsd_mask_by_token[1:]
    if not (
        len(input_tokens)
        == len(target_tokens)
        == len(old_logprobs)
        == len(advantages)
        == len(opsd_mask)
    ):
        raise ValueError("datum input/target/logprob/advantage/OPSD mask lengths differ")
    if len(input_tokens) > MAX_TRAIN_CONTEXT_TOKENS:
        raise ValueError(f"datum exceeds {MAX_TRAIN_CONTEXT_TOKENS} tokens")

    datum = trio.Datum(
        model_input=trio.ModelInput.from_ints(input_tokens),
        loss_fn_inputs={
            "target_tokens": np.asarray(target_tokens, dtype=np.int64),
            "logprobs": np.asarray(old_logprobs, dtype=np.float32),
            "advantages": np.asarray(advantages, dtype=np.float32),
        },
    )
    return TrainingDatum(datum, len(input_tokens), opsd_mask=opsd_mask, trajectory=trajectory)


def build_training_datums(
    trajectories: list[Trajectory],
    turn_credit: TurnCreditConfig | None = None,
    *,
    opsd_mask_policy: str = "none",
    opsd_positive_policy: str = "all",
) -> list[TrainingDatum]:
    """为一批 trajectory 构造非零 loss token 的 training datums。"""
    apply_turn_credit(trajectories, turn_credit or TurnCreditConfig())
    datums: list[TrainingDatum] = []
    for trajectory in trajectories:
        if any(turn.completion_tokens for turn in trajectory.turns):
            datum = build_datum(
                trajectory,
                opsd_mask_policy=opsd_mask_policy,
                opsd_positive_policy=opsd_positive_policy,
            )
            # Skill OPSD may still learn from a degenerate GRPO group whose
            # advantage is zero; keep it when the auxiliary mask has tokens.
            if datum_loss_token_count(datum) > 0 or any(datum.opsd_mask or []):
                datums.append(datum)
    return datums


def apply_turn_credit(
    trajectories: list[Trajectory],
    config: TurnCreditConfig,
) -> None:
    """按配置为每个 assistant turn 分配 effective_advantage。"""
    for trajectory in trajectories:
        # 每轮重置，保证重复构造 datums 或离线分析时不会带入上一次 credit 标记。
        for turn in trajectory.turns:
            turn.effective_advantage = trajectory.advantage
            turn.credit_label = ""
            turn.credit_bonus = 0.0
            turn.credit_query = None
        _clear_event_turn_credit(trajectory)
        if config.policy == "none":
            continue
        if config.policy == "helpful_bridge" and config.helpful_search_turn_bonus == 0.0:
            continue
        if config.policy == "helpful_bridge":
            _apply_helpful_bridge_credit(trajectory, config.helpful_search_turn_bonus)
        elif config.policy == "evidence_bridge":
            _apply_evidence_bridge_credit(trajectory, config)
        elif config.policy == "final_hop_bridge":
            _apply_final_hop_bridge_credit(trajectory, config)


def _apply_helpful_bridge_credit(trajectory: Trajectory, bonus: float) -> None:
    """应用早期 helpful_bridge 形状级搜索奖励。"""
    if trajectory.exact_match or not trajectory.valid_format:
        return

    for match in find_helpful_bridge_shape_turns(
        events=trajectory.events,
        question=trajectory.example.question,
    ):
        _apply_turn_label(
            trajectory,
            match.turn_index,
            label=match.label,
            query=match.query,
            effective_advantage=max(trajectory.advantage, 0.0) + bonus,
            bonus=bonus,
        )


def _apply_evidence_bridge_credit(
    trajectory: Trajectory,
    config: TurnCreditConfig,
) -> None:
    """应用 evidence_bridge 搜索奖励和 early answer 惩罚。"""
    if not trajectory.valid_format or trajectory.exact_match:
        return

    if config.evidence_search_turn_bonus > 0.0:
        for match in find_evidence_bridge_turns(
            events=trajectory.events,
            question=trajectory.example.question,
            answers=trajectory.example.answers,
        ):
            _apply_turn_label(
                trajectory,
                match.turn_index,
                label=match.label,
                query=match.query,
                effective_advantage=(
                    max(trajectory.advantage, 0.0)
                    + config.evidence_search_turn_bonus
                ),
                bonus=config.evidence_search_turn_bonus,
            )

    if config.early_answer_turn_penalty <= 0.0:
        return
    risk = detect_early_answer_risk(
        events=trajectory.events,
        question=trajectory.example.question,
        queries=_tool_queries(trajectory.events),
        final_answer=extract_answer(trajectory.final_text) or "",
        search_calls=trajectory.search_calls,
        stop_reason=trajectory.stop_reason,
    )
    if not risk.risky:
        return
    final_turn_index = _final_answer_turn_index(trajectory.events)
    if final_turn_index is None:
        return
    _apply_turn_label(
        trajectory,
        final_turn_index,
        label="early_answer_missing_followup",
        query=None,
        effective_advantage=(
            min(trajectory.advantage, 0.0) - config.early_answer_turn_penalty
        ),
        bonus=-config.early_answer_turn_penalty,
    )


def _apply_final_hop_bridge_credit(
    trajectory: Trajectory,
    config: TurnCreditConfig,
) -> None:
    """应用当前主线 final-hop bridge credit/guard 策略。"""
    if trajectory.exact_match:
        return

    final_turn_index = _final_answer_turn_index(trajectory.events)
    # final_answer_guard 先执行：即使 final answer 格式错误，也要能惩罚“搜索后没干净回答”。
    if (
        final_turn_index is not None
        and config.final_answer_guard_turn_penalty > 0.0
    ):
        guard_risk = detect_final_answer_guard_risk(
            search_calls=trajectory.search_calls,
            stop_reason=trajectory.stop_reason,
            valid_format=trajectory.valid_format,
            exact_match=trajectory.exact_match,
        )
        if guard_risk.risky:
            _apply_turn_label(
                trajectory,
                final_turn_index,
                label="final_answer_guard",
                query=None,
                effective_advantage=(
                    min(trajectory.advantage, 0.0)
                    - config.final_answer_guard_turn_penalty
                ),
                bonus=-config.final_answer_guard_turn_penalty,
            )

    if not trajectory.valid_format:
        return

    # 正向 credit 只给格式有效但 EM 错误的轨迹，目标是保留错误轨迹中的好搜索动作。
    if config.evidence_search_turn_bonus > 0.0:
        for match in find_evidence_bridge_turns(
            events=trajectory.events,
            question=trajectory.example.question,
            answers=trajectory.example.answers,
        ):
            _apply_turn_label(
                trajectory,
                match.turn_index,
                label=match.label,
                query=match.query,
                effective_advantage=(
                    max(trajectory.advantage, 0.0)
                    + config.evidence_search_turn_bonus
                ),
                bonus=config.evidence_search_turn_bonus,
            )

    if config.final_hop_search_turn_bonus > 0.0:
        for match in find_final_hop_attribute_turns(
            events=trajectory.events,
            question=trajectory.example.question,
            answers=trajectory.example.answers,
        ):
            _apply_turn_label(
                trajectory,
                match.turn_index,
                label=match.label,
                query=match.query,
                effective_advantage=(
                    max(trajectory.advantage, 0.0)
                    + config.final_hop_search_turn_bonus
                ),
                bonus=config.final_hop_search_turn_bonus,
            )

    if final_turn_index is None:
        return

    # 负向 credit 写在 final turn 上，训练时会降低该 final answer token 的优势。
    if config.early_answer_turn_penalty > 0.0:
        risk = detect_early_answer_risk(
            events=trajectory.events,
            question=trajectory.example.question,
            queries=_tool_queries(trajectory.events),
            final_answer=extract_answer(trajectory.final_text) or "",
            search_calls=trajectory.search_calls,
            stop_reason=trajectory.stop_reason,
        )
        if risk.risky:
            _apply_turn_label(
                trajectory,
                final_turn_index,
                label="early_answer_missing_followup",
                query=None,
                effective_advantage=(
                    min(trajectory.advantage, 0.0)
                    - config.early_answer_turn_penalty
                ),
                bonus=-config.early_answer_turn_penalty,
            )

    if config.missing_final_hop_turn_penalty <= 0.0:
        return
    risk = detect_missing_final_hop_risk(
        events=trajectory.events,
        question=trajectory.example.question,
        queries=_tool_queries(trajectory.events),
        final_answer=extract_answer(trajectory.final_text) or "",
        search_calls=trajectory.search_calls,
        stop_reason=trajectory.stop_reason,
    )
    if not risk.risky:
        return
    _apply_turn_label(
        trajectory,
        final_turn_index,
        label="missing_final_hop_attribute",
        query=None,
        effective_advantage=(
            min(trajectory.advantage, 0.0)
            - config.missing_final_hop_turn_penalty
        ),
        bonus=-config.missing_final_hop_turn_penalty,
    )


def _apply_turn_label(
    trajectory: Trajectory,
    turn_index: int,
    *,
    label: str,
    query: str | None,
    effective_advantage: float,
    bonus: float,
) -> None:
    """把 credit 标签写入 turn，并同步写入 events 方便 JSONL/report 复盘。"""
    if turn_index >= len(trajectory.turns):
        return
    turn = trajectory.turns[turn_index]
    turn.effective_advantage = effective_advantage
    turn.credit_label = label
    turn.credit_bonus = bonus
    turn.credit_query = query
    _mark_event_turn_credit(
        trajectory.events,
        turn_index,
        label=label,
        bonus=bonus,
        effective_advantage=effective_advantage,
    )


def _clear_event_turn_credit(trajectory: Trajectory) -> None:
    for event in trajectory.events:
        if isinstance(event, dict):
            event.pop("turn_credit", None)


def _mark_event_turn_credit(
    events: list[dict[str, Any]],
    turn_index: int,
    *,
    label: str,
    bonus: float,
    effective_advantage: float,
) -> None:
    assistant_index = 0
    for event in events:
        if event.get("role") != "assistant":
            continue
        if assistant_index == turn_index:
            event["turn_credit"] = {
                "label": label,
                "bonus": bonus,
                "effective_advantage": effective_advantage,
            }
            return
        assistant_index += 1


def _tool_queries(events: list[dict[str, Any]]) -> list[str]:
    queries: list[str] = []
    for event in events:
        tool_call = event.get("tool_call")
        if isinstance(tool_call, dict) and isinstance(tool_call.get("query"), str):
            queries.append(tool_call["query"])
    return queries


def _final_answer_turn_index(events: list[dict[str, Any]]) -> int | None:
    assistant_indices: list[tuple[int, dict[str, Any]]] = []
    assistant_index = 0
    for event in events:
        if event.get("role") != "assistant":
            continue
        assistant_indices.append((assistant_index, event))
        assistant_index += 1
    for index, event in reversed(assistant_indices):
        if not isinstance(event.get("tool_call"), dict):
            return index
    return assistant_indices[-1][0] if assistant_indices else None


def _opsd_turn_selected(
    trajectory: Trajectory,
    turn_index: int,
    policy: str,
    *,
    positive_policy: str,
    turn_advantage: float,
) -> bool:
    """判断当前 turn 是否进入 OPSD token mask。"""
    if policy == "none":
        return False
    if policy == "assistant_all":
        return _opsd_positive_gate(
            trajectory, positive_policy, turn_advantage=turn_advantage
        )
    turn = trajectory.turns[turn_index]
    credited = bool(turn.credit_label)
    final_answer = _is_final_answer_turn(trajectory.events, turn_index)
    if policy == "final_answer":
        selected = final_answer
    elif policy == "credited_turns":
        selected = credited
    elif policy == "final_and_credited":
        selected = final_answer or credited
    else:
        raise ValueError(f"unsupported OPSD mask policy: {policy}")
    return selected and _opsd_positive_gate(
        trajectory,
        positive_policy,
        turn_advantage=turn_advantage,
    )


def _opsd_positive_gate(
    trajectory: Trajectory,
    policy: str,
    *,
    turn_advantage: float,
) -> bool:
    """OPSD 正向 gate，避免无条件蒸馏坏轨迹或负优势 turn。"""
    if policy == "all":
        return True
    if policy == "positive_advantage":
        return turn_advantage > 0.0
    if policy == "positive_reward":
        return trajectory.reward > 0.0
    if policy == "exact_match":
        return trajectory.exact_match
    raise ValueError(f"unsupported OPSD positive policy: {policy}")


def _is_final_answer_turn(events: list[dict[str, Any]], turn_index: int) -> bool:
    assistant_index = 0
    for event in events:
        if event.get("role") != "assistant":
            continue
        if assistant_index == turn_index:
            return event.get("parsed_kind") == "answer"
        assistant_index += 1
    return False


def datum_size(item: TrainingDatum) -> int:
    """返回未 padding 的 datum token 长度。"""
    return item.num_tokens


def datum_loss_token_count(item: TrainingDatum) -> int:
    """统计真正参与 policy loss 的 target token 数。"""
    advantages = _to_numpy(item.datum.loss_fn_inputs["advantages"])
    return int(np.count_nonzero(advantages))


def pack_micro_batches(datums: list[TrainingDatum]) -> list[list[TrainingDatum]]:
    """用 first-fit decreasing 对变长 datums 做 micro-batch 装箱。"""
    batches: list[list[TrainingDatum]] = []
    batch_max_tokens: list[int] = []
    for item in sorted(datums, key=datum_size, reverse=True):
        if item.num_tokens > MAX_TRAIN_CONTEXT_TOKENS:
            raise ValueError("single datum exceeds training context limit")
        for index, batch in enumerate(batches):
            next_items = len(batch) + 1
            next_max_tokens = max(batch_max_tokens[index], item.num_tokens)
            next_padded_tokens = next_items * next_max_tokens
            # padding 后 token 总量才是显存/后端成本关键，因此同时限制条数和 padded tokens。
            if (
                next_items <= MAX_MICRO_BATCH_ITEMS
                and next_padded_tokens <= MAX_MICRO_BATCH_PADDED_TOKENS
            ):
                batch.append(item)
                batch_max_tokens[index] = next_max_tokens
                break
        else:
            batches.append([item])
            batch_max_tokens.append(item.num_tokens)
    return batches


def weight_micro_batch_for_global_mean(
    micro_batch: list[TrainingDatum],
    total_samples: int,
) -> list[trio.Datum]:
    """缩放 micro-batch advantage，使梯度累积后等价于整批求均值。"""
    return [
        item.datum
        for item in weight_micro_batch_items_for_global_mean(micro_batch, total_samples)
    ]


def weight_micro_batch_items_for_global_mean(
    micro_batch: list[TrainingDatum],
    total_samples: int,
) -> list[TrainingDatum]:
    """缩放 advantage，同时保留 reference/OPSD 等自定义 loss 元数据。"""
    if not micro_batch:
        return []
    if total_samples <= 0:
        raise ValueError("total_samples must be positive")
    if len(micro_batch) > total_samples:
        raise ValueError("micro-batch size cannot exceed total_samples")

    micro_batch_weight = np.float32(len(micro_batch) / total_samples)
    weighted_items: list[TrainingDatum] = []
    for item in micro_batch:
        loss_inputs = item.datum.loss_fn_inputs
        weighted_datum = trio.Datum(
            model_input=item.datum.model_input,
            loss_fn_inputs={
                "target_tokens": _to_numpy(loss_inputs["target_tokens"]),
                "logprobs": _to_numpy(loss_inputs["logprobs"]),
                "advantages": _to_numpy(loss_inputs["advantages"]) * micro_batch_weight,
            },
        )
        weighted_items.append(
            TrainingDatum(
                weighted_datum,
                item.num_tokens,
                reference_logprobs=item.reference_logprobs,
                opsd_logprobs=item.opsd_logprobs,
                opsd_mask=item.opsd_mask,
                trajectory=item.trajectory,
                skill_id=item.skill_id,
                teacher_calls=item.teacher_calls,
                teacher_seconds=item.teacher_seconds,
                opsd_skip_reason=item.opsd_skip_reason,
            )
        )
    return weighted_items


def add_reference_logprobs(
    datums: list[TrainingDatum],
    reference_client: Any,
) -> list[TrainingDatum]:
    """为 datum 附加 frozen reference policy 的 shifted target logprobs。"""
    return [
        TrainingDatum(
            item.datum,
            item.num_tokens,
            reference_logprobs=compute_reference_logprobs(item, reference_client),
            opsd_logprobs=item.opsd_logprobs,
            opsd_mask=item.opsd_mask,
            trajectory=item.trajectory,
            skill_id=item.skill_id,
            teacher_calls=item.teacher_calls,
            teacher_seconds=item.teacher_seconds,
            opsd_skip_reason=item.opsd_skip_reason,
        )
        for item in datums
    ]


def compute_reference_logprobs(
    item: TrainingDatum,
    reference_client: Any,
) -> list[float]:
    """计算一个已 shift datum 的 reference logprobs。"""
    advantages = _to_numpy(item.datum.loss_fn_inputs["advantages"])
    # KL 只要求 trainable token 有 reference logprob；非训练 token 缺失时可填 0。
    required_mask = [float(value) != 0.0 for value in advantages]
    return _compute_shifted_logprobs(
        item,
        reference_client,
        required_mask=required_mask,
        missing_message="missing reference logprob for trainable token",
    )


def add_opsd_teacher_logprobs(
    datums: list[TrainingDatum],
    teacher_client: Any,
    *,
    min_teacher_logprob: float | None = None,
) -> list[TrainingDatum]:
    """为 datum 附加 OPSD teacher logprobs，并按 teacher 置信度收窄 mask。"""
    updated: list[TrainingDatum] = []
    for item in datums:
        opsd_logprobs = compute_opsd_teacher_logprobs(item, teacher_client)
        opsd_mask = list(_require_opsd_mask(item))
        if min_teacher_logprob is not None:
            # min_teacher_logprob 是保守 gate：teacher 自己概率太低的 token 不蒸馏。
            opsd_mask = [
                mask if logprob >= min_teacher_logprob else 0.0
                for mask, logprob in zip(opsd_mask, opsd_logprobs, strict=True)
            ]
        updated.append(
            TrainingDatum(
                item.datum,
                item.num_tokens,
                reference_logprobs=item.reference_logprobs,
                opsd_logprobs=opsd_logprobs,
                opsd_mask=opsd_mask,
                trajectory=item.trajectory,
                skill_id=item.skill_id,
            )
        )
    return updated


def add_skill_teacher_logprobs(
    datums: list[TrainingDatum],
    teacher_client: Any,
    tokenizer: Any,
    skill_bank: SkillBank,
    *,
    min_teacher_logprob: float | None = None,
) -> list[TrainingDatum]:
    """Score the same sampled assistant tokens with a current-policy Skill teacher."""
    updated: list[TrainingDatum] = []
    for item in datums:
        trajectory = item.trajectory
        if trajectory is None:
            raise ValueError("skill OPSD datum is missing its trajectory")
        skill = skill_bank.select(trajectory.example.data_source)
        mask = list(_require_opsd_mask(item))
        scores = [0.0] * len(mask)
        teacher_history = teacher_messages_with_skill(
            trajectory.messages[:2], skill.teacher_context
        )
        teacher_prompt = build_prompt(tokenizer, teacher_history)
        message_index = 2
        previous_full: list[int] = []
        calls = 0
        started = perf_counter()
        skip_reason: str | None = None

        for turn_index, turn in enumerate(trajectory.turns):
            if turn_index > 0 and turn.prompt_tokens[: len(previous_full)] != previous_full:
                raise ValueError("student turn prompt is not a trajectory prefix")
            if len(turn.completion_tokens) != len(turn.logprobs):
                raise ValueError("assistant completion/logprob lengths differ")
            start = len(turn.prompt_tokens) - 1
            end = start + len(turn.completion_tokens)
            if end > len(mask) or start < 0:
                raise ValueError("teacher completion does not align with datum targets")
            if message_index >= len(trajectory.messages):
                raise ValueError("missing assistant message for sampled turn")
            if trajectory.messages[message_index].get("role") != "assistant" or trajectory.messages[message_index].get("content") != turn.text:
                raise ValueError("assistant message differs from sampled turn")

            if any(mask[start:end]):
                full_teacher_tokens = [*teacher_prompt, *turn.completion_tokens]
                if len(full_teacher_tokens) > MAX_TRAIN_CONTEXT_TOKENS:
                    skip_reason = "teacher_context_limit"
                    mask = [0.0] * len(mask)
                    scores = [0.0] * len(scores)
                    break
                response = teacher_client.compute_logprobs(
                    trio.ModelInput.from_ints(full_teacher_tokens)
                ).result()
                calls += 1
                if len(response) != len(full_teacher_tokens):
                    raise ValueError("skill teacher logprob length mismatch")
                for offset, value in enumerate(response[-len(turn.completion_tokens):]):
                    position = start + offset
                    if value is None:
                        if mask[position]:
                            raise ValueError("missing skill teacher logprob for masked token")
                        continue
                    scores[position] = float(value)
                    if min_teacher_logprob is not None and scores[position] < min_teacher_logprob:
                        mask[position] = 0.0

            next_turn_exists = turn_index + 1 < len(trajectory.turns)
            if next_turn_exists:
                if message_index + 1 >= len(trajectory.messages):
                    raise ValueError("missing tool observation before next assistant turn")
                tool = trajectory.messages[message_index + 1]
                if tool.get("role") != "tool":
                    raise ValueError("non-tool observation before next assistant turn")
                teacher_prompt = build_next_prompt(
                    tokenizer,
                    teacher_history,
                    turn.text,
                    teacher_prompt,
                    turn.completion_tokens,
                    tool,
                )
                teacher_history.extend((trajectory.messages[message_index], tool))
                message_index += 2
            else:
                message_index += 1
            previous_full = [*turn.prompt_tokens, *turn.completion_tokens]
        if skip_reason is None and message_index != len(trajectory.messages):
            raise ValueError("teacher replay did not consume the full trajectory")
        updated.append(
            TrainingDatum(
                item.datum,
                item.num_tokens,
                reference_logprobs=item.reference_logprobs,
                opsd_logprobs=scores,
                opsd_mask=mask,
                trajectory=trajectory,
                skill_id=skill.skill_id,
                teacher_calls=calls,
                teacher_seconds=perf_counter() - started,
                opsd_skip_reason=skip_reason,
            )
        )
    return updated


def compute_opsd_teacher_logprobs(
    item: TrainingDatum,
    teacher_client: Any,
) -> list[float]:
    """计算 OPSD mask token 所需的 teacher logprobs。"""
    opsd_mask = _require_opsd_mask(item)
    # OPSD 只强制 masked token 有 teacher logprob；未选 token 可填 0。
    required_mask = [float(value) != 0.0 for value in opsd_mask]
    return _compute_shifted_logprobs(
        item,
        teacher_client,
        required_mask=required_mask,
        missing_message="missing OPSD teacher logprob for masked token",
    )


def _compute_shifted_logprobs(
    item: TrainingDatum,
    logprob_client: Any,
    *,
    required_mask: list[bool],
    missing_message: str,
) -> list[float]:
    """调用 logprob client，并把返回值对齐到已 shift 的 target token。"""
    input_tokens = [int(token) for token in item.datum.model_input.tolist()]
    target_tokens = [
        int(token) for token in _to_numpy(item.datum.loss_fn_inputs["target_tokens"])
    ]
    if not input_tokens or not target_tokens:
        raise ValueError("datum must contain input and target tokens")
    if len(input_tokens) != len(target_tokens):
        raise ValueError("datum input and target lengths differ")
    if len(required_mask) != len(target_tokens):
        raise ValueError("required logprob mask length does not match target tokens")

    # compute_logprobs 接收完整序列，返回每个位置 token 的 logprob；丢掉第一个位置后
    # 才与 target_tokens 一一对齐。
    full_tokens = [*input_tokens, target_tokens[-1]]
    all_logprobs = logprob_client.compute_logprobs(
        trio.ModelInput.from_ints(full_tokens)
    ).result()
    shifted_logprobs = all_logprobs[1:]
    if len(shifted_logprobs) != len(target_tokens):
        raise ValueError("model logprob length does not match target tokens")

    aligned: list[float] = []
    for logprob, required in zip(shifted_logprobs, required_mask, strict=True):
        if logprob is None:
            if required:
                raise ValueError(missing_message)
            aligned.append(0.0)
        else:
            aligned.append(float(logprob))
    return aligned


def _require_opsd_mask(item: TrainingDatum) -> list[float]:
    if item.opsd_mask is None:
        raise ValueError("OPSD mask is missing")
    target_tokens = _to_numpy(item.datum.loss_fn_inputs["target_tokens"])
    if len(item.opsd_mask) != len(target_tokens):
        raise ValueError("OPSD mask length does not match target tokens")
    return item.opsd_mask


def build_custom_forward_datums(items: list[TrainingDatum]) -> list[trio.Datum]:
    """送入 PyTRIO forward 时只保留 target_tokens，其它 loss 元数据留在闭包里。"""
    return [
        trio.Datum(
            model_input=item.datum.model_input,
            loss_fn_inputs={
                "target_tokens": item.datum.loss_fn_inputs["target_tokens"],
            },
        )
        for item in items
    ]


def make_grpo_kl_loss_fn(
    sampling_logprobs_list: list[list[float]],
    advantages_list: list[list[float]],
    reference_logprobs_list: list[list[float]] | None = None,
    *,
    kl_coef: float,
    policy_ratio_clip: float = 0.0,
    opsd_coef: float = 0.0,
    opsd_logprobs_list: list[list[float]] | None = None,
    opsd_mask_list: list[list[float]] | None = None,
    opsd_context_policy: str = "same_context",
    opsd_gate_beta: float = 5.0,
    opsd_total_trajectories: int | None = None,
    opsd_action_kind_list: list[list[int]] | None = None,
) -> Callable[[list[trio.Datum], list[Any]], tuple[Any, dict[str, float]]]:
    """Create GRPO + reference drift + explicit legacy or Skill OPSD loss."""
    if kl_coef < 0.0:
        raise ValueError("kl_coef must be non-negative")
    if policy_ratio_clip < 0.0:
        raise ValueError("policy_ratio_clip must be non-negative")
    if opsd_coef < 0.0:
        raise ValueError("OPSD coef must be non-negative")
    if kl_coef > 0.0 and reference_logprobs_list is None:
        raise ValueError("KL training requires reference logprobs")
    if opsd_coef > 0.0 and (
        opsd_logprobs_list is None or opsd_mask_list is None
    ):
        raise ValueError("OPSD training requires teacher logprobs and mask")
    if opsd_context_policy not in OPSD_CONTEXT_POLICIES:
        raise ValueError("unsupported OPSD context policy")
    if opsd_gate_beta <= 0.0:
        raise ValueError("OPSD gate beta must be positive")
    if opsd_coef > 0.0 and opsd_context_policy == "skill_context" and (
        opsd_total_trajectories is None or opsd_total_trajectories <= 0
    ):
        raise ValueError("skill OPSD requires a positive global trajectory count")

    def loss_fn(
        data: list[trio.Datum],
        current_logprobs_list: list[Any],
    ) -> tuple[Any, dict[str, float]]:
        import torch

        batch_len = len(data)
        if not (
            batch_len
            == len(current_logprobs_list)
            == len(sampling_logprobs_list)
            == len(advantages_list)
        ):
            raise ValueError("GRPO KL loss got mismatched batch lengths")
        if reference_logprobs_list is not None and len(reference_logprobs_list) != batch_len:
            raise ValueError("GRPO KL reference batch length mismatch")
        if opsd_logprobs_list is not None and len(opsd_logprobs_list) != batch_len:
            raise ValueError("OPSD teacher batch length mismatch")
        if opsd_mask_list is not None and len(opsd_mask_list) != batch_len:
            raise ValueError("OPSD mask batch length mismatch")
        if opsd_action_kind_list is not None and len(opsd_action_kind_list) != batch_len:
            raise ValueError("OPSD action-kind batch length mismatch")

        losses = []
        ratio_chunks = []
        kl_chunks = []
        clip_chunks = []
        opsd_current_chunks = []
        opsd_teacher_chunks = []
        opsd_gap_chunks = []
        opsd_signed_gap_chunks = []
        opsd_gate_chunks = []
        action_gate_chunks: dict[int, list[Any]] = defaultdict(list)
        action_gap_chunks: dict[int, list[Any]] = defaultdict(list)
        skill_opsd_terms = []
        train_tokens = 0
        opsd_masked_tokens = 0
        denominator = 0

        for item_index, (current_logprobs, old_values, advantage_values) in enumerate(zip(
            current_logprobs_list,
            sampling_logprobs_list,
            advantages_list,
            strict=True,
        )):
            current = current_logprobs.float()
            device = current.device
            old = torch.as_tensor(old_values, dtype=torch.float32, device=device)
            advantages = torch.as_tensor(
                advantage_values,
                dtype=torch.float32,
                device=device,
            )
            reference = None
            if reference_logprobs_list is not None:
                reference = torch.as_tensor(
                    reference_logprobs_list[item_index],
                    dtype=torch.float32,
                    device=device,
                )
            if not (len(current) == len(old) == len(advantages)):
                raise ValueError("GRPO KL datum fields must have the same length")
            if reference is not None and len(reference) != len(current):
                raise ValueError("GRPO KL reference field must match current length")

            # importance ratio 使用当前策略 logprob 与采样时 old logprob 的差。
            ratio = torch.exp(current - old)
            effective_ratio = ratio
            if policy_ratio_clip > 0.0:
                effective_ratio = torch.clamp(
                    ratio,
                    min=1.0 - policy_ratio_clip,
                    max=1.0 + policy_ratio_clip,
                )

            # advantage 为 0 的 token 是 prompt/tool observation 或无训练信号 token。
            train_mask = advantages != 0.0
            objective = effective_ratio * advantages
            datum_loss = -objective.sum()
            if torch.any(train_mask) and kl_coef > 0.0:
                if reference is None:
                    raise ValueError("KL training requires reference logprobs")
                # sampled-token logprob drift 的二次惩罚，用于约束当前策略不要在训练
                # token 上偏离 frozen reference 太远。
                logprob_drift = current - reference
                kl_penalty = 0.5 * logprob_drift.pow(2)
                datum_loss = datum_loss + kl_coef * kl_penalty[train_mask].sum()
                kl_chunks.append(kl_penalty.detach()[train_mask])
            losses.append(datum_loss)

            denominator += int(current.numel())
            if torch.any(train_mask):
                ratio_chunks.append(ratio.detach()[train_mask])
                if policy_ratio_clip > 0.0:
                    clip_chunks.append(
                        (ratio.detach()[train_mask] != effective_ratio.detach()[train_mask]).float()
                    )
                train_tokens += int(train_mask.sum().item())

            if opsd_logprobs_list is not None and opsd_mask_list is not None:
                teacher = torch.as_tensor(
                    opsd_logprobs_list[item_index],
                    dtype=torch.float32,
                    device=device,
                )
                opsd_mask = torch.as_tensor(
                    opsd_mask_list[item_index],
                    dtype=torch.float32,
                    device=device,
                )
                if not (len(current) == len(teacher) == len(opsd_mask)):
                    raise ValueError("OPSD datum fields must have the same length")
                # selected token 已经过 mask policy、positive gate 和可选 teacher
                # logprob gate；这里只聚合被选中的当前 logprob。
                selected = opsd_mask != 0.0
                if torch.any(selected):
                    selected_current = current[selected]
                    selected_teacher = teacher[selected]
                    opsd_current_chunks.append(selected_current)
                    opsd_teacher_chunks.append(selected_teacher.detach())
                    opsd_gap_chunks.append((selected_current - selected_teacher).abs().detach())
                    opsd_masked_tokens += int(selected.sum().item())
                    if opsd_context_policy == "skill_context":
                        signed_gap = (selected_teacher.detach() - selected_current.detach())
                        gate = torch.sigmoid(opsd_gate_beta * signed_gap).detach()
                        skill_opsd_terms.append(
                            (gate * (selected_teacher.detach() - selected_current)).mean()
                        )
                        opsd_signed_gap_chunks.append(signed_gap)
                        opsd_gate_chunks.append(gate)
                        if opsd_action_kind_list is not None:
                            kind_values = torch.as_tensor(
                                opsd_action_kind_list[item_index], dtype=torch.int64, device=device
                            )
                            if len(kind_values) != len(current):
                                raise ValueError("OPSD action kinds must match current length")
                            selected_kinds = kind_values[selected]
                            for kind in (1, 2, 3):
                                kind_mask = selected_kinds == kind
                                if torch.any(kind_mask):
                                    action_gate_chunks[kind].append(gate[kind_mask])
                                    action_gap_chunks[kind].append(signed_gap[kind_mask])

        grpo_loss = torch.stack(losses).sum()
        loss = grpo_loss
        opsd_loss_value = 0.0
        if opsd_coef > 0.0 and opsd_current_chunks:
            if opsd_context_policy == "skill_context":
                # Every trajectory uses its own selected-token mean, then a fixed
                # whole-step denominator. Partitioning into micro-batches cannot
                # change the auxiliary gradient scale.
                opsd_loss = torch.stack(skill_opsd_terms).sum() / opsd_total_trajectories
            else:
                # Explicit legacy reproduction: teacher only filters a positive
                # imitation loss and does not set its gradient weight.
                opsd_loss = -torch.cat(opsd_current_chunks).mean()
            loss = loss + opsd_coef * opsd_loss
            opsd_loss_value = float(opsd_loss.detach().item())
        metrics = {
            "loss_mean": float(loss.detach().item() / denominator)
            if denominator > 0
            else 0.0,
            "grpo_kl/coef": float(kl_coef),
            "grpo_kl/train_tokens": float(train_tokens),
        }
        if opsd_coef > 0.0 or opsd_logprobs_list is not None or opsd_mask_list is not None:
            metrics.update(
                {
                    "opsd/coef": float(opsd_coef),
                    "opsd/masked_tokens": float(opsd_masked_tokens),
                    "opsd/mask_rate": (
                        float(opsd_masked_tokens / denominator)
                        if denominator > 0
                        else 0.0
                    ),
                    "opsd/total_tokens": float(denominator),
                    "opsd/loss_mean": opsd_loss_value,
                    "opsd/skill_mode": float(opsd_context_policy == "skill_context"),
                }
            )
        if ratio_chunks:
            ratios = torch.cat(ratio_chunks)
            metrics["grpo_kl/ratio_mean"] = float(ratios.mean().item())
            metrics["grpo_kl/ratio_max"] = float(ratios.max().item())
        if kl_chunks:
            penalties = torch.cat(kl_chunks)
            metrics["grpo_kl/logprob_mse_mean"] = float(penalties.mean().item())
        if clip_chunks:
            clips = torch.cat(clip_chunks)
            metrics["grpo_kl/clip_fraction"] = float(clips.mean().item())
        if opsd_teacher_chunks:
            teacher_logprobs = torch.cat(opsd_teacher_chunks)
            gaps = torch.cat(opsd_gap_chunks)
            metrics["opsd/teacher_logprob_mean"] = float(
                teacher_logprobs.mean().item()
            )
            metrics["opsd/student_teacher_gap_mean"] = float(gaps.mean().item())
        if opsd_gate_chunks:
            gates = torch.cat(opsd_gate_chunks)
            signed_gaps = torch.cat(opsd_signed_gap_chunks)
            metrics["opsd/gate_mean"] = float(gates.mean().item())
            metrics["opsd/signed_gap_mean"] = float(signed_gaps.mean().item())
            metrics["opsd/positive_gap_fraction"] = float(
                (signed_gaps > 0).float().mean().item()
            )
            metrics["opsd/selected_trajectories"] = float(len(skill_opsd_terms))
            for kind, label in ((1, "search"), (2, "answer"), (3, "invalid")):
                if action_gate_chunks[kind]:
                    kind_gates = torch.cat(action_gate_chunks[kind])
                    kind_gaps = torch.cat(action_gap_chunks[kind])
                    metrics[f"opsd/{label}/selected_tokens"] = float(kind_gates.numel())
                    metrics[f"opsd/{label}/gate_mean"] = float(kind_gates.mean().item())
                    metrics[f"opsd/{label}/signed_gap_mean"] = float(kind_gaps.mean().item())
        return loss, metrics

    return loss_fn


def loss_input_float_lists(items: list[TrainingDatum], key: str) -> list[list[float]]:
    """从 TrainingDatum 批量读取 float loss input 字段。"""
    values: list[list[float]] = []
    for item in items:
        values.append([float(value) for value in _to_numpy(item.datum.loss_fn_inputs[key])])
    return values


def opsd_logprob_float_lists(items: list[TrainingDatum]) -> list[list[float]]:
    """从 TrainingDatum 批量读取 OPSD teacher logprobs。"""
    values: list[list[float]] = []
    for item in items:
        if item.opsd_logprobs is None:
            raise ValueError("OPSD teacher logprobs are missing")
        values.append([float(value) for value in item.opsd_logprobs])
    return values


def opsd_mask_float_lists(items: list[TrainingDatum]) -> list[list[float]]:
    """从 TrainingDatum 批量读取 OPSD token mask。"""
    values: list[list[float]] = []
    for item in items:
        values.append([float(value) for value in _require_opsd_mask(item)])
    return values


def opsd_action_kind_lists(items: list[TrainingDatum]) -> list[list[int]]:
    """Mark sampled search, answer, and invalid assistant targets for diagnostics."""
    values: list[list[int]] = []
    for item in items:
        trajectory = item.trajectory
        if trajectory is None:
            raise ValueError("OPSD action diagnostics require trajectory metadata")
        events = [event for event in trajectory.events if event.get("role") == "assistant"]
        if len(events) != len(trajectory.turns):
            raise ValueError("assistant events and turns differ")
        kinds = [0] * len(_require_opsd_mask(item))
        for turn, event in zip(trajectory.turns, events, strict=True):
            start = len(turn.prompt_tokens) - 1
            end = start + len(turn.completion_tokens)
            if start < 0 or end > len(kinds):
                raise ValueError("assistant action kind does not align with targets")
            kind = {"tool": 1, "answer": 2}.get(event.get("parsed_kind"), 3)
            kinds[start:end] = [kind] * len(turn.completion_tokens)
        values.append(kinds)
    return values


def mean(values: list[float]) -> float:
    """返回算术平均值；空列表返回 0。"""
    return sum(values) / len(values) if values else 0.0


def source_reward(trajectories: list[Trajectory], source_name: str) -> float:
    """Return mean reward for one data-source substring."""
    rewards = [
        trajectory.reward
        for trajectory in trajectories
        if source_name in trajectory.example.data_source.lower()
    ]
    return mean(rewards)


def degenerate_group_count(trajectories: list[Trajectory]) -> int:
    """Count question groups whose centered advantages are all zero."""
    groups: dict[int, list[float]] = defaultdict(list)
    for trajectory in trajectories:
        groups[trajectory.question_index].append(trajectory.advantage)
    return sum(all(advantage == 0.0 for advantage in values) for values in groups.values())


def rollout_metrics(
    trajectories: list[Trajectory],
    datums: list[TrainingDatum],
    micro_batches: list[list[TrainingDatum]],
    question_count: int,
    *,
    turn_credit_policy: str = "none",
) -> dict[str, float]:
    """Summarize local rollout, reward, and packing metrics."""
    tool_attempts = sum(
        event.get("role") == "assistant" and "tool_call" in event
        for trajectory in trajectories
        for event in trajectory.events
    )
    valid_tool_calls = sum(trajectory.search_calls for trajectory in trajectories)
    trajectory_lengths = [
        len(trajectory.turns[-1].prompt_tokens)
        + len(trajectory.turns[-1].completion_tokens)
        for trajectory in trajectories
        if trajectory.turns
    ]
    micro_batch_padded_tokens = [
        max((item.num_tokens for item in batch), default=0) * len(batch)
        for batch in micro_batches
    ]
    input_tokens = sum(item.num_tokens for item in datums)
    loss_tokens = sum(datum_loss_token_count(item) for item in datums)
    padded_tokens = sum(micro_batch_padded_tokens)
    metrics = {
        "reward/mean": mean([trajectory.reward for trajectory in trajectories]),
        "reward/correct": mean([float(trajectory.exact_match) for trajectory in trajectories]),
        "reward/format": mean([float(trajectory.valid_format) for trajectory in trajectories]),
        "reward/nq": source_reward(trajectories, "nq"),
        "reward/hotpotqa": source_reward(trajectories, "hotpotqa"),
        "rollout/turns": mean([float(len(trajectory.turns)) for trajectory in trajectories]),
        "rollout/search_calls": mean(
            [float(trajectory.search_calls) for trajectory in trajectories]
        ),
        "rollout/no_search_rate": mean(
            [float(trajectory.search_calls == 0) for trajectory in trajectories]
        ),
        "rollout/trajectory_tokens": mean([float(value) for value in trajectory_lengths]),
        "rollout/valid_tool_call_rate": valid_tool_calls / max(tool_attempts, 1),
        "rollout/degenerate_group_rate": degenerate_group_count(trajectories)
        / max(question_count, 1),
        "train/datums_per_rollout_batch": float(len(datums)),
        "train/micro_batches_per_step": float(len(micro_batches)),
        "train/tokens_per_rollout_batch": float(input_tokens),
        "train/loss_tokens_per_rollout_batch": float(loss_tokens),
        "train/padded_tokens_per_rollout_batch": float(padded_tokens),
        "train/max_micro_batch_padded_tokens": float(
            max(micro_batch_padded_tokens, default=0)
        ),
    }
    metrics.update(_behavior_metrics(trajectories))
    metrics.update(_turn_credit_metrics(trajectories, turn_credit_policy))
    skill_datums = [item for item in datums if item.skill_id is not None]
    if skill_datums:
        metrics["opsd/teacher_calls"] = float(sum(item.teacher_calls for item in skill_datums))
        metrics["opsd/teacher_seconds"] = sum(item.teacher_seconds for item in skill_datums)
        metrics["opsd/teacher_context_skips"] = float(
            sum(item.opsd_skip_reason == "teacher_context_limit" for item in skill_datums)
        )
        for skill_id in sorted({item.skill_id for item in skill_datums}):
            metrics[f"opsd/skill/{skill_id}/datums"] = float(
                sum(item.skill_id == skill_id for item in skill_datums)
            )
    return metrics


def evaluation_metrics(trajectories: list[Trajectory]) -> dict[str, float]:
    """Summarize deterministic eval trajectories."""
    by_source: dict[str, list[Trajectory]] = defaultdict(list)
    for trajectory in trajectories:
        by_source[trajectory.example.data_source].append(trajectory)
    metrics: dict[str, float] = {}
    source_scores: list[float] = []
    for source, items in sorted(by_source.items()):
        score = mean([float(item.exact_match) for item in items])
        metrics[f"em/{source}"] = score
        source_scores.append(score)
    metrics.update(
        {
            "em/macro": mean(source_scores),
            "format/rate": mean([float(item.valid_format) for item in trajectories]),
            "rollout/search_calls": mean(
                [float(item.search_calls) for item in trajectories]
            ),
            "rollout/no_search_rate": mean(
                [float(item.search_calls == 0) for item in trajectories]
            ),
            "rollout/turns": mean([float(len(item.turns)) for item in trajectories]),
        }
    )
    metrics.update(_behavior_metrics(trajectories))
    return metrics


def _behavior_metrics(trajectories: list[Trajectory]) -> dict[str, float]:
    return behavior_metrics(
        diagnose_fields(
            turns=trajectory.events,
            search_calls=trajectory.search_calls,
            exact_match=trajectory.exact_match,
            valid_format=trajectory.valid_format,
            stop_reason=trajectory.stop_reason,
        )
        for trajectory in trajectories
    )


def _turn_credit_metrics(
    trajectories: list[Trajectory],
    policy: str,
) -> dict[str, float]:
    helpful_search_turns = 0
    evidence_search_turns = 0
    final_hop_search_turns = 0
    early_answer_penalty_turns = 0
    missing_final_hop_penalty_turns = 0
    final_answer_guard_penalty_turns = 0
    credited_trajectories = 0
    credited_tokens = 0
    for trajectory in trajectories:
        trajectory_has_credit = False
        for turn in trajectory.turns:
            if not turn.credit_label:
                continue
            if turn.credit_label == "helpful_bridge_search":
                helpful_search_turns += 1
            elif turn.credit_label == "evidence_bridge_search":
                evidence_search_turns += 1
            elif turn.credit_label == "final_hop_attribute_search":
                final_hop_search_turns += 1
            elif turn.credit_label == "early_answer_missing_followup":
                early_answer_penalty_turns += 1
            elif turn.credit_label == "missing_final_hop_attribute":
                missing_final_hop_penalty_turns += 1
            elif turn.credit_label == "final_answer_guard":
                final_answer_guard_penalty_turns += 1
            credited_tokens += len(turn.completion_tokens)
            trajectory_has_credit = True
        if trajectory_has_credit:
            credited_trajectories += 1
    return {
        "turn_credit/policy": 1.0 if policy != "none" else 0.0,
        "turn_credit/helpful_search_turns": float(
            helpful_search_turns + evidence_search_turns + final_hop_search_turns
        ),
        "turn_credit/helpful_bridge_search_turns": float(helpful_search_turns),
        "turn_credit/evidence_bridge_search_turns": float(evidence_search_turns),
        "turn_credit/final_hop_attribute_search_turns": float(final_hop_search_turns),
        "turn_credit/early_answer_penalty_turns": float(
            early_answer_penalty_turns
        ),
        "turn_credit/missing_final_hop_penalty_turns": float(
            missing_final_hop_penalty_turns
        ),
        "turn_credit/final_answer_guard_penalty_turns": float(
            final_answer_guard_penalty_turns
        ),
        "turn_credit/credited_trajectories": float(credited_trajectories),
        "turn_credit/credited_tokens": float(credited_tokens),
    }


def merge_trainer_metrics(results: list[Any]) -> dict[str, float]:
    """Merge numeric metrics returned by PyTRIO forward/backward calls."""
    values: dict[str, list[float]] = defaultdict(list)
    rows = [
        {
            key: float(value)
            for key, value in dict(result.metrics).items()
            if isinstance(value, (int, float, np.number))
        }
        for result in results
    ]
    for row in rows:
        for key, value in row.items():
            values[key].append(value)
    merged: dict[str, float] = {}
    for key, items in values.items():
        if key == "opsd/mask_rate":
            selected = sum(row.get("opsd/masked_tokens", 0.0) for row in rows)
            total = sum(row.get("opsd/total_tokens", 0.0) for row in rows)
            merged[f"trainer/{key}"] = selected / total if total else 0.0
        elif key.endswith(("/gate_mean", "/signed_gap_mean")) or key in {
            "opsd/teacher_logprob_mean", "opsd/student_teacher_gap_mean",
            "opsd/positive_gap_fraction",
        }:
            category = key.split("/")[1]
            count_key = (
                f"opsd/{category}/selected_tokens"
                if category in {"search", "answer", "invalid"}
                else "opsd/masked_tokens"
            )
            weighted = [(row[key], row.get(count_key, 0.0)) for row in rows if key in row]
            weight = sum(count for _, count in weighted)
            merged[f"trainer/{key}"] = (
                sum(value * count for value, count in weighted) / weight
                if weight else mean(items)
            )
        elif key.endswith("/selected_tokens") or key in {
            "loss_mean", "loss/mean", "opsd/loss_mean",
            "opsd/masked_tokens", "opsd/selected_trajectories", "opsd/total_tokens",
        }:
            merged[f"trainer/{key}"] = sum(items)
        else:
            merged[f"trainer/{key}"] = mean(items)
    return merged


def pick_mean_loss_metric(metrics: dict[str, float]) -> float | None:
    """Return a trainer mean-loss metric when present."""
    for key in ("trainer/loss_mean", "trainer/loss/mean"):
        if key in metrics:
            return float(metrics[key])
    return None


def save_checkpoint(training_client: Any, name: str) -> None:
    """Save PyTRIO training state and sampler weights."""
    state = training_client.save_state(name=f"{name}-state").result()
    weights = training_client.save_weights_for_sampler(name=f"{name}-weights").result()
    print(f"Saved state: {state.path}")
    print(f"Saved sampler weights: {weights.path}")


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "to_numpy"):
        return value.to_numpy()
    return np.asarray(value)
