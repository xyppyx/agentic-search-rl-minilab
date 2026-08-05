"""PyTRIO GRPO training helpers for Search-R1 MiniLab."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pytrio as trio

from search_r1_minilab.diagnostics import behavior_metrics, diagnose_fields
from search_r1_minilab.rollout import Trajectory
from search_r1_minilab.rewards import extract_answer
from search_r1_minilab.turn_credit import (
    detect_early_answer_risk,
    find_evidence_bridge_turns,
    find_helpful_bridge_shape_turns,
)


MAX_TRAIN_CONTEXT_TOKENS = 8192
MAX_MICRO_BATCH_ITEMS = 32
MAX_MICRO_BATCH_PADDED_TOKENS = 64_000
TURN_CREDIT_POLICIES = {"none", "helpful_bridge", "evidence_bridge"}


@dataclass(frozen=True)
class TurnCreditConfig:
    """Optional turn-level training credit for useful search actions."""

    policy: str = "none"
    helpful_search_turn_bonus: float = 0.0
    evidence_search_turn_bonus: float = 0.0
    early_answer_turn_penalty: float = 0.0

    def __post_init__(self) -> None:
        if self.policy not in TURN_CREDIT_POLICIES:
            raise ValueError(
                "turn credit policy must be 'none', 'helpful_bridge', or 'evidence_bridge'"
            )
        if self.helpful_search_turn_bonus < 0.0:
            raise ValueError("helpful search turn bonus must be non-negative")
        if self.evidence_search_turn_bonus < 0.0:
            raise ValueError("evidence search turn bonus must be non-negative")
        if self.early_answer_turn_penalty < 0.0:
            raise ValueError("early answer turn penalty must be non-negative")


class TrainingDatum:
    """A PyTRIO datum plus its unpadded sequence length."""

    def __init__(
        self,
        datum: trio.Datum,
        num_tokens: int,
        *,
        reference_logprobs: list[float] | None = None,
    ) -> None:
        self.datum = datum
        self.num_tokens = num_tokens
        self.reference_logprobs = reference_logprobs


def build_datum(trajectory: Trajectory) -> TrainingDatum:
    """Convert one rollout trajectory into one PyTRIO training datum."""
    if not trajectory.turns:
        raise ValueError("cannot build a training datum without assistant turns")

    full_tokens: list[int] = []
    old_logprobs_by_token: list[float] = []
    advantages_by_token: list[float] = []
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

        full_tokens.extend(delta_observation)
        full_tokens.extend(turn.completion_tokens)
        turn_advantage = (
            trajectory.advantage
            if turn.effective_advantage is None
            else turn.effective_advantage
        )
        old_logprobs_by_token.extend([0.0] * len(delta_observation))
        old_logprobs_by_token.extend(turn.logprobs)
        advantages_by_token.extend([0.0] * len(delta_observation))
        advantages_by_token.extend([turn_advantage] * len(turn.completion_tokens))
        assistant_token_count += len(turn.completion_tokens)

    if assistant_token_count == 0:
        raise ValueError("cannot build a training datum without assistant tokens")
    if not (
        len(full_tokens)
        == len(old_logprobs_by_token)
        == len(advantages_by_token)
    ):
        raise ValueError("trajectory token/logprob/advantage lengths differ")

    input_tokens = full_tokens[:-1]
    target_tokens = full_tokens[1:]
    old_logprobs = old_logprobs_by_token[1:]
    advantages = advantages_by_token[1:]
    if not (
        len(input_tokens)
        == len(target_tokens)
        == len(old_logprobs)
        == len(advantages)
    ):
        raise ValueError("datum input/target/logprob/advantage lengths differ")
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
    return TrainingDatum(datum, len(input_tokens))


def build_training_datums(
    trajectories: list[Trajectory],
    turn_credit: TurnCreditConfig | None = None,
) -> list[TrainingDatum]:
    """Build datums for trajectories with non-zero group-relative advantages."""
    apply_turn_credit(trajectories, turn_credit or TurnCreditConfig())
    datums: list[TrainingDatum] = []
    for trajectory in trajectories:
        if any(turn.completion_tokens for turn in trajectory.turns):
            datum = build_datum(trajectory)
            if datum_loss_token_count(datum) > 0:
                datums.append(datum)
    return datums


def apply_turn_credit(
    trajectories: list[Trajectory],
    config: TurnCreditConfig,
) -> None:
    """Assign effective per-turn advantages for optional search-action credit."""
    for trajectory in trajectories:
        for turn in trajectory.turns:
            turn.effective_advantage = trajectory.advantage
            turn.credit_label = ""
            turn.credit_bonus = 0.0
            turn.credit_query = None
        _clear_event_turn_credit(trajectory)
        if config.policy == "none" or config.helpful_search_turn_bonus == 0.0:
            if config.policy != "evidence_bridge":
                continue
        if config.policy == "helpful_bridge":
            _apply_helpful_bridge_credit(trajectory, config.helpful_search_turn_bonus)
        elif config.policy == "evidence_bridge":
            _apply_evidence_bridge_credit(trajectory, config)


def _apply_helpful_bridge_credit(trajectory: Trajectory, bonus: float) -> None:
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


def _apply_turn_label(
    trajectory: Trajectory,
    turn_index: int,
    *,
    label: str,
    query: str | None,
    effective_advantage: float,
    bonus: float,
) -> None:
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


def datum_size(item: TrainingDatum) -> int:
    """Return the unpadded datum token length."""
    return item.num_tokens


def datum_loss_token_count(item: TrainingDatum) -> int:
    """Count target tokens that participate in policy loss."""
    advantages = _to_numpy(item.datum.loss_fn_inputs["advantages"])
    return int(np.count_nonzero(advantages))


def pack_micro_batches(datums: list[TrainingDatum]) -> list[list[TrainingDatum]]:
    """Pack variable-length datums with first-fit decreasing."""
    batches: list[list[TrainingDatum]] = []
    batch_max_tokens: list[int] = []
    for item in sorted(datums, key=datum_size, reverse=True):
        if item.num_tokens > MAX_TRAIN_CONTEXT_TOKENS:
            raise ValueError("single datum exceeds training context limit")
        for index, batch in enumerate(batches):
            next_items = len(batch) + 1
            next_max_tokens = max(batch_max_tokens[index], item.num_tokens)
            next_padded_tokens = next_items * next_max_tokens
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
    """Scale micro-batch advantages so accumulated means equal the global mean."""
    return [
        item.datum
        for item in weight_micro_batch_items_for_global_mean(micro_batch, total_samples)
    ]


def weight_micro_batch_items_for_global_mean(
    micro_batch: list[TrainingDatum],
    total_samples: int,
) -> list[TrainingDatum]:
    """Scale advantages and preserve local metadata for custom losses."""
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
            )
        )
    return weighted_items


def add_reference_logprobs(
    datums: list[TrainingDatum],
    reference_client: Any,
) -> list[TrainingDatum]:
    """Attach frozen reference-policy logprobs aligned to each shifted target token."""
    return [
        TrainingDatum(
            item.datum,
            item.num_tokens,
            reference_logprobs=compute_reference_logprobs(item, reference_client),
        )
        for item in datums
    ]


def compute_reference_logprobs(
    item: TrainingDatum,
    reference_client: Any,
) -> list[float]:
    """Compute reference logprobs for one already-shifted training datum."""
    input_tokens = [int(token) for token in item.datum.model_input.tolist()]
    target_tokens = [
        int(token) for token in _to_numpy(item.datum.loss_fn_inputs["target_tokens"])
    ]
    if not input_tokens or not target_tokens:
        raise ValueError("datum must contain input and target tokens")
    if len(input_tokens) != len(target_tokens):
        raise ValueError("datum input and target lengths differ")

    full_tokens = [*input_tokens, target_tokens[-1]]
    all_logprobs = reference_client.compute_logprobs(
        trio.ModelInput.from_ints(full_tokens)
    ).result()
    reference_logprobs = all_logprobs[1:]
    if len(reference_logprobs) != len(target_tokens):
        raise ValueError("reference logprob length does not match target tokens")

    advantages = _to_numpy(item.datum.loss_fn_inputs["advantages"])
    aligned: list[float] = []
    for logprob, advantage in zip(reference_logprobs, advantages, strict=True):
        if logprob is None:
            if float(advantage) != 0.0:
                raise ValueError("missing reference logprob for trainable token")
            aligned.append(0.0)
        else:
            aligned.append(float(logprob))
    return aligned


def build_custom_forward_datums(items: list[TrainingDatum]) -> list[trio.Datum]:
    """Strip custom metadata before PyTRIO cross-entropy forward."""
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
    reference_logprobs_list: list[list[float]],
    *,
    kl_coef: float,
    policy_ratio_clip: float = 0.0,
) -> Callable[[list[trio.Datum], list[Any]], tuple[Any, dict[str, float]]]:
    """Create a custom GRPO loss with sampled-token reference logprob drift penalty."""
    if kl_coef < 0.0:
        raise ValueError("kl_coef must be non-negative")
    if policy_ratio_clip < 0.0:
        raise ValueError("policy_ratio_clip must be non-negative")

    def loss_fn(
        data: list[trio.Datum],
        current_logprobs_list: list[Any],
    ) -> tuple[Any, dict[str, float]]:
        import torch

        if not (
            len(data)
            == len(current_logprobs_list)
            == len(sampling_logprobs_list)
            == len(advantages_list)
            == len(reference_logprobs_list)
        ):
            raise ValueError("GRPO KL loss got mismatched batch lengths")

        losses = []
        ratio_chunks = []
        kl_chunks = []
        clip_chunks = []
        train_tokens = 0
        denominator = 0

        for current_logprobs, old_values, advantage_values, reference_values in zip(
            current_logprobs_list,
            sampling_logprobs_list,
            advantages_list,
            reference_logprobs_list,
            strict=True,
        ):
            current = current_logprobs.float()
            device = current.device
            old = torch.as_tensor(old_values, dtype=torch.float32, device=device)
            advantages = torch.as_tensor(
                advantage_values,
                dtype=torch.float32,
                device=device,
            )
            reference = torch.as_tensor(
                reference_values,
                dtype=torch.float32,
                device=device,
            )
            if not (len(current) == len(old) == len(advantages) == len(reference)):
                raise ValueError("GRPO KL datum fields must have the same length")

            ratio = torch.exp(current - old)
            effective_ratio = ratio
            if policy_ratio_clip > 0.0:
                effective_ratio = torch.clamp(
                    ratio,
                    min=1.0 - policy_ratio_clip,
                    max=1.0 + policy_ratio_clip,
                )

            train_mask = advantages != 0.0
            objective = effective_ratio * advantages
            datum_loss = -objective.sum()
            if torch.any(train_mask) and kl_coef > 0.0:
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

        loss = torch.stack(losses).sum()
        metrics = {
            "loss_mean": float(loss.detach().item() / denominator)
            if denominator > 0
            else 0.0,
            "grpo_kl/coef": float(kl_coef),
            "grpo_kl/train_tokens": float(train_tokens),
        }
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
        return loss, metrics

    return loss_fn


def loss_input_float_lists(items: list[TrainingDatum], key: str) -> list[list[float]]:
    """Read one float loss-input field from a list of training datums."""
    values: list[list[float]] = []
    for item in items:
        values.append([float(value) for value in _to_numpy(item.datum.loss_fn_inputs[key])])
    return values


def mean(values: list[float]) -> float:
    """Return the arithmetic mean or zero for empty inputs."""
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
    early_answer_penalty_turns = 0
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
            elif turn.credit_label == "early_answer_missing_followup":
                early_answer_penalty_turns += 1
            credited_tokens += len(turn.completion_tokens)
            trajectory_has_credit = True
        if trajectory_has_credit:
            credited_trajectories += 1
    return {
        "turn_credit/policy": 1.0 if policy != "none" else 0.0,
        "turn_credit/helpful_search_turns": float(
            helpful_search_turns + evidence_search_turns
        ),
        "turn_credit/helpful_bridge_search_turns": float(helpful_search_turns),
        "turn_credit/evidence_bridge_search_turns": float(evidence_search_turns),
        "turn_credit/early_answer_penalty_turns": float(
            early_answer_penalty_turns
        ),
        "turn_credit/credited_trajectories": float(credited_trajectories),
        "turn_credit/credited_tokens": float(credited_tokens),
    }


def merge_trainer_metrics(results: list[Any]) -> dict[str, float]:
    """Merge numeric metrics returned by PyTRIO forward/backward calls."""
    values: dict[str, list[float]] = defaultdict(list)
    for result in results:
        for key, value in dict(result.metrics).items():
            if isinstance(value, (int, float, np.number)):
                values[key].append(float(value))
    merged: dict[str, float] = {}
    for key, items in values.items():
        if key in {"loss_mean", "loss/mean"}:
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
