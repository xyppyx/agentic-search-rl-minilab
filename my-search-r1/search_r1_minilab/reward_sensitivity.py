"""Offline reward sensitivity analysis for persisted eval trajectories."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from search_r1_minilab.diagnostics import diagnose_record as diagnose_behavior
from search_r1_minilab.offline_diagnostics import (
    OfflineDiagnostic,
    diagnose_record as diagnose_offline,
    final_answer_from_record,
    load_records,
)
from search_r1_minilab.rewards import (
    RewardResult,
    RewardShapingConfig,
    apply_reward_shaping,
)


CONFIG_FIELDS = {
    "duplicate": "duplicate_query_penalty",
    "empty": "empty_result_penalty",
    "max_search": "max_search_no_answer_penalty",
    "verbose": "verbose_answer_penalty",
    "verbose_threshold": "verbose_answer_token_threshold",
    "bad_max_search": "bad_max_search_penalty",
    "date_granularity": "date_granularity_penalty",
    "multi_candidate": "multi_candidate_answer_penalty",
    "helpful_followup": "helpful_followup_bonus",
    "no_search": "no_search_penalty",
}


@dataclass(frozen=True)
class SensitivityConfig:
    """One named reward shaping configuration."""

    name: str
    reward_shaping: RewardShapingConfig

    def to_dict(self) -> dict[str, Any]:
        """Return config fields for reports."""
        return {
            "name": self.name,
            "duplicate_query_penalty": self.reward_shaping.duplicate_query_penalty,
            "empty_result_penalty": self.reward_shaping.empty_result_penalty,
            "max_search_no_answer_penalty": (
                self.reward_shaping.max_search_no_answer_penalty
            ),
            "verbose_answer_penalty": self.reward_shaping.verbose_answer_penalty,
            "verbose_answer_token_threshold": (
                self.reward_shaping.verbose_answer_token_threshold
            ),
            "bad_max_search_penalty": self.reward_shaping.bad_max_search_penalty,
            "date_granularity_penalty": self.reward_shaping.date_granularity_penalty,
            "multi_candidate_answer_penalty": (
                self.reward_shaping.multi_candidate_answer_penalty
            ),
            "helpful_followup_bonus": self.reward_shaping.helpful_followup_bonus,
            "no_search_penalty": self.reward_shaping.no_search_penalty,
        }


@dataclass(frozen=True)
class RescoreResult:
    """Reward sensitivity result for one record/config pair."""

    config_name: str
    record_id: str
    question: str
    final_answer: str | None
    exact_match: bool
    valid_format: bool
    base_reward: float
    final_reward: float
    reward_delta: float
    duplicate_query_penalty: float
    empty_result_penalty: float
    max_search_no_answer_penalty: float
    verbose_answer_penalty: float
    bad_max_search_penalty: float
    date_granularity_penalty: float
    multi_candidate_answer_penalty: float
    helpful_followup_bonus: float
    no_search_penalty: float
    possible_alias_match: bool
    answer_granularity_miss: bool
    missing_followup_query: bool
    helpful_followup_query: bool
    bad_max_search_loop: bool
    multi_candidate_answer: bool

    @property
    def penalized(self) -> bool:
        return self.reward_delta < 0.0

    @property
    def boosted(self) -> bool:
        return self.reward_delta > 0.0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "config_name": self.config_name,
            "id": self.record_id,
            "question": self.question,
            "final_answer": self.final_answer,
            "exact_match": self.exact_match,
            "valid_format": self.valid_format,
            "base_reward": self.base_reward,
            "final_reward": self.final_reward,
            "reward_delta": self.reward_delta,
            "duplicate_query_penalty": self.duplicate_query_penalty,
            "empty_result_penalty": self.empty_result_penalty,
            "max_search_no_answer_penalty": self.max_search_no_answer_penalty,
            "verbose_answer_penalty": self.verbose_answer_penalty,
            "bad_max_search_penalty": self.bad_max_search_penalty,
            "date_granularity_penalty": self.date_granularity_penalty,
            "multi_candidate_answer_penalty": self.multi_candidate_answer_penalty,
            "helpful_followup_bonus": self.helpful_followup_bonus,
            "no_search_penalty": self.no_search_penalty,
            "possible_alias_match": self.possible_alias_match,
            "answer_granularity_miss": self.answer_granularity_miss,
            "missing_followup_query": self.missing_followup_query,
            "helpful_followup_query": self.helpful_followup_query,
            "bad_max_search_loop": self.bad_max_search_loop,
            "multi_candidate_answer": self.multi_candidate_answer,
        }


@dataclass(frozen=True)
class SensitivitySummary:
    """Aggregate reward sensitivity counts for one config."""

    config_name: str
    total: int
    mean_base_reward: float
    mean_final_reward: float
    mean_delta: float
    penalized_count: int
    correct_penalized_count: int
    wrong_valid_penalized_count: int
    duplicate_penalized_count: int
    empty_penalized_count: int
    max_search_penalized_count: int
    verbose_penalized_count: int
    bad_max_search_penalized_count: int
    date_granularity_penalized_count: int
    multi_candidate_penalized_count: int
    no_search_penalized_count: int
    missing_followup_penalized_count: int
    helpful_followup_penalized_count: int
    bad_max_search_loop_penalized_count: int
    possible_alias_penalized_count: int
    answer_granularity_penalized_count: int
    multi_candidate_answer_penalized_count: int
    boosted_count: int
    helpful_followup_bonus_count: int
    correct_boosted_count: int
    wrong_valid_boosted_count: int
    missing_followup_boosted_count: int
    helpful_followup_boosted_count: int

    def to_dict(self) -> dict[str, Any]:
        """Return summary fields."""
        return {
            "config_name": self.config_name,
            "total": self.total,
            "mean_base_reward": self.mean_base_reward,
            "mean_final_reward": self.mean_final_reward,
            "mean_delta": self.mean_delta,
            "penalized_count": self.penalized_count,
            "correct_penalized_count": self.correct_penalized_count,
            "wrong_valid_penalized_count": self.wrong_valid_penalized_count,
            "duplicate_penalized_count": self.duplicate_penalized_count,
            "empty_penalized_count": self.empty_penalized_count,
            "max_search_penalized_count": self.max_search_penalized_count,
            "verbose_penalized_count": self.verbose_penalized_count,
            "bad_max_search_penalized_count": self.bad_max_search_penalized_count,
            "date_granularity_penalized_count": (
                self.date_granularity_penalized_count
            ),
            "multi_candidate_penalized_count": self.multi_candidate_penalized_count,
            "no_search_penalized_count": self.no_search_penalized_count,
            "missing_followup_penalized_count": (
                self.missing_followup_penalized_count
            ),
            "helpful_followup_penalized_count": (
                self.helpful_followup_penalized_count
            ),
            "bad_max_search_loop_penalized_count": (
                self.bad_max_search_loop_penalized_count
            ),
            "possible_alias_penalized_count": self.possible_alias_penalized_count,
            "answer_granularity_penalized_count": (
                self.answer_granularity_penalized_count
            ),
            "multi_candidate_answer_penalized_count": (
                self.multi_candidate_answer_penalized_count
            ),
            "boosted_count": self.boosted_count,
            "helpful_followup_bonus_count": self.helpful_followup_bonus_count,
            "correct_boosted_count": self.correct_boosted_count,
            "wrong_valid_boosted_count": self.wrong_valid_boosted_count,
            "missing_followup_boosted_count": self.missing_followup_boosted_count,
            "helpful_followup_boosted_count": self.helpful_followup_boosted_count,
        }


def default_sensitivity_configs() -> tuple[SensitivityConfig, ...]:
    """Return the built-in reward sensitivity configs."""
    return (
        SensitivityConfig("base_reward_v0", RewardShapingConfig()),
        SensitivityConfig(
            "penalty_v1",
            RewardShapingConfig(
                duplicate_query_penalty=0.05,
                empty_result_penalty=0.03,
                max_search_no_answer_penalty=0.05,
                verbose_answer_penalty=0.02,
                verbose_answer_token_threshold=8,
            ),
        ),
        SensitivityConfig(
            "penalty_v2_candidate",
            RewardShapingConfig(
                duplicate_query_penalty=0.03,
                empty_result_penalty=0.01,
            ),
        ),
        SensitivityConfig(
            "penalty_v2_no_empty",
            RewardShapingConfig(duplicate_query_penalty=0.03),
        ),
        SensitivityConfig(
            "penalty_v3_followup_aware",
            RewardShapingConfig(
                duplicate_query_penalty=0.03,
                empty_result_penalty=0.01,
                bad_max_search_penalty=0.01,
                date_granularity_penalty=0.05,
                multi_candidate_answer_penalty=0.02,
            ),
        ),
        SensitivityConfig(
            "reward_v4_followup_bonus",
            RewardShapingConfig(
                duplicate_query_penalty=0.03,
                empty_result_penalty=0.01,
                bad_max_search_penalty=0.01,
                date_granularity_penalty=0.05,
                multi_candidate_answer_penalty=0.02,
                helpful_followup_bonus=0.02,
            ),
        ),
        SensitivityConfig(
            "reward_v5_no_search_guard",
            RewardShapingConfig(
                duplicate_query_penalty=0.02,
                bad_max_search_penalty=0.005,
                date_granularity_penalty=0.05,
                multi_candidate_answer_penalty=0.02,
                no_search_penalty=0.03,
            ),
        ),
    )


def parse_sensitivity_config(value: str) -> SensitivityConfig:
    """Parse a CLI config spec in name:key=value,key=value form."""
    name, separator, fields_text = value.partition(":")
    if not separator or not name.strip() or not fields_text.strip():
        raise ValueError("config must use non-empty name:key=value form")

    fields: dict[str, Any] = {
        "duplicate_query_penalty": 0.0,
        "empty_result_penalty": 0.0,
        "max_search_no_answer_penalty": 0.0,
        "verbose_answer_penalty": 0.0,
        "verbose_answer_token_threshold": 0,
        "bad_max_search_penalty": 0.0,
        "date_granularity_penalty": 0.0,
        "multi_candidate_answer_penalty": 0.0,
        "helpful_followup_bonus": 0.0,
        "no_search_penalty": 0.0,
    }
    for item in fields_text.split(","):
        key, item_separator, raw_value = item.partition("=")
        if not item_separator or not key.strip() or not raw_value.strip():
            raise ValueError(f"invalid config item: {item!r}")
        key = key.strip()
        if key not in CONFIG_FIELDS:
            raise ValueError(f"unknown config field: {key}")
        target = CONFIG_FIELDS[key]
        if target == "verbose_answer_token_threshold":
            parsed: Any = int(raw_value)
        else:
            parsed = float(raw_value)
        if parsed < 0:
            raise ValueError(f"{key} must be non-negative")
        fields[target] = parsed
    return SensitivityConfig(name.strip(), RewardShapingConfig(**fields))


def load_configs(custom_specs: Iterable[str] = ()) -> tuple[SensitivityConfig, ...]:
    """Return default configs followed by parsed custom configs."""
    return (
        *default_sensitivity_configs(),
        *(parse_sensitivity_config(spec) for spec in custom_specs),
    )


def analyze_records(
    records: Iterable[dict[str, Any]],
    configs: Iterable[SensitivityConfig],
) -> list[RescoreResult]:
    """Run reward sensitivity analysis for records and configs."""
    materialized_records = list(records)
    diagnostics = [diagnose_offline(record) for record in materialized_records]
    results: list[RescoreResult] = []
    for config in configs:
        for record, offline in zip(materialized_records, diagnostics, strict=True):
            results.append(rescore_record(record, config, offline))
    return results


def analyze_jsonl(
    path: Path,
    configs: Iterable[SensitivityConfig],
) -> list[RescoreResult]:
    """Load trajectory JSONL and run sensitivity analysis."""
    return analyze_records(load_records(path), configs)


def rescore_record(
    record: dict[str, Any],
    config: SensitivityConfig,
    offline: OfflineDiagnostic | None = None,
) -> RescoreResult:
    """Rescore one persisted trajectory under one sensitivity config."""
    offline = offline or diagnose_offline(record)
    final_answer = offline.final_answer if offline.final_answer is not None else final_answer_from_record(record)
    base = base_reward_result(record, final_answer)
    components = apply_reward_shaping(
        base,
        diagnose_behavior(record),
        config.reward_shaping,
        answer_token_count=whitespace_token_count(final_answer),
        references=[str(answer) for answer in record.get("answers") or []],
    )
    return RescoreResult(
        config_name=config.name,
        record_id=offline.record_id,
        question=offline.question,
        final_answer=final_answer,
        exact_match=base.exact_match,
        valid_format=base.valid_format,
        base_reward=components.base_reward,
        final_reward=components.final_reward,
        reward_delta=components.final_reward - components.base_reward,
        duplicate_query_penalty=components.duplicate_query_penalty,
        empty_result_penalty=components.empty_result_penalty,
        max_search_no_answer_penalty=components.max_search_no_answer_penalty,
        verbose_answer_penalty=components.verbose_answer_penalty,
        bad_max_search_penalty=components.bad_max_search_penalty,
        date_granularity_penalty=components.date_granularity_penalty,
        multi_candidate_answer_penalty=components.multi_candidate_answer_penalty,
        helpful_followup_bonus=components.helpful_followup_bonus,
        no_search_penalty=components.no_search_penalty,
        possible_alias_match=offline.possible_alias_match,
        answer_granularity_miss=offline.answer_granularity_miss,
        missing_followup_query=offline.missing_followup_query,
        helpful_followup_query=offline.helpful_followup_query,
        bad_max_search_loop=offline.bad_max_search_loop,
        multi_candidate_answer=offline.multi_candidate_answer,
    )


def base_reward_result(record: dict[str, Any], final_answer: str | None) -> RewardResult:
    """Reconstruct the base reward from persisted eval fields."""
    valid_format = record.get("valid_format") is True
    exact_match = record.get("exact_match") is True
    if not valid_format:
        return RewardResult(-0.1, False, False, final_answer)
    return RewardResult(float(exact_match), True, exact_match, final_answer)


def summarize_results(results: Iterable[RescoreResult]) -> list[SensitivitySummary]:
    """Summarize sensitivity results by config."""
    grouped: dict[str, list[RescoreResult]] = {}
    for result in results:
        grouped.setdefault(result.config_name, []).append(result)
    return [
        _summarize_config(name, items)
        for name, items in grouped.items()
    ]


def build_summary_payload(
    configs: Iterable[SensitivityConfig],
    results: Iterable[RescoreResult],
) -> dict[str, Any]:
    """Return a JSON summary payload with configs and aggregate metrics."""
    materialized_configs = list(configs)
    materialized_results = list(results)
    return {
        "configs": [config.to_dict() for config in materialized_configs],
        "summaries": [summary.to_dict() for summary in summarize_results(materialized_results)],
    }


def write_results_jsonl(results: Iterable[RescoreResult], path: Path) -> None:
    """Write per-record sensitivity results as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for result in results:
            stream.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")


def write_summary_json(payload: dict[str, Any], path: Path) -> None:
    """Write summary payload as pretty JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_markdown_report(
    configs: Iterable[SensitivityConfig],
    results: Iterable[RescoreResult],
    *,
    title: str = "Reward Sensitivity Report",
    max_cases_per_config: int = 5,
) -> str:
    """Build a Markdown reward sensitivity report."""
    materialized_configs = list(configs)
    materialized_results = list(results)
    summaries = summarize_results(materialized_results)
    lines = [
        f"# {title}",
        "",
        "离线 verbose penalty 使用 final answer whitespace word count 近似 tokenizer token count。",
        "",
        "## Configs",
        "",
        "| Config | duplicate | empty | max_search | bad_max_search | date_granularity | multi_candidate | helpful_followup_bonus | no_search | verbose | verbose_threshold |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for config in materialized_configs:
        shaping = config.reward_shaping
        lines.append(
            f"| `{config.name}` | {shaping.duplicate_query_penalty:.4f} | "
            f"{shaping.empty_result_penalty:.4f} | "
            f"{shaping.max_search_no_answer_penalty:.4f} | "
            f"{shaping.bad_max_search_penalty:.4f} | "
            f"{shaping.date_granularity_penalty:.4f} | "
            f"{shaping.multi_candidate_answer_penalty:.4f} | "
            f"{shaping.helpful_followup_bonus:.4f} | "
            f"{shaping.no_search_penalty:.4f} | "
            f"{shaping.verbose_answer_penalty:.4f} | "
            f"{shaping.verbose_answer_token_threshold} |"
        )

    lines.extend(
        [
            "",
            "## Summary",
            "",
            "| Config | mean_base_reward | mean_final_reward | mean_delta | penalized | boosted | correct_penalized | correct_boosted | wrong_valid_penalized | wrong_valid_boosted | duplicate | empty | max_search | bad_max_search | date_penalty | multi_penalty | no_search | helpful_bonus | verbose | missing_followup_penalized | missing_followup_boosted | helpful_followup_penalized | helpful_followup_boosted | bad_max_loop | possible_alias | answer_granularity | multi_candidate_answer |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for summary in summaries:
        lines.append(
            f"| `{summary.config_name}` | {summary.mean_base_reward:.4f} | "
            f"{summary.mean_final_reward:.4f} | {summary.mean_delta:.4f} | "
            f"{summary.penalized_count} | {summary.boosted_count} | "
            f"{summary.correct_penalized_count} | {summary.correct_boosted_count} | "
            f"{summary.wrong_valid_penalized_count} | {summary.wrong_valid_boosted_count} | "
            f"{summary.duplicate_penalized_count} | {summary.empty_penalized_count} | "
            f"{summary.max_search_penalized_count} | "
            f"{summary.bad_max_search_penalized_count} | "
            f"{summary.date_granularity_penalized_count} | "
            f"{summary.multi_candidate_penalized_count} | "
            f"{summary.no_search_penalized_count} | "
            f"{summary.helpful_followup_bonus_count} | "
            f"{summary.verbose_penalized_count} | "
            f"{summary.missing_followup_penalized_count} | "
            f"{summary.missing_followup_boosted_count} | "
            f"{summary.helpful_followup_penalized_count} | "
            f"{summary.helpful_followup_boosted_count} | "
            f"{summary.bad_max_search_loop_penalized_count} | "
            f"{summary.possible_alias_penalized_count} | "
            f"{summary.answer_granularity_penalized_count} | "
            f"{summary.multi_candidate_answer_penalized_count} |"
        )

    lines.extend(["", "## Top Penalty Cases", ""])
    for config in materialized_configs:
        config_results = [
            result
            for result in materialized_results
            if result.config_name == config.name and result.penalized
        ]
        config_results.sort(key=lambda item: item.reward_delta)
        lines.extend(["", f"### {config.name}", ""])
        if not config_results:
            lines.append("_No penalized cases._")
            continue
        for index, result in enumerate(config_results[:max_cases_per_config], start=1):
            lines.extend(_format_case(index, result))
    lines.extend(["", "## Top Bonus Cases", ""])
    for config in materialized_configs:
        config_results = [
            result
            for result in materialized_results
            if result.config_name == config.name and result.helpful_followup_bonus > 0.0
        ]
        config_results.sort(
            key=lambda item: (item.helpful_followup_bonus, item.reward_delta),
            reverse=True,
        )
        lines.extend(["", f"### {config.name}", ""])
        if not config_results:
            lines.append("_No boosted cases._")
            continue
        for index, result in enumerate(config_results[:max_cases_per_config], start=1):
            lines.extend(_format_case(index, result))
    return "\n".join(lines) + "\n"


def whitespace_token_count(text: str | None) -> int:
    """Approximate answer token count for offline verbose penalty."""
    return len(str(text or "").split())


def _summarize_config(
    config_name: str,
    results: list[RescoreResult],
) -> SensitivitySummary:
    total = len(results)
    denominator = max(total, 1)
    penalized = [result for result in results if result.penalized]
    boosted = [result for result in results if result.boosted]
    return SensitivitySummary(
        config_name=config_name,
        total=total,
        mean_base_reward=sum(item.base_reward for item in results) / denominator,
        mean_final_reward=sum(item.final_reward for item in results) / denominator,
        mean_delta=sum(item.reward_delta for item in results) / denominator,
        penalized_count=len(penalized),
        correct_penalized_count=sum(item.exact_match for item in penalized),
        wrong_valid_penalized_count=sum(
            item.valid_format and not item.exact_match for item in penalized
        ),
        duplicate_penalized_count=sum(
            item.duplicate_query_penalty > 0.0 for item in penalized
        ),
        empty_penalized_count=sum(
            item.empty_result_penalty > 0.0 for item in penalized
        ),
        max_search_penalized_count=sum(
            item.max_search_no_answer_penalty > 0.0 for item in penalized
        ),
        verbose_penalized_count=sum(
            item.verbose_answer_penalty > 0.0 for item in penalized
        ),
        bad_max_search_penalized_count=sum(
            item.bad_max_search_penalty > 0.0 for item in penalized
        ),
        date_granularity_penalized_count=sum(
            item.date_granularity_penalty > 0.0 for item in penalized
        ),
        multi_candidate_penalized_count=sum(
            item.multi_candidate_answer_penalty > 0.0 for item in penalized
        ),
        no_search_penalized_count=sum(
            item.no_search_penalty > 0.0 for item in penalized
        ),
        missing_followup_penalized_count=sum(
            item.missing_followup_query for item in penalized
        ),
        helpful_followup_penalized_count=sum(
            item.helpful_followup_query for item in penalized
        ),
        bad_max_search_loop_penalized_count=sum(
            item.bad_max_search_loop for item in penalized
        ),
        possible_alias_penalized_count=sum(
            item.possible_alias_match for item in penalized
        ),
        answer_granularity_penalized_count=sum(
            item.answer_granularity_miss for item in penalized
        ),
        multi_candidate_answer_penalized_count=sum(
            item.multi_candidate_answer for item in penalized
        ),
        boosted_count=len(boosted),
        helpful_followup_bonus_count=sum(
            item.helpful_followup_bonus > 0.0 for item in results
        ),
        correct_boosted_count=sum(item.exact_match for item in boosted),
        wrong_valid_boosted_count=sum(
            item.valid_format and not item.exact_match for item in boosted
        ),
        missing_followup_boosted_count=sum(
            item.missing_followup_query for item in boosted
        ),
        helpful_followup_boosted_count=sum(
            item.helpful_followup_query for item in boosted
        ),
    )


def _format_case(index: int, result: RescoreResult) -> list[str]:
    penalties = {
        "duplicate": result.duplicate_query_penalty,
        "empty": result.empty_result_penalty,
        "max_search": result.max_search_no_answer_penalty,
        "bad_max_search": result.bad_max_search_penalty,
        "date_granularity": result.date_granularity_penalty,
        "multi_candidate": result.multi_candidate_answer_penalty,
        "no_search": result.no_search_penalty,
        "helpful_followup_bonus": -result.helpful_followup_bonus,
        "verbose": result.verbose_answer_penalty,
    }
    applied_items = [
        f"`{name}={value:.4f}`"
        for name, value in penalties.items()
        if value > 0.0
    ]
    if result.helpful_followup_bonus > 0.0:
        applied_items.append(f"`helpful_followup_bonus=+{result.helpful_followup_bonus:.4f}`")
    applied = ", ".join(applied_items)
    if not applied:
        applied = "_none_"
    return [
        f"#### Case {index}: {result.record_id or 'unknown'}",
        "",
        f"- Delta: `{result.reward_delta:.4f}`",
        f"- Base/final reward: `{result.base_reward:.4f}` -> `{result.final_reward:.4f}`",
        f"- Exact/format: `{result.exact_match}` / `{result.valid_format}`",
        f"- Final answer: `{result.final_answer}`",
        f"- Applied penalties: {applied}",
        f"- Missing follow-up: `{result.missing_followup_query}`",
        f"- Helpful follow-up: `{result.helpful_followup_query}`",
        f"- Bad max-search loop: `{result.bad_max_search_loop}`",
        f"- Possible alias: `{result.possible_alias_match}`",
        f"- Answer granularity miss: `{result.answer_granularity_miss}`",
        f"- Multi-candidate answer: `{result.multi_candidate_answer}`",
        f"- Question: {result.question}",
        "",
    ]
