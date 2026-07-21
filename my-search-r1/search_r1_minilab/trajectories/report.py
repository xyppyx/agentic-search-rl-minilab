"""Markdown trajectory reports for Search-R1 debugging."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

from search_r1_minilab.diagnostics import (
    BehaviorSummary,
    diagnose_record,
    summarize_diagnostics,
)
from search_r1_minilab.rewards import extract_answer


@dataclass(frozen=True)
class TrajectorySummary:
    """Aggregate trajectory counts and rates."""

    total: int
    correct: int
    wrong: int
    invalid_format: int
    tool_failure: int
    repeated_search: int
    average_reward: float | None
    average_search_calls: float
    behavior: BehaviorSummary


def classify_trajectory(record: dict[str, Any]) -> set[str]:
    """Classify one trajectory into report buckets."""
    labels: set[str] = set()
    diagnostics = diagnose_record(record)
    if record.get("exact_match") is True:
        labels.add("correct")
    elif record.get("valid_format") is False:
        labels.add("invalid_format")
    elif record.get("exact_match") is False:
        labels.add("wrong")

    if int(record.get("tool_failures") or 0) > 0 or _has_failed_tool_turn(record):
        labels.add("tool_failure")
    if diagnostics.repeated_search:
        labels.add("repeated_search")
    if diagnostics.direct_correct:
        labels.add("direct_correct")
    if diagnostics.searched_correct:
        labels.add("searched_correct")
    if diagnostics.searched_wrong:
        labels.add("searched_wrong")
    if diagnostics.direct_wrong:
        labels.add("direct_wrong")
    if diagnostics.searched_invalid_format:
        labels.add("searched_invalid_format")
    if diagnostics.empty_search:
        labels.add("empty_search")
    if diagnostics.max_search_no_answer:
        labels.add("max_search_no_answer")
    if diagnostics.too_many_search_no_gain:
        labels.add("too_many_search_no_gain")
    return labels


def summarize_trajectories(records: Iterable[dict[str, Any]]) -> TrajectorySummary:
    """Build aggregate counts for report headers."""
    materialized = list(records)
    label_counts: Counter[str] = Counter()
    reward_sum = 0.0
    reward_count = 0
    search_calls = 0
    for record in materialized:
        label_counts.update(classify_trajectory(record))
        reward = record.get("reward")
        if isinstance(reward, int | float):
            reward_sum += float(reward)
            reward_count += 1
        search_calls += int(record.get("search_calls") or 0)
    total = len(materialized)
    diagnostics = [diagnose_record(record) for record in materialized]
    return TrajectorySummary(
        total=total,
        correct=label_counts["correct"],
        wrong=label_counts["wrong"],
        invalid_format=label_counts["invalid_format"],
        tool_failure=label_counts["tool_failure"],
        repeated_search=label_counts["repeated_search"],
        average_reward=(reward_sum / reward_count if reward_count else None),
        average_search_calls=search_calls / max(total, 1),
        behavior=summarize_diagnostics(diagnostics),
    )


def build_markdown_report(
    records: Iterable[dict[str, Any]],
    *,
    title: str = "Trajectory Report",
    max_examples_per_section: int = 3,
) -> str:
    """Render trajectory records as a compact Markdown report."""
    materialized = list(records)
    summary = summarize_trajectories(materialized)
    lines = [
        f"# {title}",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Trajectories | {summary.total} |",
        f"| Correct | {summary.correct} |",
        f"| Wrong | {summary.wrong} |",
        f"| Invalid format | {summary.invalid_format} |",
        f"| Tool failure | {summary.tool_failure} |",
        f"| Repeated search | {summary.repeated_search} |",
        f"| Direct correct | {summary.behavior.direct_correct} |",
        f"| Searched correct | {summary.behavior.searched_correct} |",
        f"| Searched wrong | {summary.behavior.searched_wrong} |",
        f"| Direct wrong | {summary.behavior.direct_wrong} |",
        f"| Searched invalid format | {summary.behavior.searched_invalid_format} |",
        f"| Empty search trajectories | {summary.behavior.empty_search} |",
        f"| Empty observations | {summary.behavior.empty_observation_count} |",
        f"| Duplicate query trajectories | {summary.behavior.duplicate_query} |",
        f"| Duplicate query count | {summary.behavior.duplicate_query_count} |",
        f"| Max-search no-answer | {summary.behavior.max_search_no_answer} |",
        f"| Too many search no gain | {summary.behavior.too_many_search_no_gain} |",
        f"| Tool observations | {summary.behavior.tool_observation_count} |",
        f"| Pending tool calls | {summary.behavior.pending_tool_call_count} |",
        f"| Direct correct rate | {_format_rate(summary.behavior.direct_correct, summary.total)} |",
        f"| Searched correct rate | {_format_rate(summary.behavior.searched_correct, summary.total)} |",
        f"| Searched wrong rate | {_format_rate(summary.behavior.searched_wrong, summary.total)} |",
        f"| Empty observation rate | {_format_rate(summary.behavior.empty_observation_count, summary.behavior.tool_observation_count)} |",
        f"| Duplicate query rate | {_format_rate(summary.behavior.duplicate_query, summary.total)} |",
        f"| Max-search no-answer rate | {_format_rate(summary.behavior.max_search_no_answer, summary.total)} |",
        f"| Too many search no gain rate | {_format_rate(summary.behavior.too_many_search_no_gain, summary.total)} |",
        f"| Average reward | {_format_optional_float(summary.average_reward)} |",
        f"| Average search calls | {summary.average_search_calls:.2f} |",
        "",
        "## Buckets",
        "",
        *_bucket_table(materialized),
        "",
    ]
    for label, heading in [
        ("correct", "Correct Cases"),
        ("wrong", "Wrong Cases"),
        ("invalid_format", "Invalid Format Cases"),
        ("tool_failure", "Tool Failure Cases"),
        ("repeated_search", "Repeated Search Cases"),
        ("direct_correct", "Direct Correct Cases"),
        ("searched_correct", "Searched Correct Cases"),
        ("searched_wrong", "Searched Wrong Cases"),
        ("empty_search", "Empty Search Cases"),
        ("max_search_no_answer", "Max-Search No-Answer Cases"),
        ("too_many_search_no_gain", "Too Many Search No Gain Cases"),
    ]:
        lines.extend(
            _example_section(
                materialized,
                label=label,
                heading=heading,
                max_examples=max_examples_per_section,
            )
        )
    lines.extend(_group_comparison_section(materialized, max_examples_per_section))
    return "\n".join(lines).rstrip() + "\n"


def _bucket_table(records: list[dict[str, Any]]) -> list[str]:
    rows = ["| Bucket | Count |", "| --- | ---: |"]
    for label in [
        "correct",
        "wrong",
        "invalid_format",
        "tool_failure",
        "repeated_search",
        "direct_correct",
        "searched_correct",
        "searched_wrong",
        "direct_wrong",
        "searched_invalid_format",
        "empty_search",
        "max_search_no_answer",
        "too_many_search_no_gain",
    ]:
        rows.append(
            f"| {label} | {sum(1 for record in records if label in classify_trajectory(record))} |"
        )
    return rows


def _example_section(
    records: list[dict[str, Any]],
    *,
    label: str,
    heading: str,
    max_examples: int,
) -> list[str]:
    examples = [
        record for record in records if label in classify_trajectory(record)
    ][:max_examples]
    lines = [f"## {heading}", ""]
    if not examples:
        return [*lines, "_No examples._", ""]
    for index, record in enumerate(examples, start=1):
        lines.extend(_render_example(index, record))
    return lines


def _group_comparison_section(
    records: list[dict[str, Any]],
    max_examples: int,
) -> list[str]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        question = str(record.get("question") or "")
        groups[question].append(record)
    candidates = [
        group
        for group in groups.values()
        if len(group) > 1
        and any(record.get("advantage") is not None for record in group)
    ]
    lines = ["## Group Comparisons", ""]
    if not candidates:
        return [*lines, "_No grouped trajectories with advantage values._", ""]
    for index, group in enumerate(candidates[:max_examples], start=1):
        sorted_group = sorted(
            group,
            key=lambda record: float(record.get("advantage") or 0.0),
            reverse=True,
        )
        lines.append(f"### Group {index}: {_one_line(sorted_group[0].get('question'))}")
        lines.append("")
        lines.append("| Advantage | Reward | Exact match | Search calls | Final text |")
        lines.append("| ---: | ---: | --- | ---: | --- |")
        for record in sorted_group[:max_examples]:
            lines.append(
                "| "
                f"{_format_optional_float(record.get('advantage'))} | "
                f"{_format_optional_float(record.get('reward'))} | "
                f"{record.get('exact_match')} | "
                f"{int(record.get('search_calls') or 0)} | "
                f"{_escape_table(_final_assistant_text(record))} |"
            )
        lines.append("")
    return lines


def _render_example(index: int, record: dict[str, Any]) -> list[str]:
    labels = ", ".join(sorted(classify_trajectory(record))) or "uncategorized"
    diagnostics = diagnose_record(record)
    lines = [
        f"### Example {index}: {_one_line(record.get('question'))}",
        "",
        f"- Labels: `{labels}`",
        f"- Gold answers: `{_one_line(record.get('answers'))}`",
        f"- Extracted answer: `{extract_answer(_final_assistant_text(record))}`",
        f"- Stop reason: `{_metadata_value(record, 'stop_reason')}`",
        f"- Reward: `{record.get('reward')}`",
        f"- Advantage: `{record.get('advantage')}`",
        f"- Exact match: `{record.get('exact_match')}`",
        f"- Valid format: `{record.get('valid_format')}`",
        f"- Search calls: `{int(record.get('search_calls') or 0)}`",
        f"- Tool failures: `{int(record.get('tool_failures') or 0)}`",
        f"- Empty observations: `{diagnostics.empty_observation_count}`",
        f"- Duplicate queries: `{diagnostics.duplicate_query_count}`",
        f"- Pending tool calls: `{diagnostics.pending_tool_call_count}`",
        "",
        "```text",
        _final_assistant_text(record) or "[no assistant text]",
        "```",
        "",
    ]
    queries = _tool_queries(record)
    if queries:
        lines.extend(["Tool queries:", ""])
        for query in queries:
            lines.append(f"- `{query}`")
        lines.append("")
    return lines


def _has_failed_tool_turn(record: dict[str, Any]) -> bool:
    for turn in record.get("turns") or []:
        if not isinstance(turn, dict):
            continue
        if turn.get("role") == "tool" and turn.get("ok") is False:
            return True
    return False


def _tool_queries(record: dict[str, Any]) -> list[str]:
    queries: list[str] = []
    for turn in record.get("turns") or []:
        if not isinstance(turn, dict):
            continue
        tool_call = turn.get("tool_call")
        if isinstance(tool_call, dict) and isinstance(tool_call.get("query"), str):
            queries.append(tool_call["query"])
    return queries


def _metadata_value(record: dict[str, Any], key: str) -> Any:
    metadata = record.get("metadata")
    if isinstance(metadata, dict):
        return metadata.get(key)
    return None


def _final_assistant_text(record: dict[str, Any]) -> str:
    for turn in reversed(record.get("turns") or []):
        if isinstance(turn, dict) and turn.get("role") == "assistant":
            return str(turn.get("text") or "")
    return ""


def _one_line(value: Any, limit: int = 96) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _format_optional_float(value: Any) -> str:
    if isinstance(value, int | float):
        return f"{float(value):.4f}"
    return "-"


def _format_rate(numerator: int, denominator: int) -> str:
    return f"{(numerator / max(denominator, 1)):.4f}"


def _escape_table(value: str) -> str:
    return _one_line(value).replace("|", "\\|")
