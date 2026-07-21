"""Trajectory behavior diagnostics shared by reports, metrics, and rewards."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


FOLLOWUP_CUE_TOKENS = {
    "born",
    "birth",
    "child",
    "competition",
    "date",
    "death",
    "died",
    "director",
    "father",
    "founder",
    "grandfather",
    "grandmother",
    "husband",
    "mother",
    "place",
    "school",
    "studied",
    "study",
    "wife",
    "winner",
    "writer",
}
QUERY_STOPWORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "by",
    "for",
    "from",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "whose",
}


@dataclass(frozen=True)
class TrajectoryDiagnostics:
    """Derived behavior flags and counts for one trajectory."""

    direct_correct: bool
    searched_correct: bool
    searched_wrong: bool
    direct_wrong: bool
    searched_invalid_format: bool
    empty_observation_count: int
    duplicate_query_count: int
    max_search_no_answer: bool
    too_many_search_no_gain: bool
    helpful_followup_query: bool
    bad_max_search_loop: bool
    tool_observation_count: int
    pending_tool_call_count: int

    @property
    def empty_search(self) -> bool:
        return self.empty_observation_count > 0

    @property
    def repeated_search(self) -> bool:
        return self.duplicate_query_count > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "direct_correct": self.direct_correct,
            "searched_correct": self.searched_correct,
            "searched_wrong": self.searched_wrong,
            "direct_wrong": self.direct_wrong,
            "searched_invalid_format": self.searched_invalid_format,
            "empty_observation_count": self.empty_observation_count,
            "duplicate_query_count": self.duplicate_query_count,
            "max_search_no_answer": self.max_search_no_answer,
            "too_many_search_no_gain": self.too_many_search_no_gain,
            "helpful_followup_query": self.helpful_followup_query,
            "bad_max_search_loop": self.bad_max_search_loop,
            "tool_observation_count": self.tool_observation_count,
            "pending_tool_call_count": self.pending_tool_call_count,
        }


@dataclass(frozen=True)
class BehaviorSummary:
    """Aggregate behavior counts for trajectory collections."""

    total: int
    direct_correct: int
    searched_correct: int
    searched_wrong: int
    direct_wrong: int
    searched_invalid_format: int
    empty_search: int
    empty_observation_count: int
    duplicate_query: int
    duplicate_query_count: int
    max_search_no_answer: int
    too_many_search_no_gain: int
    helpful_followup_query: int
    bad_max_search_loop: int
    tool_observation_count: int
    pending_tool_call_count: int


def diagnose_record(record: dict[str, Any]) -> TrajectoryDiagnostics:
    """Diagnose behavior from a persisted trajectory record."""
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    return diagnose_fields(
        turns=record.get("turns") or [],
        search_calls=int(record.get("search_calls") or 0),
        exact_match=record.get("exact_match") is True,
        valid_format=record.get("valid_format") is True,
        stop_reason=str(metadata.get("stop_reason") or ""),
        question=str(record.get("question") or ""),
    )


def diagnose_fields(
    *,
    turns: Iterable[Any],
    search_calls: int,
    exact_match: bool,
    valid_format: bool,
    stop_reason: str,
    question: str = "",
    too_many_search_threshold: int = 3,
) -> TrajectoryDiagnostics:
    """Diagnose behavior from normalized trajectory fields."""
    materialized_turns = [turn for turn in turns if isinstance(turn, dict)]
    queries = _tool_queries(materialized_turns)
    normalized_queries = [_normalize_query(query) for query in queries if query.strip()]
    tool_observation_count = sum(
        1 for turn in materialized_turns if turn.get("role") == "tool"
    )
    empty_observation_count = sum(
        1 for turn in materialized_turns if _is_empty_observation(turn)
    )
    duplicate_query_count = len(normalized_queries) - len(set(normalized_queries))
    pending_tool_call_count = max(0, len(queries) - tool_observation_count)

    searched = search_calls > 0
    max_search_no_answer = stop_reason == "max_search_calls" and not valid_format
    too_many_search_no_gain = search_calls >= too_many_search_threshold and not exact_match
    helpful_followup_query = _has_helpful_followup_query(question, queries)
    bad_max_search_loop = (
        (max_search_no_answer or too_many_search_no_gain)
        and (duplicate_query_count > 0 or not helpful_followup_query)
    )
    return TrajectoryDiagnostics(
        direct_correct=(not searched and exact_match),
        searched_correct=(searched and exact_match),
        searched_wrong=(searched and valid_format and not exact_match),
        direct_wrong=(not searched and valid_format and not exact_match),
        searched_invalid_format=(searched and not valid_format),
        empty_observation_count=empty_observation_count,
        duplicate_query_count=duplicate_query_count,
        max_search_no_answer=max_search_no_answer,
        too_many_search_no_gain=too_many_search_no_gain,
        helpful_followup_query=helpful_followup_query,
        bad_max_search_loop=bad_max_search_loop,
        tool_observation_count=tool_observation_count,
        pending_tool_call_count=pending_tool_call_count,
    )


def summarize_diagnostics(
    diagnostics: Iterable[TrajectoryDiagnostics],
) -> BehaviorSummary:
    """Summarize per-trajectory diagnostics into aggregate counts."""
    items = list(diagnostics)
    return BehaviorSummary(
        total=len(items),
        direct_correct=sum(item.direct_correct for item in items),
        searched_correct=sum(item.searched_correct for item in items),
        searched_wrong=sum(item.searched_wrong for item in items),
        direct_wrong=sum(item.direct_wrong for item in items),
        searched_invalid_format=sum(item.searched_invalid_format for item in items),
        empty_search=sum(item.empty_search for item in items),
        empty_observation_count=sum(item.empty_observation_count for item in items),
        duplicate_query=sum(item.repeated_search for item in items),
        duplicate_query_count=sum(item.duplicate_query_count for item in items),
        max_search_no_answer=sum(item.max_search_no_answer for item in items),
        too_many_search_no_gain=sum(item.too_many_search_no_gain for item in items),
        helpful_followup_query=sum(item.helpful_followup_query for item in items),
        bad_max_search_loop=sum(item.bad_max_search_loop for item in items),
        tool_observation_count=sum(item.tool_observation_count for item in items),
        pending_tool_call_count=sum(item.pending_tool_call_count for item in items),
    )


def behavior_metrics(
    diagnostics: Iterable[TrajectoryDiagnostics],
) -> dict[str, float]:
    """Return rate metrics for training/eval logging."""
    summary = summarize_diagnostics(diagnostics)
    total = max(summary.total, 1)
    observations = max(summary.tool_observation_count, 1)
    return {
        "behavior/direct_correct_rate": summary.direct_correct / total,
        "behavior/searched_correct_rate": summary.searched_correct / total,
        "behavior/searched_wrong_rate": summary.searched_wrong / total,
        "behavior/empty_observation_rate": summary.empty_observation_count
        / observations,
        "behavior/duplicate_query_rate": summary.duplicate_query / total,
        "behavior/max_search_no_answer_rate": summary.max_search_no_answer / total,
        "behavior/too_many_search_no_gain_rate": summary.too_many_search_no_gain
        / total,
        "behavior/helpful_followup_query_rate": summary.helpful_followup_query / total,
        "behavior/bad_max_search_loop_rate": summary.bad_max_search_loop / total,
    }


def _tool_queries(turns: list[dict[str, Any]]) -> list[str]:
    queries: list[str] = []
    for turn in turns:
        tool_call = turn.get("tool_call")
        if isinstance(tool_call, dict) and isinstance(tool_call.get("query"), str):
            queries.append(tool_call["query"])
    return queries


def _is_empty_observation(turn: dict[str, Any]) -> bool:
    if turn.get("role") != "tool":
        return False
    items = turn.get("items")
    if not isinstance(items, list) or items:
        return False
    return turn.get("ok") is True


def _normalize_query(query: str) -> str:
    return " ".join(query.lower().split())


def _query_terms(query: str) -> set[str]:
    normalized = "".join(
        char.lower() if char.isalnum() else " " for char in query
    )
    return {
        token
        for token in normalized.split()
        if len(token) > 2 and token not in QUERY_STOPWORDS
    }


def _has_helpful_followup_query(question: str, queries: list[str]) -> bool:
    if len(queries) < 2:
        return False
    question_terms = _query_terms(question)
    seen = _query_terms(queries[0])
    for query in queries[1:]:
        terms = _query_terms(query)
        new_terms = terms - seen
        cue_overlap = terms & FOLLOWUP_CUE_TOKENS
        if len(new_terms) >= 2 and (cue_overlap or new_terms - question_terms):
            return True
        seen |= terms
    return False
