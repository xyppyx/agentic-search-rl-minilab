"""Offline diagnostics for persisted Search-R1 eval trajectories."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

from search_r1_minilab.diagnostics import diagnose_record as diagnose_behavior
from search_r1_minilab.rewards import extract_answer, normalize_answer
from search_r1_minilab.rewards import _multi_candidate_answer


YEAR_PATTERN = re.compile(r"\b(1[5-9]\d{2}|20\d{2})\b")
RELATION_CUES = (
    "maternal grandfather",
    "paternal grandfather",
    "grandfather",
    "grandmother",
    "director of",
    "written by",
    "written in part by",
    "writer",
    "recorded by",
    "person who",
    "film has the director",
    "whose director",
    "first gained",
    "delivered",
    "winner of",
    "younger",
    "older",
    "husband of",
    "wife of",
)


@dataclass(frozen=True)
class OfflineDiagnostic:
    """Offline labels for one eval trajectory."""

    record_id: str
    question: str
    answers: list[str]
    final_answer: str | None
    exact_match: bool
    valid_format: bool
    search_calls: int
    queries: list[str]
    possible_alias_match: bool
    answer_granularity_miss: bool
    missing_followup_query: bool
    helpful_followup_query: bool
    bad_max_search_loop: bool
    multi_candidate_answer: bool
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "id": self.record_id,
            "question": self.question,
            "answers": self.answers,
            "final_answer": self.final_answer,
            "exact_match": self.exact_match,
            "valid_format": self.valid_format,
            "search_calls": self.search_calls,
            "queries": self.queries,
            "possible_alias_match": self.possible_alias_match,
            "answer_granularity_miss": self.answer_granularity_miss,
            "missing_followup_query": self.missing_followup_query,
            "helpful_followup_query": self.helpful_followup_query,
            "bad_max_search_loop": self.bad_max_search_loop,
            "multi_candidate_answer": self.multi_candidate_answer,
            "reasons": self.reasons,
        }


@dataclass(frozen=True)
class OfflineDiagnosticSummary:
    """Aggregate counts for offline diagnostic labels."""

    total: int
    wrong_valid: int
    possible_alias_match: int
    answer_granularity_miss: int
    missing_followup_query: int
    helpful_followup_query: int
    bad_max_search_loop: int
    multi_candidate_answer: int

    def to_dict(self) -> dict[str, Any]:
        """Return counts and rates."""
        total = max(self.total, 1)
        wrong_valid = max(self.wrong_valid, 1)
        return {
            "total": self.total,
            "wrong_valid": self.wrong_valid,
            "possible_alias_match": self.possible_alias_match,
            "answer_granularity_miss": self.answer_granularity_miss,
            "missing_followup_query": self.missing_followup_query,
            "helpful_followup_query": self.helpful_followup_query,
            "bad_max_search_loop": self.bad_max_search_loop,
            "multi_candidate_answer": self.multi_candidate_answer,
            "possible_alias_match_rate": self.possible_alias_match / total,
            "answer_granularity_miss_rate": self.answer_granularity_miss / total,
            "missing_followup_query_rate": self.missing_followup_query / total,
            "helpful_followup_query_rate": self.helpful_followup_query / total,
            "bad_max_search_loop_rate": self.bad_max_search_loop / total,
            "multi_candidate_answer_rate": self.multi_candidate_answer / total,
            "possible_alias_match_wrong_valid_rate": self.possible_alias_match
            / wrong_valid,
            "answer_granularity_miss_wrong_valid_rate": self.answer_granularity_miss
            / wrong_valid,
            "missing_followup_query_wrong_valid_rate": self.missing_followup_query
            / wrong_valid,
            "multi_candidate_answer_wrong_valid_rate": self.multi_candidate_answer
            / wrong_valid,
        }


def load_records(path: Path) -> list[dict[str, Any]]:
    """Load non-summary trajectory records from JSONL."""
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON in {path}:{line_number}") from error
            if not isinstance(record, dict):
                raise ValueError(f"record in {path}:{line_number} must be an object")
            if record.get("type") != "summary":
                records.append(record)
    return records


def diagnose_record(record: dict[str, Any]) -> OfflineDiagnostic:
    """Derive offline diagnostic labels from one persisted trajectory."""
    final_answer = final_answer_from_record(record)
    answers = [str(answer) for answer in record.get("answers") or []]
    exact_match = record.get("exact_match") is True
    valid_format = record.get("valid_format") is True
    search_calls = int(record.get("search_calls") or 0)
    queries = tool_queries(record)
    reasons: list[str] = []

    possible_alias_match = False
    answer_granularity_miss = False
    missing_followup_query = False
    multi_candidate_answer = False
    if valid_format and not exact_match and final_answer:
        possible_alias_match = _possible_alias_match(final_answer, answers)
        if possible_alias_match:
            reasons.append("final_answer_overlaps_gold_alias_or_spelling_variant")

        answer_granularity_miss = _answer_granularity_miss(final_answer, answers)
        if answer_granularity_miss:
            reasons.append("final_answer_is_less_specific_than_gold")

        missing_followup_query = _missing_followup_query(record, final_answer)
        if missing_followup_query:
            reasons.append("single_search_multihop_or_role_binding_risk")

        multi_candidate_answer = _multi_candidate_answer(final_answer, answers)
        if multi_candidate_answer:
            reasons.append("final_answer_contains_multiple_candidates")

    behavior = diagnose_behavior(record)
    if behavior.helpful_followup_query:
        reasons.append("query_sequence_has_helpful_followup")
    if behavior.bad_max_search_loop:
        reasons.append("bad_max_search_loop")

    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    record_id = str(metadata.get("id") or record.get("id") or "")
    return OfflineDiagnostic(
        record_id=record_id,
        question=str(record.get("question") or ""),
        answers=answers,
        final_answer=final_answer,
        exact_match=exact_match,
        valid_format=valid_format,
        search_calls=search_calls,
        queries=queries,
        possible_alias_match=possible_alias_match,
        answer_granularity_miss=answer_granularity_miss,
        missing_followup_query=missing_followup_query,
        helpful_followup_query=behavior.helpful_followup_query,
        bad_max_search_loop=behavior.bad_max_search_loop,
        multi_candidate_answer=multi_candidate_answer,
        reasons=reasons,
    )


def diagnose_records(records: Iterable[dict[str, Any]]) -> list[OfflineDiagnostic]:
    """Diagnose a collection of trajectory records."""
    return [diagnose_record(record) for record in records]


def summarize_diagnostics(
    diagnostics: Iterable[OfflineDiagnostic],
) -> OfflineDiagnosticSummary:
    """Aggregate offline diagnostic counts."""
    items = list(diagnostics)
    wrong_valid = [
        item for item in items if item.valid_format and not item.exact_match
    ]
    return OfflineDiagnosticSummary(
        total=len(items),
        wrong_valid=len(wrong_valid),
        possible_alias_match=sum(item.possible_alias_match for item in items),
        answer_granularity_miss=sum(item.answer_granularity_miss for item in items),
        missing_followup_query=sum(item.missing_followup_query for item in items),
        helpful_followup_query=sum(item.helpful_followup_query for item in items),
        bad_max_search_loop=sum(item.bad_max_search_loop for item in items),
        multi_candidate_answer=sum(item.multi_candidate_answer for item in items),
    )


def build_markdown_report(
    diagnostics: list[OfflineDiagnostic],
    *,
    title: str = "Offline Diagnostic Report",
    max_cases_per_section: int = 8,
) -> str:
    """Build a compact Markdown report."""
    summary = summarize_diagnostics(diagnostics).to_dict()
    lines = [
        f"# {title}",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for key in [
        "total",
        "wrong_valid",
        "possible_alias_match",
        "answer_granularity_miss",
        "missing_followup_query",
        "helpful_followup_query",
        "bad_max_search_loop",
        "multi_candidate_answer",
        "possible_alias_match_wrong_valid_rate",
        "answer_granularity_miss_wrong_valid_rate",
        "missing_followup_query_wrong_valid_rate",
        "multi_candidate_answer_wrong_valid_rate",
    ]:
        value = summary[key]
        text = f"{value:.4f}" if isinstance(value, float) else str(value)
        lines.append(f"| `{key}` | {text} |")

    sections = [
        ("Possible Alias Match", "possible_alias_match"),
        ("Answer Granularity Miss", "answer_granularity_miss"),
        ("Missing Follow-Up Query", "missing_followup_query"),
        ("Helpful Follow-Up Query", "helpful_followup_query"),
        ("Bad Max-Search Loop", "bad_max_search_loop"),
        ("Multi-Candidate Answer", "multi_candidate_answer"),
    ]
    for heading, field_name in sections:
        cases = [
            item for item in diagnostics if bool(getattr(item, field_name))
        ][:max_cases_per_section]
        lines.extend(["", f"## {heading}", ""])
        if not cases:
            lines.append("_No cases found._")
            continue
        for index, item in enumerate(cases, start=1):
            lines.extend(_format_case(index, item))
    return "\n".join(lines) + "\n"


def write_diagnostic_jsonl(
    diagnostics: Iterable[OfflineDiagnostic],
    path: Path,
) -> None:
    """Write per-record diagnostic labels as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for item in diagnostics:
            stream.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")


def final_answer_from_record(record: dict[str, Any]) -> str | None:
    """Extract the final Answer line from a trajectory record."""
    text = _final_assistant_text(record)
    return extract_answer(text) if text else None


def tool_queries(record: dict[str, Any]) -> list[str]:
    """Return tool search queries in trajectory order."""
    queries: list[str] = []
    for turn in record.get("turns") or []:
        if not isinstance(turn, dict):
            continue
        tool_call = turn.get("tool_call")
        if isinstance(tool_call, dict) and isinstance(tool_call.get("query"), str):
            queries.append(tool_call["query"])
    return queries


def _final_assistant_text(record: dict[str, Any]) -> str:
    turns = record.get("turns") or []
    for turn in reversed(turns):
        if (
            isinstance(turn, dict)
            and turn.get("role") == "assistant"
            and turn.get("parsed_kind") != "tool"
        ):
            return str(turn.get("text") or "")
    return ""


def _possible_alias_match(answer: str, references: list[str]) -> bool:
    normalized_answer = normalize_answer(answer)
    if not normalized_answer:
        return False
    if not any(char.isalpha() for char in normalized_answer):
        return False
    for reference in references:
        normalized_reference = normalize_answer(reference)
        if not normalized_reference or normalized_answer == normalized_reference:
            continue
        if not any(char.isalpha() for char in normalized_reference):
            continue
        if _token_subset(normalized_answer, normalized_reference):
            return True
        if SequenceMatcher(None, normalized_answer, normalized_reference).ratio() >= 0.82:
            return True
    return False


def _answer_granularity_miss(answer: str, references: list[str]) -> bool:
    normalized_answer = normalize_answer(answer)
    answer_years = set(YEAR_PATTERN.findall(answer))
    if normalized_answer and answer_years and normalized_answer in answer_years:
        for reference in references:
            reference_years = set(YEAR_PATTERN.findall(reference))
            if answer_years & reference_years and normalize_answer(reference) != normalized_answer:
                return True
    return False


def _missing_followup_query(record: dict[str, Any], answer: str) -> bool:
    if int(record.get("search_calls") or 0) != 1:
        return False
    question = str(record.get("question") or "")
    if not _has_relation_cue(question):
        return False
    query = " ".join(tool_queries(record))
    observation = _observation_text(record)
    if not observation:
        return False
    combined = f"{question} {query} {observation} {answer}"
    return _named_entity_like_count(combined) >= 3


def _has_relation_cue(question: str) -> bool:
    lowered = question.lower()
    return any(cue in lowered for cue in RELATION_CUES)


def _observation_text(record: dict[str, Any]) -> str:
    parts: list[str] = []
    for turn in record.get("turns") or []:
        if isinstance(turn, dict) and turn.get("role") == "tool":
            parts.append(str(turn.get("observation") or turn.get("text") or ""))
    return " ".join(parts)


def _named_entity_like_count(text: str) -> int:
    matches = re.findall(r"\b[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3}\b", text)
    normalized = {match.strip() for match in matches}
    return len(normalized)


def _token_subset(left: str, right: str) -> bool:
    left_tokens = set(left.split())
    right_tokens = set(right.split())
    if not left_tokens or not right_tokens:
        return False
    if min(len(left_tokens), len(right_tokens)) == 1:
        return left_tokens <= right_tokens or right_tokens <= left_tokens
    overlap = left_tokens & right_tokens
    return (
        overlap == left_tokens
        or overlap == right_tokens
        or len(overlap) / min(len(left_tokens), len(right_tokens)) >= 0.8
    )


def _format_case(index: int, item: OfflineDiagnostic) -> list[str]:
    queries = "; ".join(f"`{query}`" for query in item.queries) or "_none_"
    reasons = ", ".join(f"`{reason}`" for reason in item.reasons) or "_none_"
    return [
        f"### Case {index}: {item.record_id or 'unknown'}",
        "",
        f"- Question: {item.question}",
        f"- Gold: `{item.answers}`",
        f"- Final answer: `{item.final_answer}`",
        f"- Search calls: `{item.search_calls}`",
        f"- Queries: {queries}",
        f"- Reasons: {reasons}",
        "",
    ]
