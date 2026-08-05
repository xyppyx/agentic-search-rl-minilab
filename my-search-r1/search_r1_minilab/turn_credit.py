"""Turn-level credit heuristics shared by training and offline analysis."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from search_r1_minilab.diagnostics import FOLLOWUP_CUE_TOKENS, QUERY_STOPWORDS
from search_r1_minilab.rewards import normalize_answer


ENTITY_PATTERN = re.compile(r"\b[A-Z][A-Za-z0-9]*(?:[-'][A-Za-z0-9]+)?(?:\s+[A-Z][A-Za-z0-9]*(?:[-'][A-Za-z0-9]+)?){0,4}\b")
NUMBER_PATTERN = re.compile(r"\b\d+(?:\.\d+)?\b")
RELATION_TOKENS = FOLLOWUP_CUE_TOKENS | {
    "alias",
    "american",
    "attend",
    "attended",
    "filmmaker",
    "known",
    "length",
    "league",
    "name",
    "named",
    "national",
    "nationality",
    "parent",
    "parents",
    "school",
    "system",
    "university",
}
EARLY_ANSWER_CUES = (
    "also known as",
    "director",
    "father",
    "founder",
    "grandfather",
    "grandmother",
    "husband",
    "known as",
    "length",
    "how many",
    "km",
    "long",
    "maternal",
    "mother",
    "parent",
    "parents",
    "paternal",
    "studied",
    "university",
    "wife",
    "writer",
)


@dataclass(frozen=True)
class SearchTurnMatch:
    """A search turn that satisfies a turn-credit detector."""

    turn_index: int
    query: str
    label: str
    reasons: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class EarlyAnswerRisk:
    """A final-answer turn that likely stopped before needed follow-up search."""

    risky: bool
    reasons: tuple[str, ...] = field(default_factory=tuple)


def find_helpful_bridge_shape_turns(
    *,
    events: Iterable[dict[str, Any]],
    question: str,
) -> list[SearchTurnMatch]:
    """Return v1 shape-level follow-up search candidates."""
    matches: list[SearchTurnMatch] = []
    seen_queries: set[str] = set()
    seen_terms: set[str] = set()
    question_terms = query_terms(question)
    previous_tool_event: dict[str, Any] | None = None
    for turn in _iter_search_turns(list(events)):
        normalized_query = normalize_query(turn.query)
        terms = query_terms(turn.query)
        query_number = len(seen_queries) + 1
        is_duplicate = normalized_query in seen_queries
        new_terms = terms - seen_terms
        has_followup_shape = len(new_terms) >= 2 and (
            bool(terms & FOLLOWUP_CUE_TOKENS) or bool(new_terms - question_terms)
        )
        if (
            query_number >= 2
            and not is_duplicate
            and has_followup_shape
            and successful_nonempty_tool_event(previous_tool_event)
        ):
            matches.append(
                SearchTurnMatch(
                    turn_index=turn.turn_index,
                    query=turn.query,
                    label="helpful_bridge_search",
                    reasons=("query_sequence_has_helpful_followup",),
                )
            )
        if normalized_query:
            seen_queries.add(normalized_query)
        seen_terms |= terms
        previous_tool_event = turn.current_tool_event
    return matches


def find_evidence_bridge_turns(
    *,
    events: Iterable[dict[str, Any]],
    question: str,
    answers: Iterable[str],
) -> list[SearchTurnMatch]:
    """Return search turns that connect a prior bridge entity to evidence."""
    materialized = list(events)
    matches: list[SearchTurnMatch] = []
    seen_queries: set[str] = set()
    previous_tool_event: dict[str, Any] | None = None
    previous_terms: set[str] = set()
    for turn in _iter_search_turns(materialized):
        normalized_query = normalize_query(turn.query)
        terms = query_terms(turn.query)
        query_number = len(seen_queries) + 1
        is_duplicate = normalized_query in seen_queries
        current_tool_event = turn.current_tool_event
        if (
            query_number >= 2
            and not is_duplicate
            and successful_nonempty_tool_event(previous_tool_event)
            and successful_nonempty_tool_event(current_tool_event)
        ):
            reasons: list[str] = []
            bridge_hit = _query_hits_bridge_entity(
                query=turn.query,
                previous_event=previous_tool_event,
                question=question,
            )
            if bridge_hit:
                reasons.append("query_hits_previous_observation_entity")
            if _query_adds_relation_or_disambiguation(
                question=question,
                query_terms=terms,
                previous_terms=previous_terms,
            ):
                reasons.append("query_adds_relation_or_disambiguation")
            if _current_observation_has_evidence(
                query_term_set=terms,
                event=current_tool_event,
                answers=list(answers),
            ):
                reasons.append("current_observation_has_answer_or_relation_evidence")
            if len(reasons) == 3:
                matches.append(
                    SearchTurnMatch(
                        turn_index=turn.turn_index,
                        query=turn.query,
                        label="evidence_bridge_search",
                        reasons=tuple(reasons),
                    )
                )
        if normalized_query:
            seen_queries.add(normalized_query)
        previous_terms |= terms
        previous_tool_event = current_tool_event
    return matches


def detect_early_answer_risk(
    *,
    events: Iterable[dict[str, Any]],
    question: str,
    queries: Iterable[str],
    final_answer: str,
    search_calls: int,
    stop_reason: str,
) -> EarlyAnswerRisk:
    """Return whether a wrong answer likely stopped before required follow-up."""
    if search_calls > 1 or stop_reason != "answer":
        return EarlyAnswerRisk(False)
    if not _has_early_answer_cue(question):
        return EarlyAnswerRisk(False)
    materialized = list(events)
    first_tool_event = next(
        (event for event in materialized if event.get("role") == "tool"),
        None,
    )
    if not successful_nonempty_tool_event(first_tool_event):
        return EarlyAnswerRisk(False)
    combined = " ".join(
        [
            question,
            " ".join(str(query) for query in queries),
            tool_event_text(first_tool_event),
            final_answer,
        ]
    )
    entity_count = len(entity_spans(combined))
    number_count = len(set(NUMBER_PATTERN.findall(combined)))
    if entity_count + number_count < 3:
        return EarlyAnswerRisk(False)
    return EarlyAnswerRisk(
        True,
        (
            "single_search_multihop_or_role_binding_risk",
            "nonempty_first_observation_has_multiple_candidates",
        ),
    )


def successful_nonempty_tool_event(event: dict[str, Any] | None) -> bool:
    """Return whether an event is a successful non-empty tool observation."""
    if not isinstance(event, dict):
        return False
    if event.get("role") != "tool" or event.get("ok") is not True:
        return False
    items = event.get("items")
    return isinstance(items, list) and bool(items)


def normalize_query(query: str) -> str:
    """Normalize query text for duplicate checks."""
    return " ".join(query.lower().split())


def query_terms(query: str) -> set[str]:
    """Return non-stopword alphanumeric terms from query-like text."""
    normalized = "".join(
        char.lower() if char.isalnum() else " " for char in query
    )
    return {
        token
        for token in normalized.split()
        if len(token) > 2 and token not in QUERY_STOPWORDS
    }


def entity_spans(text: str) -> set[str]:
    """Extract simple title/capitalized entity-like spans."""
    spans = {match.strip() for match in ENTITY_PATTERN.findall(text or "")}
    return {span for span in spans if len(query_terms(span)) > 0}


def tool_event_text(event: dict[str, Any] | None) -> str:
    """Return text from a tool event, including item titles and content."""
    if not isinstance(event, dict):
        return ""
    parts: list[str] = []
    for item in event.get("items") or []:
        if not isinstance(item, dict):
            continue
        for key in ("title", "source", "content"):
            value = item.get(key)
            if value:
                parts.append(str(value))
    parts.append(str(event.get("observation") or event.get("text") or ""))
    return " ".join(parts)


@dataclass(frozen=True)
class _SearchTurn:
    turn_index: int
    query: str
    current_tool_event: dict[str, Any] | None


def _iter_search_turns(events: list[dict[str, Any]]) -> list[_SearchTurn]:
    turns: list[_SearchTurn] = []
    assistant_index = 0
    for index, event in enumerate(events):
        if event.get("role") != "assistant":
            continue
        turn_index = assistant_index
        assistant_index += 1
        tool_call = event.get("tool_call")
        if not isinstance(tool_call, dict):
            continue
        query = tool_call.get("query")
        if not isinstance(query, str):
            continue
        next_event = events[index + 1] if index + 1 < len(events) else None
        current_tool_event = (
            next_event
            if isinstance(next_event, dict) and next_event.get("role") == "tool"
            else None
        )
        turns.append(_SearchTurn(turn_index, query, current_tool_event))
    return turns


def _query_hits_previous_entity(query: str, event: dict[str, Any] | None) -> bool:
    query_term_set = query_terms(query)
    for span in _entity_candidates_from_tool_event(event):
        span_terms = query_terms(span)
        if span_terms and span_terms <= query_term_set:
            return True
        if (
            span_terms
            and span_terms & query_term_set
            and any(len(term) >= 6 for term in span_terms & query_term_set)
        ):
            return True
    return False


def _query_hits_bridge_entity(
    *,
    query: str,
    previous_event: dict[str, Any] | None,
    question: str,
) -> bool:
    if _query_hits_previous_entity(query, previous_event):
        return True
    query_term_set = query_terms(query)
    for span in entity_spans(question):
        span_terms = query_terms(span)
        if span_terms and span_terms <= query_term_set:
            return True
    return False


def _entity_candidates_from_tool_event(event: dict[str, Any] | None) -> set[str]:
    candidates = set(entity_spans(tool_event_text(event)))
    if isinstance(event, dict):
        for item in event.get("items") or []:
            if isinstance(item, dict) and item.get("title"):
                candidates.add(str(item["title"]))
    return candidates


def _query_adds_relation_or_disambiguation(
    *,
    question: str,
    query_terms: set[str],
    previous_terms: set[str],
) -> bool:
    if query_terms & RELATION_TOKENS:
        return True
    new_terms = query_terms - previous_terms - query_terms_from_question(question)
    return len(new_terms) >= 1


def query_terms_from_question(question: str) -> set[str]:
    """Return terms from the original question."""
    return query_terms(question)


def _current_observation_has_evidence(
    *,
    query_term_set: set[str],
    event: dict[str, Any] | None,
    answers: list[str],
) -> bool:
    text = tool_event_text(event)
    normalized_text = normalize_answer(text)
    for answer in answers:
        answer_terms = query_terms_from_answer(answer)
        if answer_terms and answer_terms <= set(normalized_text.split()):
            return True
    text_terms = query_terms(text)
    query_overlap = query_term_set & text_terms
    if bool(query_overlap) and bool(text_terms & RELATION_TOKENS):
        return True
    if len(query_overlap) >= 2 and (
        bool(NUMBER_PATTERN.findall(text)) or bool(entity_spans(text))
    ):
        return True
    return False


def query_terms_from_answer(answer: str) -> set[str]:
    """Return normalized answer terms suitable for evidence matching."""
    normalized = normalize_answer(str(answer or ""))
    return {
        token
        for token in normalized.split()
        if len(token) > 1 and token not in QUERY_STOPWORDS
    }


def _has_early_answer_cue(question: str) -> bool:
    lowered = question.lower()
    return any(cue in lowered for cue in EARLY_ANSWER_CUES)
