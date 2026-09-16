"""Turn-level credit 启发式规则。

本文件只负责“识别哪些 assistant turn 值得单独奖励/惩罚”。真正把识别结果写回
effective_advantage、生成 token-level advantage 和 OPSD mask 的逻辑在 training.py。
这些规则同时服务训练和离线分析，便于复盘 credit 命中是否符合预期。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from search_r1_minilab.diagnostics import FOLLOWUP_CUE_TOKENS, QUERY_STOPWORDS
from search_r1_minilab.rewards import normalize_answer


# 简单实体/数字/关系词规则用于轻量启发式，不依赖额外 NER 模型，保证 smoke 和
# 离线复盘都能本地可复现。
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
# 从原问题中识别“最后一跳需要查的属性类型”，例如死亡日期、国籍、父母、作者。
FINAL_HOP_ATTRIBUTE_CUES: dict[str, tuple[str, ...]] = {
    "date": (
        "birth",
        "born",
        "birthdate",
        "date",
        "death",
        "deathdate",
        "die",
        "died",
        "died earlier",
        "older",
        "younger",
        "year",
    ),
    "nationality": (
        "birthplace",
        "born in",
        "country",
        "national",
        "nationality",
    ),
    "founder": ("founded", "founder"),
    "family": (
        "father",
        "grandfather",
        "grandmother",
        "husband",
        "maternal",
        "mother",
        "parent",
        "parents",
        "paternal",
        "spouse",
        "wife",
    ),
    "creative_role": (
        "author",
        "composer",
        "director",
        "producer",
        "writer",
    ),
}
# 查询中出现这些词，才认为模型真的发起了对应 final-hop 属性搜索。
ATTRIBUTE_QUERY_TERMS: dict[str, set[str]] = {
    "date": {
        "birth",
        "birthdate",
        "born",
        "date",
        "death",
        "deathdate",
        "died",
        "year",
    },
    "nationality": {"birthplace", "born", "country", "national", "nationality"},
    "founder": {"founded", "founder", "organization", "company"},
    "family": {
        "father",
        "grandfather",
        "grandmother",
        "husband",
        "maternal",
        "mother",
        "parent",
        "parents",
        "paternal",
        "spouse",
        "wife",
    },
    "creative_role": {"author", "composer", "director", "producer", "writer"},
}
# 问题含这些 cue 且模型只搜一次就回答时，更可能是跳过了必要 follow-up。
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
    """命中某个 turn-credit detector 的搜索 turn。"""

    turn_index: int
    query: str
    label: str
    reasons: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class EarlyAnswerRisk:
    """final-answer turn 是否疑似过早停止。"""

    risky: bool
    reasons: tuple[str, ...] = field(default_factory=tuple)


def find_helpful_bridge_shape_turns(
    *,
    events: Iterable[dict[str, Any]],
    question: str,
) -> list[SearchTurnMatch]:
    """返回 v1 形状级 follow-up 搜索候选。

    该规则只看第二跳以后是否有新信息词、非重复 query、前一跳有非空结果；它是
    早期较宽松的 credit 规则。
    """
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
        # follow-up shape 表示 query 相比历史搜索引入了足够新词，并带关系词或
        # 问题外新实体/属性词。
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
    """返回把前一跳 bridge entity 连接到证据的搜索 turn。"""
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
            # evidence bridge 要同时满足三件事：query 命中前一跳实体、加入关系/消歧、
            # 当前 observation 也提供答案或关系证据。三者缺一不加 credit。
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


def find_final_hop_attribute_turns(
    *,
    events: Iterable[dict[str, Any]],
    question: str,
    answers: Iterable[str],
) -> list[SearchTurnMatch]:
    """返回补齐 final-hop 属性证据的搜索 turn。"""
    materialized = list(events)
    required_attributes = required_final_hop_attributes(question)
    if not required_attributes:
        return []

    matches: list[SearchTurnMatch] = []
    seen_queries: set[str] = set()
    previous_tool_event: dict[str, Any] | None = None
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
            matched_attributes = _matched_final_hop_attributes(
                attributes=required_attributes,
                query=turn.query,
                event=current_tool_event,
            )
            if (
                matched_attributes
                and _query_hits_bridge_entity(
                    query=turn.query,
                    previous_event=previous_tool_event,
                    question=question,
                )
                and _current_observation_has_evidence(
                    query_term_set=terms,
                    event=current_tool_event,
                    answers=list(answers),
                )
            ):
                # final-hop credit 比 evidence bridge 更收窄：必须查到问题要求的具体
                # 属性类型，避免奖励普通 biography/泛化搜索。
                matches.append(
                    SearchTurnMatch(
                        turn_index=turn.turn_index,
                        query=turn.query,
                        label="final_hop_attribute_search",
                        reasons=tuple(
                            f"final_hop_attribute:{attribute}"
                            for attribute in sorted(matched_attributes)
                        ),
                    )
                )
        if normalized_query:
            seen_queries.add(normalized_query)
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
    """判断错误答案是否疑似只搜一跳就过早回答。"""
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


def detect_missing_final_hop_risk(
    *,
    events: Iterable[dict[str, Any]],
    question: str,
    queries: Iterable[str],
    final_answer: str,
    search_calls: int,
    stop_reason: str,
) -> EarlyAnswerRisk:
    """判断 final answer 是否跳过了问题要求的 final-hop 属性查询。"""
    if search_calls <= 0 or stop_reason != "answer":
        return EarlyAnswerRisk(False)
    required_attributes = required_final_hop_attributes(question)
    if not required_attributes:
        return EarlyAnswerRisk(False)

    materialized = list(events)
    if not any(
        successful_nonempty_tool_event(event)
        for event in materialized
        if isinstance(event, dict) and event.get("role") == "tool"
    ):
        return EarlyAnswerRisk(False)

    covered = _covered_final_hop_attributes(
        attributes=required_attributes,
        question=question,
        queries=queries,
    )
    missing = required_attributes - covered
    if not missing:
        return EarlyAnswerRisk(False)
    # 如果最终答案本身能被 observation 中的属性文本支持，就不再判为缺失 final-hop。
    if _final_answer_is_supported_by_attribute_text(
        final_answer=final_answer,
        attributes=required_attributes,
        events=materialized,
    ):
        return EarlyAnswerRisk(False)
    return EarlyAnswerRisk(
        True,
        tuple(f"missing_final_hop_attribute:{attribute}" for attribute in sorted(missing)),
    )


def detect_final_answer_guard_risk(
    *,
    search_calls: int,
    stop_reason: str,
    valid_format: bool,
    exact_match: bool,
) -> EarlyAnswerRisk:
    """判断搜索后最后一轮是否需要因格式/未回答干净而被惩罚。"""
    if exact_match or search_calls <= 0:
        return EarlyAnswerRisk(False)
    if stop_reason == "max_search_calls" and not valid_format:
        return EarlyAnswerRisk(True, ("max_search_no_answer_after_search",))
    if stop_reason in {"answer", "invalid_format"} and not valid_format:
        return EarlyAnswerRisk(True, ("invalid_final_answer_format_after_search",))
    return EarlyAnswerRisk(False)


def required_final_hop_attributes(question: str) -> set[str]:
    """从问题文本中抽取所需 final-hop 属性类型。"""
    lowered = question.lower()
    return {
        attribute
        for attribute, cues in FINAL_HOP_ATTRIBUTE_CUES.items()
        if any(cue in lowered for cue in cues)
    }


def successful_nonempty_tool_event(event: dict[str, Any] | None) -> bool:
    """判断 tool event 是否成功且至少有一条搜索结果。"""
    if not isinstance(event, dict):
        return False
    if event.get("role") != "tool" or event.get("ok") is not True:
        return False
    items = event.get("items")
    return isinstance(items, list) and bool(items)


def normalize_query(query: str) -> str:
    """归一化 query，用于重复搜索判断。"""
    return " ".join(query.lower().split())


def query_terms(query: str) -> set[str]:
    """抽取 query-like 文本中的非停用词字母数字 term。"""
    normalized = "".join(
        char.lower() if char.isalnum() else " " for char in query
    )
    return {
        token
        for token in normalized.split()
        if len(token) > 2 and token not in QUERY_STOPWORDS
    }


def entity_spans(text: str) -> set[str]:
    """抽取简单的大写开头实体候选 span。"""
    spans = {match.strip() for match in ENTITY_PATTERN.findall(text or "")}
    return {span for span in spans if len(query_terms(span)) > 0}


def tool_event_text(event: dict[str, Any] | None) -> str:
    """拼接 tool event 中的 title/source/content/observation 文本。"""
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
    """从交替的 assistant/tool events 中抽取搜索 turn 及其紧随 observation。"""
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


def _matched_final_hop_attributes(
    *,
    attributes: set[str],
    query: str,
    event: dict[str, Any] | None,
) -> set[str]:
    query_term_set = query_terms(query)
    text_term_set = query_terms(tool_event_text(event))
    matched: set[str] = set()
    for attribute in attributes:
        attribute_terms = ATTRIBUTE_QUERY_TERMS.get(attribute, set())
        if query_term_set & attribute_terms:
            matched.add(attribute)
            continue
        if text_term_set & attribute_terms:
            matched.add(attribute)
            continue
        if attribute == "date" and NUMBER_PATTERN.search(tool_event_text(event)):
            matched.add(attribute)
    return matched


def _covered_final_hop_attributes(
    *,
    attributes: set[str],
    question: str,
    queries: Iterable[str],
) -> set[str]:
    query_text = " ".join(str(query) for query in queries)
    question_attribute_terms = _question_attribute_terms(attributes, question)
    covered: set[str] = set()
    for attribute in attributes:
        attribute_terms = (
            question_attribute_terms.get(attribute)
            or ATTRIBUTE_QUERY_TERMS.get(attribute, set())
        )
        query_terms_set = query_terms(query_text)
        if query_terms_set & attribute_terms:
            covered.add(attribute)
    return covered


def _question_attribute_terms(
    attributes: set[str],
    question: str,
) -> dict[str, set[str]]:
    """Return query terms that count as explicit coverage for each required attribute."""
    lowered = question.lower()
    terms_by_attribute: dict[str, set[str]] = {}
    for attribute in attributes:
        terms_by_attribute[attribute] = set(
            ATTRIBUTE_QUERY_TERMS.get(attribute, set())
        )
    if "date" in attributes:
        date_terms = {"date", "year"}
        if any(cue in lowered for cue in ("birth", "born", "older", "younger")):
            date_terms.update({"birth", "birthdate", "born"})
        if any(cue in lowered for cue in ("death", "die", "died", "died earlier")):
            date_terms.update({"death", "deathdate", "die", "died"})
        terms_by_attribute["date"] = date_terms
    return terms_by_attribute


def _final_answer_is_supported_by_attribute_text(
    *,
    final_answer: str,
    attributes: set[str],
    events: list[dict[str, Any]],
) -> bool:
    answer_terms = query_terms_from_answer(final_answer)
    if not answer_terms:
        return False
    for event in events:
        if not isinstance(event, dict) or event.get("role") != "tool":
            continue
        event_text = tool_event_text(event)
        event_terms = query_terms(event_text)
        if answer_terms <= event_terms and _matched_final_hop_attributes(
            attributes=attributes,
            query="",
            event=event,
        ):
            return True
    return False


def _has_early_answer_cue(question: str) -> bool:
    lowered = question.lower()
    return any(cue in lowered for cue in EARLY_ANSWER_CUES)
