"""Analyze turn-level credit heuristics over persisted trajectory JSONL."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from search_r1_minilab.offline_diagnostics import (
    final_answer_from_record,
    load_records,
    tool_queries,
)
from search_r1_minilab.turn_credit import (
    detect_early_answer_risk,
    detect_missing_final_hop_risk,
    find_evidence_bridge_turns,
    find_final_hop_attribute_turns,
    find_helpful_bridge_shape_turns,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "eval_results" / "trajectories.jsonl"
DEFAULT_JSONL_OUTPUT = ROOT / "eval_results" / "turn_credit_analysis.jsonl"
DEFAULT_REPORT_OUTPUT = ROOT / "eval_results" / "turn_credit_analysis.md"
KEY_CASE_IDS = {
    "dev_174",
    "dev_2429",
    "dev_3741",
    "dev_4869",
    "test_2231",
    "test_7511",
}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--jsonl-output", type=Path, default=DEFAULT_JSONL_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    parser.add_argument("--title", default="Turn Credit Analysis")
    parser.add_argument("--max-cases-per-section", type=int, default=12)
    return parser.parse_args()


def analyze_record(record: dict[str, Any]) -> dict[str, Any]:
    """Return turn-credit analysis for one trajectory record."""
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    record_id = str(metadata.get("id") or record.get("id") or "")
    exact_match = record.get("exact_match") is True
    valid_format = record.get("valid_format") is True
    wrong_valid = valid_format and not exact_match
    turns = [turn for turn in record.get("turns") or [] if isinstance(turn, dict)]
    question = str(record.get("question") or "")
    answers = [str(answer) for answer in record.get("answers") or []]
    queries = tool_queries(record)
    final_answer = final_answer_from_record(record) or ""
    search_calls = int(record.get("search_calls") or 0)
    stop_reason = str(metadata.get("stop_reason") or "")

    v1_shape = find_helpful_bridge_shape_turns(events=turns, question=question)
    evidence = find_evidence_bridge_turns(
        events=turns,
        question=question,
        answers=answers,
    )
    final_hop = find_final_hop_attribute_turns(
        events=turns,
        question=question,
        answers=answers,
    )
    early_risk = detect_early_answer_risk(
        events=turns,
        question=question,
        queries=queries,
        final_answer=final_answer,
        search_calls=search_calls,
        stop_reason=stop_reason,
    )
    final_hop_risk = detect_missing_final_hop_risk(
        events=turns,
        question=question,
        queries=queries,
        final_answer=final_answer,
        search_calls=search_calls,
        stop_reason=stop_reason,
    )
    return {
        "id": record_id,
        "data_source": record.get("data_source"),
        "question": question,
        "answers": answers,
        "final_answer": final_answer,
        "exact_match": exact_match,
        "valid_format": valid_format,
        "search_calls": search_calls,
        "stop_reason": stop_reason,
        "queries": queries,
        "v1_shape_candidate_count": len(v1_shape),
        "v1_training_credit_count": len(v1_shape) if wrong_valid else 0,
        "v1_shape_candidates": [_match_to_dict(match) for match in v1_shape],
        "evidence_candidate_detected": bool(evidence),
        "evidence_candidate_count": len(evidence),
        "training_credit_applied": bool(evidence) and wrong_valid,
        "evidence_training_credit_count": len(evidence) if wrong_valid else 0,
        "evidence_candidates": [_match_to_dict(match) for match in evidence],
        "final_hop_candidate_detected": bool(final_hop),
        "final_hop_candidate_count": len(final_hop),
        "final_hop_training_credit_count": len(final_hop) if wrong_valid else 0,
        "final_hop_candidates": [_match_to_dict(match) for match in final_hop],
        "early_answer_penalty_applied": early_risk.risky and wrong_valid,
        "early_answer_penalty_reasons": list(early_risk.reasons),
        "missing_final_hop_penalty_applied": final_hop_risk.risky and wrong_valid,
        "missing_final_hop_penalty_reasons": list(final_hop_risk.reasons),
        "key_case": record_id in KEY_CASE_IDS,
    }


def analyze_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Analyze all records."""
    return [analyze_record(record) for record in records]


def build_summary(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Return aggregate counts and buckets."""
    total = len(items)
    wrong_valid = sum(item["valid_format"] and not item["exact_match"] for item in items)
    bucket_counts: Counter[str] = Counter()
    for item in items:
        bucket = "|".join(
            [
                str(item.get("data_source") or "unknown"),
                f"exact={item['exact_match']}",
                f"format={item['valid_format']}",
                f"search_calls={item['search_calls']}",
            ]
        )
        bucket_counts[bucket] += 1
    return {
        "total": total,
        "wrong_valid": wrong_valid,
        "v1_shape_candidate_turns": sum(
            item["v1_shape_candidate_count"] for item in items
        ),
        "v1_training_credit_turns": sum(
            item["v1_training_credit_count"] for item in items
        ),
        "evidence_candidate_turns": sum(
            item["evidence_candidate_count"] for item in items
        ),
        "evidence_training_credit_turns": sum(
            item["evidence_training_credit_count"] for item in items
        ),
        "final_hop_candidate_turns": sum(
            item["final_hop_candidate_count"] for item in items
        ),
        "final_hop_training_credit_turns": sum(
            item["final_hop_training_credit_count"] for item in items
        ),
        "evidence_candidate_records": sum(
            item["evidence_candidate_detected"] for item in items
        ),
        "final_hop_candidate_records": sum(
            item["final_hop_candidate_detected"] for item in items
        ),
        "training_credit_records": sum(
            item["training_credit_applied"]
            or (item["final_hop_candidate_detected"] and item["valid_format"] and not item["exact_match"])
            for item in items
        ),
        "early_answer_penalty_records": sum(
            item["early_answer_penalty_applied"] for item in items
        ),
        "missing_final_hop_penalty_records": sum(
            item["missing_final_hop_penalty_applied"] for item in items
        ),
        "bucket_counts": dict(sorted(bucket_counts.items())),
    }


def write_jsonl(items: list[dict[str, Any]], path: Path) -> None:
    """Write per-record analysis JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for item in items:
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")


def build_markdown_report(
    items: list[dict[str, Any]],
    *,
    title: str,
    max_cases_per_section: int,
) -> str:
    """Build a compact Markdown report."""
    summary = build_summary(items)
    lines = [
        f"# {title}",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for key, value in summary.items():
        if key == "bucket_counts":
            continue
        lines.append(f"| `{key}` | {value} |")
    lines.extend(["", "## Buckets", "", "| Bucket | Count |", "| --- | ---: |"])
    for bucket, count in summary["bucket_counts"].items():
        lines.append(f"| `{bucket}` | {count} |")
    sections = [
        ("Evidence Candidates", lambda item: item["evidence_candidate_detected"]),
        ("Final-Hop Attribute Candidates", lambda item: item["final_hop_candidate_detected"]),
        ("Training Evidence Credits", lambda item: item["training_credit_applied"]),
        (
            "Training Final-Hop Credits",
            lambda item: item["final_hop_candidate_detected"]
            and item["valid_format"]
            and not item["exact_match"],
        ),
        (
            "Early Answer Penalties",
            lambda item: item["early_answer_penalty_applied"],
        ),
        (
            "Missing Final-Hop Penalties",
            lambda item: item["missing_final_hop_penalty_applied"],
        ),
        ("Key Cases", lambda item: item["key_case"]),
    ]
    for heading, predicate in sections:
        cases = [item for item in items if predicate(item)][:max_cases_per_section]
        lines.extend(["", f"## {heading}", ""])
        if not cases:
            lines.append("_No cases found._")
            continue
        for index, item in enumerate(cases, start=1):
            lines.extend(_format_case(index, item))
    return "\n".join(lines) + "\n"


def main() -> None:
    """Run turn-credit analysis and write outputs."""
    args = parse_args()
    items = analyze_records(load_records(args.input))
    write_jsonl(items, args.jsonl_output)
    report = build_markdown_report(
        items,
        title=args.title,
        max_cases_per_section=args.max_cases_per_section,
    )
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(report, encoding="utf-8")
    print(json.dumps(build_summary(items), indent=2, sort_keys=True))
    print(f"wrote {len(items)} records to {args.jsonl_output}")
    print(f"wrote report to {args.report_output}")


def _match_to_dict(match: Any) -> dict[str, Any]:
    return {
        "turn_index": match.turn_index,
        "query": match.query,
        "label": match.label,
        "reasons": list(match.reasons),
    }


def _format_case(index: int, item: dict[str, Any]) -> list[str]:
    queries = "; ".join(f"`{query}`" for query in item["queries"]) or "_none_"
    evidence = "; ".join(
        f"`{candidate['query']}`" for candidate in item["evidence_candidates"]
    ) or "_none_"
    final_hop = "; ".join(
        f"`{candidate['query']}`" for candidate in item["final_hop_candidates"]
    ) or "_none_"
    return [
        f"### Case {index}: {item['id'] or 'unknown'}",
        "",
        f"- Source: `{item.get('data_source')}`",
        f"- Question: {item['question']}",
        f"- Gold: `{item['answers']}`",
        f"- Final answer: `{item['final_answer']}`",
        f"- Exact/format/search: `{item['exact_match']}` / `{item['valid_format']}` / `{item['search_calls']}`",
        f"- Queries: {queries}",
        f"- Evidence candidates: {evidence}",
        f"- Final-hop candidates: {final_hop}",
        f"- Training credit applied: `{item['training_credit_applied']}`",
        f"- Early answer penalty: `{item['early_answer_penalty_applied']}`",
        f"- Missing final-hop penalty: `{item['missing_final_hop_penalty_applied']}`",
        "",
    ]


if __name__ == "__main__":
    main()
