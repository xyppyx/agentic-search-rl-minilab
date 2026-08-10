"""Build deterministic targeted eval sets from existing Search-R1 JSONL data."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUTS = [
    ROOT / "datasets" / "dev.jsonl",
    ROOT / "datasets" / "test.jsonl",
    ROOT / "datasets" / "train.jsonl",
]
DEFAULT_BRIDGE_OUTPUT = ROOT / "datasets" / "bridge_eval_150.jsonl"
DEFAULT_ALIAS_OUTPUT = ROOT / "datasets" / "alias_granularity_eval_80.jsonl"
DEFAULT_REPORT_OUTPUT = ROOT / "datasets" / "targeted_eval_manifest.md"

MULTIHOP_SOURCES = {"2wikimultihopqa", "hotpotqa", "musique", "bamboogle"}
BRIDGE_CUES = (
    "grandfather",
    "grandmother",
    "father",
    "mother",
    "parents",
    "parent",
    "spouse",
    "husband",
    "wife",
    "founder",
    "founded",
    "director",
    "producer",
    "writer",
    "written by",
    "author",
    "composer",
    "starring",
    "played by",
    "known as",
    "called",
    "located in",
    "capital of",
    "born in",
    "died in",
    "died earlier",
    "younger",
    "older",
    "larger",
    "smaller",
    "longer",
    "shorter",
    "studied",
    "educated",
    "alma mater",
    "length",
    "population",
    "country",
    "nationality",
    "team",
    "recorded by",
    "won by",
    "winner of",
    "who is",
    "which film",
    "which album",
    "which city",
)
ALIAS_CUES = (
    "when",
    "date",
    "year",
    "how many",
    "how long",
    "length",
    "km",
    "kilometer",
    "metre",
    "meter",
    "mile",
    "population",
    "number",
    "who",
    "which",
)
YEAR_PATTERN = re.compile(r"\b(1[5-9]\d{2}|20\d{2})\b")
NUMBER_PATTERN = re.compile(r"\b\d+(?:[.,]\d+)?\b")


@dataclass(frozen=True)
class Candidate:
    row: dict
    origin: str
    score: int
    tags: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", default=DEFAULT_INPUTS)
    parser.add_argument("--bridge-output", type=Path, default=DEFAULT_BRIDGE_OUTPUT)
    parser.add_argument("--alias-output", type=Path, default=DEFAULT_ALIAS_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    parser.add_argument("--bridge-size", type=int, default=150)
    parser.add_argument("--alias-size", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260722)
    return parser.parse_args()


def main() -> None:
    """Select examples and write JSONL plus a Markdown manifest."""
    args = parse_args()
    rows = list(load_rows(args.inputs))
    bridge = select_candidates(
        rows,
        size=args.bridge_size,
        seed=args.seed,
        scorer=score_bridge,
        excluded_ids=set(),
    )
    alias = select_candidates(
        rows,
        size=args.alias_size,
        seed=args.seed + 1,
        scorer=score_alias_granularity,
        excluded_ids={str(item.row["id"]) for item in bridge},
    )
    write_jsonl([item.row for item in bridge], args.bridge_output)
    write_jsonl([item.row for item in alias], args.alias_output)
    write_report(bridge, alias, args.report_output, args)
    print(
        json.dumps(
            {
                "bridge_output": str(args.bridge_output),
                "bridge_size": len(bridge),
                "alias_output": str(args.alias_output),
                "alias_size": len(alias),
                "report_output": str(args.report_output),
                "bridge_sources": dict(source_counts(bridge)),
                "alias_sources": dict(source_counts(alias)),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


def load_rows(paths: Iterable[Path]) -> Iterable[tuple[dict, str]]:
    """Yield normalized rows with their source filename."""
    seen: set[str] = set()
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                row_id = str(row.get("id") or f"{path.stem}_{line_number}")
                if row_id in seen:
                    continue
                answers = row.get("answers") or []
                if not isinstance(answers, list):
                    answers = [answers]
                normalized = {
                    "id": row_id,
                    "question": str(row["question"]),
                    "answers": [str(answer) for answer in answers],
                    "data_source": str(row.get("data_source") or "unknown"),
                }
                seen.add(row_id)
                yield normalized, path.stem


def select_candidates(
    rows: list[tuple[dict, str]],
    *,
    size: int,
    seed: int,
    scorer,
    excluded_ids: set[str],
) -> list[Candidate]:
    """Select deterministic candidates, balanced loosely by source."""
    rng = random.Random(seed)
    candidates: list[Candidate] = []
    for row, origin in rows:
        if row["id"] in excluded_ids:
            continue
        score, tags = scorer(row)
        if score <= 0:
            continue
        candidates.append(Candidate(row=row, origin=origin, score=score, tags=tuple(tags)))
    rng.shuffle(candidates)
    candidates.sort(
        key=lambda item: (
            -item.score,
            source_priority(item.origin),
            item.row["data_source"],
            item.row["id"],
        )
    )
    selected: list[Candidate] = []
    per_source_limit = max(8, size // 5)
    source_counter: Counter[str] = Counter()
    for item in candidates:
        source = item.row["data_source"]
        if source_counter[source] >= per_source_limit:
            continue
        selected.append(item)
        source_counter[source] += 1
        if len(selected) >= size:
            return selected
    for item in candidates:
        if item in selected:
            continue
        selected.append(item)
        if len(selected) >= size:
            return selected
    if len(selected) < size:
        raise ValueError(f"only selected {len(selected)} candidates, need {size}")
    return selected


def score_bridge(row: dict) -> tuple[int, list[str]]:
    """Score examples likely to require bridge/entity reasoning."""
    question = row["question"].lower()
    source = row["data_source"]
    if source not in MULTIHOP_SOURCES:
        return 0, []
    score = 0
    tags: list[str] = []
    score += 4
    tags.append("multihop_source")
    cue_hits = [cue for cue in BRIDGE_CUES if cue in question]
    if cue_hits:
        score += min(len(cue_hits), 5) * 2
        tags.extend(f"cue:{cue}" for cue in cue_hits[:6])
    if " of " in question and (" who " in question or " which " in question):
        score += 2
        tags.append("nested_of_clause")
    if "," in row["question"] or "?" in row["question"]:
        score += 1
    if len(row["question"].split()) >= 11:
        score += 1
        tags.append("long_question")
    return score, tags


def score_alias_granularity(row: dict) -> tuple[int, list[str]]:
    """Score examples likely to expose alias or granularity issues."""
    question = row["question"].lower()
    answers = row["answers"]
    all_answers = " ".join(answers)
    score = 0
    tags: list[str] = []
    if len(answers) >= 2:
        score += min(len(answers), 6)
        tags.append("multi_answer_aliases")
    if any(not answer.isascii() for answer in answers):
        score += 3
        tags.append("non_ascii_answer")
    if any("(" in answer or ")" in answer for answer in answers):
        score += 2
        tags.append("parenthetical_answer")
    if YEAR_PATTERN.search(all_answers) or YEAR_PATTERN.search(question):
        score += 2
        tags.append("date_or_year")
    if NUMBER_PATTERN.search(all_answers) or any(cue in question for cue in ALIAS_CUES[:8]):
        score += 1
        tags.append("numeric_or_granularity_cue")
    answer_lengths = [len(answer.split()) for answer in answers if answer.strip()]
    if answer_lengths and max(answer_lengths) - min(answer_lengths) >= 2:
        score += 2
        tags.append("answer_length_variants")
    if any(cue in question for cue in ALIAS_CUES):
        score += 1
        tags.append("question_granularity_cue")
    return score, tags


def source_priority(origin: str) -> int:
    """Prefer held-out and tiny dev examples before training examples."""
    return {"dev": 0, "test": 1, "train": 2}.get(origin, 3)


def source_counts(items: list[Candidate]) -> Counter[str]:
    """Count data_source values."""
    return Counter(item.row["data_source"] for item in items)


def write_jsonl(rows: list[dict], path: Path) -> None:
    """Write selected examples."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_report(
    bridge: list[Candidate],
    alias: list[Candidate],
    path: Path,
    args: argparse.Namespace,
) -> None:
    """Write a compact selection manifest."""
    lines = [
        "# Targeted Eval Manifest",
        "",
        f"Seed: `{args.seed}`",
        "",
        "## Outputs",
        "",
        f"- Bridge eval: `{args.bridge_output}` ({len(bridge)} examples)",
        f"- Alias/granularity eval: `{args.alias_output}` ({len(alias)} examples)",
        "",
        "## Source Counts",
        "",
        "### Bridge",
        "",
        "| Source | Count |",
        "| --- | ---: |",
    ]
    for source, count in source_counts(bridge).most_common():
        lines.append(f"| `{source}` | {count} |")
    lines.extend(["", "### Alias / Granularity", "", "| Source | Count |", "| --- | ---: |"])
    for source, count in source_counts(alias).most_common():
        lines.append(f"| `{source}` | {count} |")
    lines.extend(["", "## Sampled Cases", "", "### Bridge", ""])
    lines.extend(sample_lines(bridge[:12]))
    lines.extend(["", "### Alias / Granularity", ""])
    lines.extend(sample_lines(alias[:12]))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def sample_lines(items: list[Candidate]) -> list[str]:
    """Render sample rows."""
    lines = ["| ID | Source | Origin | Score | Tags | Question |", "| --- | --- | --- | ---: | --- | --- |"]
    for item in items:
        question = item.row["question"].replace("|", "\\|")
        tags = ", ".join(item.tags[:4]).replace("|", "\\|")
        lines.append(
            f"| `{item.row['id']}` | `{item.row['data_source']}` | `{item.origin}` | "
            f"{item.score} | {tags} | {question} |"
        )
    return lines


if __name__ == "__main__":
    main()
