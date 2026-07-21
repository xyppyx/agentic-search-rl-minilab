"""Dataset helpers for Search-R1 MiniLab train and eval runs."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SearchExample:
    """One search QA example."""

    id: str
    question: str
    answers: list[str]
    data_source: str


def load_examples(path: str | Path, limit: int = 0) -> list[SearchExample]:
    """Load Search-R1 examples from JSONL."""
    examples: list[SearchExample] = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            answers = row.get("answers", [])
            if not isinstance(answers, list):
                answers = [answers]
            examples.append(
                SearchExample(
                    id=str(row.get("id") or f"example-{line_number}"),
                    question=str(row["question"]),
                    answers=[str(answer) for answer in answers],
                    data_source=str(row.get("data_source") or "minilab"),
                )
            )
            if limit > 0 and len(examples) >= limit:
                break
    if not examples:
        raise ValueError(f"no examples loaded from {path}")
    return examples


def shuffled_examples(path: str | Path, seed: int, limit: int = 0) -> list[SearchExample]:
    """Load examples and shuffle them deterministically."""
    examples = load_examples(path, limit=limit)
    random.Random(seed).shuffle(examples)
    return examples


def take_batch(
    examples: list[SearchExample],
    start: int,
    batch_size: int,
) -> list[SearchExample]:
    """Return a cyclic batch of examples."""
    if not examples:
        return []
    return [examples[(start + offset) % len(examples)] for offset in range(batch_size)]
