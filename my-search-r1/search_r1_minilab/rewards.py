"""Answer-format and exact-match reward helpers."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


ANSWER_PATTERN = re.compile(r"^\s*Answer:\s*(.*?)\s*$", re.IGNORECASE | re.MULTILINE)
ARTICLE_PATTERN = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)


@dataclass(frozen=True)
class RewardResult:
    """Reward and diagnostic fields for one final answer."""

    reward: float
    valid_format: bool
    exact_match: bool
    answer: str | None


def normalize_answer(text: str) -> str:
    """Normalize answer text for Search-R1 exact match."""
    lowered = text.lower()
    without_punctuation = "".join(
        char for char in lowered if not unicodedata.category(char).startswith("P")
    )
    without_articles = ARTICLE_PATTERN.sub(" ", without_punctuation)
    return " ".join(without_articles.split())


def extract_answer(text: str) -> str | None:
    """Extract a single non-empty Answer line."""
    matches = ANSWER_PATTERN.findall(text)
    if len(matches) != 1:
        return None
    answer = matches[0].strip()
    return answer or None


def score_answer(text: str, references: list[str]) -> RewardResult:
    """Score final answer as 1.0 exact match, 0.0 wrong answer, or -0.1 invalid."""
    answer = extract_answer(text)
    if answer is None:
        return RewardResult(-0.1, False, False, None)
    normalized = normalize_answer(answer)
    exact_match = any(normalized == normalize_answer(reference) for reference in references)
    return RewardResult(float(exact_match), True, exact_match, answer)
