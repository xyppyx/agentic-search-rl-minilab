"""Answer-format and exact-match reward helpers."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from search_r1_minilab.diagnostics import TrajectoryDiagnostics


ANSWER_PATTERN = re.compile(r"^\s*Answer:\s*(.*?)\s*$", re.IGNORECASE | re.MULTILINE)
ARTICLE_PATTERN = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)


@dataclass(frozen=True)
class RewardResult:
    """Reward and diagnostic fields for one final answer."""

    reward: float
    valid_format: bool
    exact_match: bool
    answer: str | None


@dataclass(frozen=True)
class RewardShapingConfig:
    """Optional trajectory-level reward penalties."""

    duplicate_query_penalty: float = 0.0
    empty_result_penalty: float = 0.0
    max_search_no_answer_penalty: float = 0.0
    verbose_answer_penalty: float = 0.0
    verbose_answer_token_threshold: int = 0
    bad_max_search_penalty: float = 0.0
    date_granularity_penalty: float = 0.0
    multi_candidate_answer_penalty: float = 0.0

    def __post_init__(self) -> None:
        penalties = {
            "duplicate_query_penalty": self.duplicate_query_penalty,
            "empty_result_penalty": self.empty_result_penalty,
            "max_search_no_answer_penalty": self.max_search_no_answer_penalty,
            "verbose_answer_penalty": self.verbose_answer_penalty,
            "bad_max_search_penalty": self.bad_max_search_penalty,
            "date_granularity_penalty": self.date_granularity_penalty,
            "multi_candidate_answer_penalty": self.multi_candidate_answer_penalty,
        }
        for name, value in penalties.items():
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if self.verbose_answer_token_threshold < 0:
            raise ValueError("verbose_answer_token_threshold must be non-negative")


@dataclass(frozen=True)
class RewardComponents:
    """Base reward, applied penalties, and final reward."""

    base_reward: float
    duplicate_query_penalty: float
    empty_result_penalty: float
    max_search_no_answer_penalty: float
    verbose_answer_penalty: float
    bad_max_search_penalty: float
    date_granularity_penalty: float
    multi_candidate_answer_penalty: float
    final_reward: float

    def to_dict(self) -> dict[str, float]:
        return {
            "base_reward": self.base_reward,
            "duplicate_query_penalty": self.duplicate_query_penalty,
            "empty_result_penalty": self.empty_result_penalty,
            "max_search_no_answer_penalty": self.max_search_no_answer_penalty,
            "verbose_answer_penalty": self.verbose_answer_penalty,
            "bad_max_search_penalty": self.bad_max_search_penalty,
            "date_granularity_penalty": self.date_granularity_penalty,
            "multi_candidate_answer_penalty": self.multi_candidate_answer_penalty,
            "final_reward": self.final_reward,
        }


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


def apply_reward_shaping(
    base: RewardResult,
    diagnostics: TrajectoryDiagnostics,
    config: RewardShapingConfig,
    *,
    answer_token_count: int = 0,
    references: list[str] | None = None,
) -> RewardComponents:
    """Apply optional trajectory behavior penalties to a base answer reward."""
    duplicate_query_penalty = 0.0
    empty_result_penalty = 0.0
    max_search_no_answer_penalty = 0.0
    verbose_answer_penalty = 0.0
    bad_max_search_penalty = 0.0
    date_granularity_penalty = 0.0
    multi_candidate_answer_penalty = 0.0

    if not base.exact_match:
        if diagnostics.duplicate_query_count > 0:
            duplicate_query_penalty = config.duplicate_query_penalty
        if diagnostics.empty_observation_count > 0:
            empty_result_penalty = config.empty_result_penalty
        if diagnostics.max_search_no_answer:
            max_search_no_answer_penalty = config.max_search_no_answer_penalty
        if diagnostics.bad_max_search_loop:
            bad_max_search_penalty = config.bad_max_search_penalty
        if (
            base.answer is not None
            and config.verbose_answer_penalty > 0.0
            and config.verbose_answer_token_threshold > 0
            and answer_token_count > config.verbose_answer_token_threshold
        ):
            verbose_answer_penalty = config.verbose_answer_penalty
        if base.answer is not None and references:
            if _date_granularity_miss(base.answer, references):
                date_granularity_penalty = config.date_granularity_penalty
            if _multi_candidate_answer(base.answer, references):
                multi_candidate_answer_penalty = config.multi_candidate_answer_penalty

    final_reward = (
        base.reward
        - duplicate_query_penalty
        - empty_result_penalty
        - max_search_no_answer_penalty
        - verbose_answer_penalty
        - bad_max_search_penalty
        - date_granularity_penalty
        - multi_candidate_answer_penalty
    )
    return RewardComponents(
        base_reward=base.reward,
        duplicate_query_penalty=duplicate_query_penalty,
        empty_result_penalty=empty_result_penalty,
        max_search_no_answer_penalty=max_search_no_answer_penalty,
        verbose_answer_penalty=verbose_answer_penalty,
        bad_max_search_penalty=bad_max_search_penalty,
        date_granularity_penalty=date_granularity_penalty,
        multi_candidate_answer_penalty=multi_candidate_answer_penalty,
        final_reward=final_reward,
    )


YEAR_PATTERN = re.compile(r"\b(1[5-9]\d{2}|20\d{2})\b")
MULTI_CANDIDATE_PATTERN = re.compile(
    r"\b(or|或者|或)\b|/|;|\((?:[^)]*\b(or|for)\b[^)]*)\)",
    re.IGNORECASE,
)


def _date_granularity_miss(answer: str, references: list[str]) -> bool:
    normalized_answer = normalize_answer(answer)
    answer_years = set(YEAR_PATTERN.findall(answer))
    if not normalized_answer or not answer_years:
        return False
    if normalized_answer not in answer_years:
        return False
    return any(
        answer_years & set(YEAR_PATTERN.findall(reference))
        and normalize_answer(reference) != normalized_answer
        for reference in references
    )


def _multi_candidate_answer(answer: str, references: list[str]) -> bool:
    if not MULTI_CANDIDATE_PATTERN.search(answer):
        return False
    normalized_answer = normalize_answer(answer)
    return not any(
        normalized_answer == normalize_answer(reference)
        for reference in references
    )
