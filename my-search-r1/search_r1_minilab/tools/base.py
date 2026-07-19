"""Shared search backend interfaces and result types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class SearchItem:
    """One normalized search result item."""

    title: str
    content: str
    url: str = ""
    source: str = ""
    id: str | None = None
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the item for trajectory JSONL."""
        return {
            "id": self.id,
            "title": self.title,
            "content": self.content,
            "url": self.url,
            "source": self.source,
            "score": self.score,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class SearchResult:
    """Normalized search result or failure."""

    ok: bool
    items: list[SearchItem]
    latency: float
    backend: str
    status: int | None = None
    error_type: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the result for trajectory JSONL."""
        return {
            "ok": self.ok,
            "items": [item.to_dict() for item in self.items],
            "latency": self.latency,
            "backend": self.backend,
            "status": self.status,
            "error_type": self.error_type,
            "error": self.error,
            "metadata": self.metadata,
        }


class SearchBackend(Protocol):
    """Minimal interface expected by rollout code."""

    name: str

    def search(self, query: str) -> SearchResult:
        """Search for evidence with a single query."""

    def metrics(self) -> dict[str, float]:
        """Return cumulative backend metrics."""


@dataclass
class SearchStats:
    """Backend-level counters with common metric names."""

    requests: int = 0
    successes: int = 0
    empty_results: int = 0
    timeouts: int = 0
    rate_limits: int = 0
    noisy_results: int = 0
    errors: int = 0
    latency_total: float = 0.0

    def observe(self, result: SearchResult) -> None:
        """Update counters from a normalized result."""
        self.requests += 1
        self.latency_total += result.latency
        if result.ok:
            self.successes += 1
            if not result.items:
                self.empty_results += 1
            if result.error_type == "noisy_result":
                self.noisy_results += 1
            return
        if result.error_type == "timeout":
            self.timeouts += 1
        elif result.error_type == "rate_limited":
            self.rate_limits += 1
        else:
            self.errors += 1

    def metrics(self, prefix: str) -> dict[str, float]:
        """Return rates and average latency with a caller-provided prefix."""
        denominator = max(self.requests, 1)
        return {
            f"{prefix}/requests": float(self.requests),
            f"{prefix}/success_rate": self.successes / denominator,
            f"{prefix}/empty_rate": self.empty_results / denominator,
            f"{prefix}/timeout_rate": self.timeouts / denominator,
            f"{prefix}/rate_limit_rate": self.rate_limits / denominator,
            f"{prefix}/noise_rate": self.noisy_results / denominator,
            f"{prefix}/error_rate": self.errors / denominator,
            f"{prefix}/latency": self.latency_total / denominator,
        }


def format_item(item: SearchItem, index: int) -> str:
    """Format a result item as tool-observation text."""
    return (
        f"[{index}] Title: {item.title}\n"
        f"    Content: {item.content}\n"
        f"    Source: {item.source}\n"
        f"    URL: {item.url}"
    )


def empty_success(query: str, backend: str, latency: float) -> SearchResult:
    """Return a successful empty result with explicit metadata."""
    return SearchResult(
        ok=True,
        items=[],
        latency=latency,
        backend=backend,
        metadata={"query": query, "empty": True},
    )
