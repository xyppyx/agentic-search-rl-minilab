"""Failure injection wrapper for search backends."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field

from search_r1_minilab.tools.base import SearchBackend, SearchItem, SearchResult, SearchStats


@dataclass(frozen=True)
class FailureConfig:
    """Probabilities for distinguishable tool failures."""

    p_timeout: float = 0.0
    p_empty: float = 0.0
    p_noise: float = 0.0
    p_rate_limited: float = 0.0
    seed: int = 0

    def __post_init__(self) -> None:
        """Validate probability values."""
        probabilities = {
            "p_timeout": self.p_timeout,
            "p_empty": self.p_empty,
            "p_noise": self.p_noise,
            "p_rate_limited": self.p_rate_limited,
        }
        for name, value in probabilities.items():
            if value < 0.0 or value > 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if sum(probabilities.values()) > 1.0:
            raise ValueError("failure probabilities must sum to <= 1")


@dataclass
class FailureWrapperBackend:
    """Wrap any backend with seeded failure injection."""

    backend: SearchBackend
    config: FailureConfig
    name: str | None = None
    stats: SearchStats = field(default_factory=SearchStats)
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize wrapper name and RNG."""
        if self.name is None:
            self.name = f"{self.backend.name}_failure"
        self._rng = random.Random(self.config.seed)

    def search(self, query: str) -> SearchResult:
        """Inject a configured failure or delegate to the wrapped backend."""
        started = time.perf_counter()
        failure_type = self._draw_failure_type()
        if failure_type == "timeout":
            result = SearchResult(
                ok=False,
                items=[],
                latency=time.perf_counter() - started,
                backend=self.name or "failure_wrapper",
                error_type="timeout",
                error="injected timeout",
                metadata={
                    "query": query,
                    "wrapped_backend": self.backend.name,
                    "injected_failure": True,
                },
            )
        elif failure_type == "rate_limited":
            result = SearchResult(
                ok=False,
                items=[],
                latency=time.perf_counter() - started,
                backend=self.name or "failure_wrapper",
                status=429,
                error_type="rate_limited",
                error="injected rate limit",
                metadata={
                    "query": query,
                    "wrapped_backend": self.backend.name,
                    "injected_failure": True,
                },
            )
        elif failure_type == "empty_result":
            result = SearchResult(
                ok=True,
                items=[],
                latency=time.perf_counter() - started,
                backend=self.name or "failure_wrapper",
                error_type="empty_result",
                metadata={
                    "query": query,
                    "wrapped_backend": self.backend.name,
                    "injected_failure": True,
                },
            )
        elif failure_type == "noisy_result":
            result = SearchResult(
                ok=True,
                items=[
                    SearchItem(
                        id="injected-noise",
                        title="Injected noisy result",
                        content=(
                            "This result was injected by the failure wrapper and should "
                            "not be treated as reliable evidence."
                        ),
                        source="failure_wrapper",
                        score=0.0,
                        metadata={"injected_failure": True},
                    )
                ],
                latency=time.perf_counter() - started,
                backend=self.name or "failure_wrapper",
                error_type="noisy_result",
                metadata={
                    "query": query,
                    "wrapped_backend": self.backend.name,
                    "injected_failure": True,
                },
            )
        else:
            delegated = self.backend.search(query)
            result = SearchResult(
                ok=delegated.ok,
                items=delegated.items,
                latency=delegated.latency,
                backend=self.name or "failure_wrapper",
                status=delegated.status,
                error_type=delegated.error_type,
                error=delegated.error,
                metadata={
                    **delegated.metadata,
                    "wrapped_backend": self.backend.name,
                    "injected_failure": False,
                },
            )
        self.stats.observe(result)
        return result

    def _draw_failure_type(self) -> str | None:
        """Draw one failure type from the configured categorical distribution."""
        sample = self._rng.random()
        threshold = self.config.p_timeout
        if sample < threshold:
            return "timeout"
        threshold += self.config.p_rate_limited
        if sample < threshold:
            return "rate_limited"
        threshold += self.config.p_empty
        if sample < threshold:
            return "empty_result"
        threshold += self.config.p_noise
        if sample < threshold:
            return "noisy_result"
        return None

    def metrics(self) -> dict[str, float]:
        """Return wrapper metrics and wrapped backend metrics."""
        metrics = self.stats.metrics(f"tool/{self.name}")
        for key, value in self.backend.metrics().items():
            metrics[f"wrapped/{key}"] = value
        return metrics
