"""Deterministic mock search backend for tests and smoke runs."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterable

from search_r1_minilab.tools.base import (
    SearchItem,
    SearchResult,
    SearchStats,
    empty_success,
)


def normalize_query(query: str) -> str:
    """Normalize query keys for deterministic fixture lookup."""
    return " ".join(query.lower().strip().split())


@dataclass
class MockSearchBackend:
    """Return fixed results for known queries and empty results otherwise."""

    fixtures: dict[str, list[SearchItem]]
    name: str = "mock_search"
    stats: SearchStats = field(default_factory=SearchStats)

    @classmethod
    def from_pairs(
        cls,
        pairs: dict[str, Iterable[SearchItem | dict[str, str]]],
        name: str = "mock_search",
    ) -> "MockSearchBackend":
        """Build fixtures from SearchItem objects or simple dictionaries."""
        fixtures: dict[str, list[SearchItem]] = {}
        for query, items in pairs.items():
            fixtures[normalize_query(query)] = [
                item
                if isinstance(item, SearchItem)
                else SearchItem(
                    title=item.get("title", ""),
                    content=item.get("content", ""),
                    url=item.get("url", ""),
                    source=item.get("source", "mock"),
                    id=item.get("id"),
                )
                for item in items
            ]
        return cls(fixtures=fixtures, name=name)

    def search(self, query: str) -> SearchResult:
        """Search the fixture map."""
        started = time.perf_counter()
        key = normalize_query(query)
        if key not in self.fixtures:
            result = empty_success(query, self.name, time.perf_counter() - started)
            self.stats.observe(result)
            return result
        result = SearchResult(
            ok=True,
            items=self.fixtures[key],
            latency=time.perf_counter() - started,
            backend=self.name,
            metadata={"query": query, "fixture_key": key},
        )
        self.stats.observe(result)
        return result

    def metrics(self) -> dict[str, float]:
        """Return cumulative mock backend metrics."""
        return self.stats.metrics(f"tool/{self.name}")
