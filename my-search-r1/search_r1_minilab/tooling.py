"""Shared backend construction for MiniLab scripts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from search_r1_minilab.tools import (
    FailureConfig,
    FailureWrapperBackend,
    LocalBM25Backend,
    MockSearchBackend,
    SearchBackend,
    SearchItem,
    ToolRegistry,
)
from search_r1_minilab.tools.zhihu import ZhihuSearchBackend


BACKEND_CHOICES = ("local_bm25", "mock_search", "zhihu_search")


@dataclass(frozen=True)
class BackendConfig:
    """Configuration for one dispatch backend."""

    backend: str = "local_bm25"
    bm25_corpus: str | Path | None = None
    env_file: str | Path | None = None
    failure_seed: int = 0
    p_timeout: float = 0.0
    p_empty: float = 0.0
    p_noise: float = 0.0
    p_rate_limited: float = 0.0


def build_backend(config: BackendConfig) -> SearchBackend:
    """Build one named backend, preserving its dispatch name under failure injection."""
    if config.backend == "local_bm25":
        if config.bm25_corpus is None:
            raise ValueError("local_bm25 requires bm25_corpus")
        backend: SearchBackend = LocalBM25Backend.from_jsonl(config.bm25_corpus)
    elif config.backend == "mock_search":
        backend = default_mock_backend()
    elif config.backend == "zhihu_search":
        if config.env_file is None:
            raise ValueError("zhihu_search requires env_file")
        backend = ZhihuSearchBackend.from_env(config.env_file)
    else:
        raise ValueError(f"unknown backend: {config.backend}")

    failure_config = FailureConfig(
        p_timeout=config.p_timeout,
        p_empty=config.p_empty,
        p_noise=config.p_noise,
        p_rate_limited=config.p_rate_limited,
        seed=config.failure_seed,
    )
    if any(
        value > 0.0
        for value in (
            failure_config.p_timeout,
            failure_config.p_empty,
            failure_config.p_noise,
            failure_config.p_rate_limited,
        )
    ):
        backend = FailureWrapperBackend(backend, failure_config, name=backend.name)
    return backend


def build_registry(config: BackendConfig) -> ToolRegistry:
    """Build a registry with exactly one configured dispatch backend."""
    registry = ToolRegistry()
    registry.register(build_backend(config))
    return registry


def default_mock_backend() -> MockSearchBackend:
    """Return a small deterministic backend for tests and smoke runs."""
    return MockSearchBackend.from_pairs(
        {
            "little prince": [
                SearchItem(
                    id="mock-little-prince",
                    title="The Little Prince",
                    content="The Little Prince is by Antoine de Saint-Exupery.",
                    url="https://example.test/little-prince",
                    source="mock",
                )
            ],
            "search-r1": [
                SearchItem(
                    id="mock-search-r1",
                    title="Search-R1",
                    content="Search-R1 trains language models to use search engines.",
                    url="https://example.test/search-r1",
                    source="mock",
                )
            ],
        }
    )
