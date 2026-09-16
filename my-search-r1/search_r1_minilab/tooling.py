"""训练、评测和 smoke 脚本共用的搜索 backend 构造逻辑。

脚本层只传入 BackendConfig；这里负责选择真实/离线/mock backend，并按需套上
failure injection wrapper。这样 rollout 不需要知道具体搜索服务的初始化细节。
"""

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
    """单个搜索 backend 的运行配置。"""

    backend: str = "local_bm25"
    bm25_corpus: str | Path | None = None
    env_file: str | Path | None = None
    failure_seed: int = 0
    p_timeout: float = 0.0
    p_empty: float = 0.0
    p_noise: float = 0.0
    p_rate_limited: float = 0.0


def build_backend(config: BackendConfig) -> SearchBackend:
    """构造一个命名 backend，并在 failure injection 下保留原 dispatch 名。"""
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
        # wrapper 的 name 仍使用底层 backend.name，保证 registry.call(args.backend)
        # 在注入故障后仍能按同一个工具名分发。
        backend = FailureWrapperBackend(backend, failure_config, name=backend.name)
    return backend


def build_registry(config: BackendConfig) -> ToolRegistry:
    """构造只注册当前 backend 的 ToolRegistry。"""
    registry = ToolRegistry()
    registry.register(build_backend(config))
    return registry


def default_mock_backend() -> MockSearchBackend:
    """返回小型确定性 mock backend，用于单测和无需外部服务的 smoke。"""
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
