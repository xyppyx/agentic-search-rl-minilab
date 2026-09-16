"""搜索工具层的通用接口与统一结果类型。

所有 backend 都要输出 SearchResult，使 rollout/report 能用同一套字段记录成功、
空结果、超时、限流和其它错误。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class SearchItem:
    """一条标准化搜索结果。"""

    title: str
    content: str
    url: str = ""
    source: str = ""
    id: str | None = None
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """序列化为 trajectory JSONL 字段。"""
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
    """一次搜索调用的标准化成功结果或失败结果。"""

    ok: bool
    items: list[SearchItem]
    latency: float
    backend: str
    status: int | None = None
    error_type: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """序列化为 trajectory JSONL 字段。"""
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
    """rollout 层依赖的最小搜索 backend 协议。"""

    name: str

    def search(self, query: str) -> SearchResult:
        """用单个 query 检索证据。"""

    def metrics(self) -> dict[str, float]:
        """返回累计 backend 指标。"""


@dataclass
class SearchStats:
    """backend 级别计数器，产出统一 metric 名。"""

    requests: int = 0
    successes: int = 0
    empty_results: int = 0
    timeouts: int = 0
    rate_limits: int = 0
    noisy_results: int = 0
    errors: int = 0
    latency_total: float = 0.0

    def observe(self, result: SearchResult) -> None:
        """根据一次标准化搜索结果更新计数器。"""
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
        """按调用方 prefix 返回成功率、错误率和平均延迟。"""
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
    """把搜索结果格式化为模型可读的 tool observation 文本。"""
    return (
        f"[{index}] Title: {item.title}\n"
        f"    Content: {item.content}\n"
        f"    Source: {item.source}\n"
        f"    URL: {item.url}"
    )


def empty_success(query: str, backend: str, latency: float) -> SearchResult:
    """构造一次成功但无结果的搜索返回，并显式记录 query。"""
    return SearchResult(
        ok=True,
        items=[],
        latency=latency,
        backend=backend,
        metadata={"query": query, "empty": True},
    )
