"""按工具名分发搜索调用的 registry。

rollout 只调用 ToolRegistry.call(tool_name, arguments)，因此训练逻辑不需要区分
mock、本地 BM25、知乎搜索或 failure wrapper。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from search_r1_minilab.tools.base import SearchBackend, SearchResult


@dataclass
class ToolRegistry:
    """按名称分发工具调用，解耦 rollout 与具体搜索 client。"""

    backends: dict[str, SearchBackend] = field(default_factory=dict)

    def register(self, backend: SearchBackend) -> None:
        """按 backend.name 注册或替换一个搜索 backend。"""
        if not backend.name:
            raise ValueError("backend name must not be empty")
        self.backends[backend.name] = backend

    def call(self, tool_name: str, arguments: dict[str, Any] | None) -> SearchResult:
        """调用已注册工具，并把参数错误也转成 SearchResult。"""
        started = time.perf_counter()
        backend = self.backends.get(tool_name)
        if backend is None:
            return SearchResult(
                ok=False,
                items=[],
                latency=time.perf_counter() - started,
                backend=tool_name,
                error_type="unknown_tool",
                error=f"unknown tool: {tool_name}",
            )
        if not isinstance(arguments, dict):
            return SearchResult(
                ok=False,
                items=[],
                latency=time.perf_counter() - started,
                backend=tool_name,
                error_type="invalid_arguments",
                error="tool arguments must be an object",
            )
        query = arguments.get("query")
        if not isinstance(query, str):
            return SearchResult(
                ok=False,
                items=[],
                latency=time.perf_counter() - started,
                backend=tool_name,
                error_type="invalid_arguments",
                error="tool argument 'query' must be a string",
            )
        return backend.search(query)

    def metrics(self) -> dict[str, float]:
        """合并所有 backend 的累计指标。"""
        merged: dict[str, float] = {}
        for backend in self.backends.values():
            merged.update(backend.metrics())
        return merged
