"""Registry for named tool backends."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from search_r1_minilab.tools.base import SearchBackend, SearchResult


@dataclass
class ToolRegistry:
    """Dispatch tool calls by name so rollout does not depend on concrete clients."""

    backends: dict[str, SearchBackend] = field(default_factory=dict)

    def register(self, backend: SearchBackend) -> None:
        """Register or replace a backend by its public name."""
        if not backend.name:
            raise ValueError("backend name must not be empty")
        self.backends[backend.name] = backend

    def call(self, tool_name: str, arguments: dict[str, Any] | None) -> SearchResult:
        """Call a registered tool with normalized arguments."""
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
        """Merge metrics from all registered backends."""
        merged: dict[str, float] = {}
        for backend in self.backends.values():
            merged.update(backend.metrics())
        return merged
