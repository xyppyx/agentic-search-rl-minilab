"""Tool smoke helpers that emit trajectory-compatible JSONL records."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from search_r1_minilab.trajectories.io import write_trajectory_jsonl
from search_r1_minilab.tools.registry import ToolRegistry


def build_tool_smoke_records(
    registry: ToolRegistry,
    tool_name: str,
    queries: Iterable[str],
) -> list[dict]:
    """Call one backend for queries and return trajectory-compatible records."""
    records: list[dict] = []
    for index, query in enumerate(queries):
        result = registry.call(tool_name, {"query": query})
        records.append(
            {
                "question": query,
                "answers": [],
                "data_source": "tool_smoke",
                "turns": [
                    {
                        "role": "assistant",
                        "text": "",
                        "tool_call": {"name": tool_name, "query": query},
                    },
                    {
                        "role": "tool",
                        "tool_name": tool_name,
                        "ok": result.ok,
                        "items": [item.to_dict() for item in result.items],
                        "error_type": result.error_type,
                        "error": result.error,
                        "metadata": result.metadata,
                    },
                ],
                "reward": None,
                "advantage": None,
                "valid_format": None,
                "exact_match": None,
                "search_calls": 1,
                "tool_failures": 0 if result.ok else 1,
                "metadata": {
                    "index": index,
                    "backend": result.backend,
                    "latency": result.latency,
                    "status": result.status,
                },
            }
        )
    return records


def write_jsonl(records: Iterable[dict], output_path: str | Path) -> None:
    """Write smoke records as UTF-8 JSONL."""
    write_trajectory_jsonl(records, output_path)
