"""Trajectory JSONL 读写。

所有训练、评测和 smoke 产物都尽量落成稳定 schema，方便后续 report、离线诊断、
turn-credit 分析和 case review 复用。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


# 统一顶层字段，避免不同脚本输出缺字段导致 report/analysis 分支复杂化。
DEFAULT_TRAJECTORY_FIELDS = {
    "question": "",
    "answers": [],
    "data_source": "",
    "turns": [],
    "reward": None,
    "advantage": None,
    "valid_format": None,
    "exact_match": None,
    "search_calls": 0,
    "tool_failures": 0,
    "metadata": {},
}


def normalize_trajectory_record(record: dict[str, Any]) -> dict[str, Any]:
    """补齐并校验 trajectory 顶层字段。"""
    normalized = {**DEFAULT_TRAJECTORY_FIELDS, **record}
    if not isinstance(normalized["answers"], list):
        normalized["answers"] = [normalized["answers"]]
    if not isinstance(normalized["turns"], list):
        raise ValueError("trajectory field 'turns' must be a list")
    if not isinstance(normalized["metadata"], dict):
        raise ValueError("trajectory field 'metadata' must be an object")
    normalized["search_calls"] = int(normalized["search_calls"] or 0)
    normalized["tool_failures"] = int(normalized["tool_failures"] or 0)
    return normalized


def write_trajectory_jsonl(
    records: Iterable[dict[str, Any]],
    output_path: str | Path,
) -> int:
    """把标准化 trajectory records 写成 UTF-8 JSONL。"""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            normalized = normalize_trajectory_record(record)
            stream.write(json.dumps(normalized, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def load_trajectory_jsonl(input_path: str | Path) -> list[dict[str, Any]]:
    """从 JSONL 读取并标准化 trajectory records。"""
    path = Path(input_path)
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON on line {line_number}: {error}") from error
            if not isinstance(raw, dict):
                raise ValueError(f"trajectory line {line_number} must be an object")
            records.append(normalize_trajectory_record(raw))
    return records
