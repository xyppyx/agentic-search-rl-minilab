"""Search-R1 MiniLab 训练和评测数据读取工具。

数据统一读成 SearchExample，隐藏不同 JSONL 字段的兼容处理，让 rollout 只关心
question、answers 和 data_source。
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SearchExample:
    """一条搜索问答样本。"""

    id: str
    question: str
    answers: list[str]
    data_source: str


def load_examples(path: str | Path, limit: int = 0) -> list[SearchExample]:
    """从 JSONL 加载 Search-R1 样本。"""
    examples: list[SearchExample] = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            answers = row.get("answers", [])
            if not isinstance(answers, list):
                answers = [answers]
            examples.append(
                SearchExample(
                    id=str(row.get("id") or f"example-{line_number}"),
                    question=str(row["question"]),
                    answers=[str(answer) for answer in answers],
                    data_source=str(row.get("data_source") or "minilab"),
                )
            )
            if limit > 0 and len(examples) >= limit:
                break
    if not examples:
        raise ValueError(f"no examples loaded from {path}")
    return examples


def shuffled_examples(path: str | Path, seed: int, limit: int = 0) -> list[SearchExample]:
    """加载样本并用固定 seed 做确定性打乱。"""
    examples = load_examples(path, limit=limit)
    random.Random(seed).shuffle(examples)
    return examples


def take_batch(
    examples: list[SearchExample],
    start: int,
    batch_size: int,
) -> list[SearchExample]:
    """按 start/batch_size 取循环 batch，用于长训练中复用小数据集。"""
    if not examples:
        return []
    return [examples[(start + offset) % len(examples)] for offset in range(batch_size)]
