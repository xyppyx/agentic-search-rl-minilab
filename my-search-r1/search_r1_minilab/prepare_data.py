"""Prepare Search-R1 train/eval JSONL data from the fixed ModelScope dataset."""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from datasets import load_dataset
from modelscope import dataset_snapshot_download


DATASET_ID = "zhuangzhuang2023/nq_hotpotqa_train"
DATASET_REVISION = "aa2da0496c1b1a50a66af7acabdf09c07a0cb79e"
RAW_COLUMNS = ["id", "question", "golden_answers", "data_source", "reward_model"]


def download_dataset(
    raw_dir: Path,
    *,
    dataset_id: str = DATASET_ID,
    revision: str = DATASET_REVISION,
) -> Path:
    """Download the pinned train/test parquet files from ModelScope."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = dataset_snapshot_download(
        dataset_id,
        revision=revision,
        local_dir=str(raw_dir),
        allow_patterns=["train.parquet", "test.parquet"],
    )
    return Path(path)


def clean_answers(values: Any) -> list[str]:
    """Clean an answer field while preserving order and removing duplicates."""
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, (list, tuple)):
        return []
    answers: list[str] = []
    for value in values:
        answer = str(value).strip()
        if answer and answer not in answers:
            answers.append(answer)
    return answers


def extract_answers(row: dict[str, Any]) -> list[str]:
    """Read golden answers, falling back to reward_model.ground_truth."""
    answers = clean_answers(row.get("golden_answers"))
    if answers:
        return answers
    reward_model = row.get("reward_model") or {}
    ground_truth = reward_model.get("ground_truth") if isinstance(reward_model, dict) else None
    if isinstance(ground_truth, dict):
        ground_truth = ground_truth.get("target")
    return clean_answers(ground_truth)


def normalize_row(row: dict[str, Any]) -> dict[str, Any] | None:
    """Normalize one raw dataset row to the MiniLab QA JSONL schema."""
    question = str(row.get("question") or "").strip()
    answers = extract_answers(row)
    if not question or not answers:
        return None
    return {
        "id": str(row.get("id") or ""),
        "question": question,
        "answers": answers,
        "data_source": str(row.get("data_source") or "unknown"),
    }


def prepare_split(
    parquet_path: Path,
    output_path: Path,
    *,
    collect_records: bool = False,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    """Clean one parquet split and write JSONL records."""
    dataset = load_dataset(
        "parquet",
        data_files=str(parquet_path),
        split="train",
        columns=RAW_COLUMNS,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    with output_path.open("w", encoding="utf-8") as stream:
        for row in dataset:
            record = normalize_row(row)
            if record is None:
                continue
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            if collect_records:
                records.append(record)
            counts[record["data_source"]] += 1
    return records, counts


def select_dev(
    records: list[dict[str, Any]],
    per_source: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Select a deterministic balanced dev set from cleaned test records."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[record["data_source"]].append(record)
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for source in sorted(groups):
        candidates = list(groups[source])
        rng.shuffle(candidates)
        selected.extend(candidates[:per_source])
    return selected


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    """Write records as UTF-8 JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def prepare_dataset(
    *,
    raw_dir: Path,
    output_dir: Path,
    dev_per_source: int = 10,
    seed: int = 42,
    dataset_id: str = DATASET_ID,
    revision: str = DATASET_REVISION,
) -> dict[str, Counter[str]]:
    """Download and prepare train, test, and balanced dev JSONL files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded_dir = download_dataset(
        raw_dir,
        dataset_id=dataset_id,
        revision=revision,
    )

    _, train_counts = prepare_split(downloaded_dir / "train.parquet", output_dir / "train.jsonl")
    test_records, test_counts = prepare_split(
        downloaded_dir / "test.parquet",
        output_dir / "test.jsonl",
        collect_records=True,
    )
    dev_records = select_dev(test_records, dev_per_source, seed)
    write_jsonl(dev_records, output_dir / "dev.jsonl")
    dev_counts = Counter(record["data_source"] for record in dev_records)
    return {"train": train_counts, "test": test_counts, "dev": dev_counts}


def format_counts(name: str, counts: Counter[str]) -> str:
    """Return a compact split count summary."""
    details = ", ".join(f"{source}={count}" for source, count in sorted(counts.items()))
    return f"{name}: total={sum(counts.values())}; {details}"
