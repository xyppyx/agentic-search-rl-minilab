"""Run a PyTRIO Search-R1 rollout smoke/eval and write JSONL plus Markdown."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pytrio as trio
from dotenv import load_dotenv

from search_r1_minilab.rollout_smoke import (
    RolloutSmokeConfig,
    build_metrics,
    load_examples,
    rollout_examples,
)
from search_r1_minilab.tooling import BackendConfig, build_registry
from search_r1_minilab.trajectories import build_markdown_report, write_trajectory_jsonl


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "tests" / "fixtures" / "smoke_eval.jsonl"
DEFAULT_BM25_CORPUS = ROOT / "tests" / "fixtures" / "bm25_corpus.jsonl"
DEFAULT_JSONL_OUTPUT = ROOT / "outputs" / "rollout_smoke" / "trajectories.jsonl"
DEFAULT_REPORT_OUTPUT = ROOT / "outputs" / "rollout_smoke" / "report.md"
DEFAULT_ENV_FILE = ROOT / ".env"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="Smoke/eval JSONL data")
    parser.add_argument("--limit", type=int, default=2, help="Maximum examples to evaluate")
    parser.add_argument("--batch-size", type=int, default=1, help="Examples per rollout batch")
    parser.add_argument(
        "--backend",
        choices=["local_bm25", "mock_search", "zhihu_search"],
        default="local_bm25",
        help="Actual backend dispatched behind the model-visible search tool",
    )
    parser.add_argument(
        "--bm25-corpus",
        default=str(DEFAULT_BM25_CORPUS),
        help="JSONL corpus for local_bm25",
    )
    parser.add_argument(
        "--base-model",
        default="Qwen/Qwen3.5-4B",
        help="Base model passed to PyTRIO sampling client",
    )
    parser.add_argument("--model-path", help="Optional PyTRIO sampler weights path")
    parser.add_argument("--env-file", default=str(DEFAULT_ENV_FILE), help="Local .env path")
    parser.add_argument(
        "--jsonl-output",
        default=str(DEFAULT_JSONL_OUTPUT),
        help="Trajectory JSONL output path",
    )
    parser.add_argument(
        "--report-output",
        default=str(DEFAULT_REPORT_OUTPUT),
        help="Markdown report output path",
    )
    parser.add_argument("--max-search-calls", type=int, default=2)
    parser.add_argument("--max-assistant-turns", type=int, default=4)
    parser.add_argument("--max-trajectory-tokens", type=int, default=4096)
    parser.add_argument("--max-assistant-tokens", type=int, default=512)
    parser.add_argument("--max-tool-response-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv(args.env_file)

    registry = build_registry(
        BackendConfig(
            backend=args.backend,
            bm25_corpus=args.bm25_corpus,
            env_file=args.env_file,
        )
    )

    service_client = trio.ServiceClient(api_key=os.getenv("PYTRIO_API_KEY") or None)
    sampling_client = service_client.create_sampling_client(
        base_model=args.base_model,
        model_path=args.model_path,
    )
    tokenizer = sampling_client.get_tokenizer()
    examples = load_examples(args.data, limit=args.limit)
    config = RolloutSmokeConfig(
        max_search_calls=args.max_search_calls,
        max_assistant_turns=args.max_assistant_turns,
        max_trajectory_tokens=args.max_trajectory_tokens,
        max_assistant_tokens=args.max_assistant_tokens,
        max_tool_response_tokens=args.max_tool_response_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
    )

    records = []
    for start in range(0, len(examples), max(args.batch_size, 1)):
        batch = examples[start : start + max(args.batch_size, 1)]
        records.extend(
            rollout_examples(
                sampling_client,
                tokenizer,
                registry,
                args.backend,
                batch,
                config,
                start_index=start,
            )
        )

    write_trajectory_jsonl(records, args.jsonl_output)
    report = build_markdown_report(
        records,
        title=f"PyTRIO Rollout Smoke Report: {args.backend}",
    )
    report_path = Path(args.report_output)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")

    metrics = build_metrics(records, registry)
    print(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"wrote {len(records)} trajectories to {args.jsonl_output}")
    print(f"wrote report to {args.report_output}")

if __name__ == "__main__":
    main()
