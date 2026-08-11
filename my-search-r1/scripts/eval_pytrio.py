"""Run Search-R1 MiniLab PyTRIO eval with a pluggable search backend."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pytrio as trio
from dotenv import load_dotenv
from tqdm import tqdm

from search_r1_minilab.data import load_examples
from search_r1_minilab.rewards import RewardShapingConfig
from search_r1_minilab.rollout import RolloutConfig, Trajectory, rollout_batch, trajectory_to_record
from search_r1_minilab.tooling import BACKEND_CHOICES, BackendConfig, build_registry
from search_r1_minilab.training import evaluation_metrics
from search_r1_minilab.trajectories import build_markdown_report, write_trajectory_jsonl


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "tests" / "fixtures" / "smoke_eval.jsonl"
DEFAULT_BM25_CORPUS = ROOT / "tests" / "fixtures" / "bm25_corpus.jsonl"
DEFAULT_ENV_FILE = ROOT / ".env"
DEFAULT_JSONL_OUTPUT = ROOT / "eval_results" / "trajectories.jsonl"
DEFAULT_REPORT_OUTPUT = ROOT / "eval_results" / "report.md"


def parse_args() -> argparse.Namespace:
    """Parse eval, rollout, and backend arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--base-model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--model-path")
    parser.add_argument("--jsonl-output", type=Path, default=DEFAULT_JSONL_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    parser.add_argument("--backend", choices=BACKEND_CHOICES, default="local_bm25")
    parser.add_argument("--bm25-corpus", type=Path, default=DEFAULT_BM25_CORPUS)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--failure-seed", type=int, default=0)
    parser.add_argument("--p-timeout", type=float, default=0.0)
    parser.add_argument("--p-empty", type=float, default=0.0)
    parser.add_argument("--p-noise", type=float, default=0.0)
    parser.add_argument("--p-rate-limited", type=float, default=0.0)
    parser.add_argument("--max-search-calls", type=int, default=4)
    parser.add_argument("--max-assistant-turns", type=int, default=6)
    parser.add_argument("--max-trajectory-tokens", type=int, default=8192)
    parser.add_argument("--max-assistant-tokens", type=int, default=1024)
    parser.add_argument("--max-tool-response-tokens", type=int, default=1024)
    parser.add_argument("--duplicate-query-penalty", type=float, default=0.0)
    parser.add_argument("--empty-result-penalty", type=float, default=0.0)
    parser.add_argument("--max-search-no-answer-penalty", type=float, default=0.0)
    parser.add_argument("--bad-max-search-penalty", type=float, default=0.0)
    parser.add_argument("--date-granularity-penalty", type=float, default=0.0)
    parser.add_argument("--multi-candidate-answer-penalty", type=float, default=0.0)
    parser.add_argument("--helpful-followup-bonus", type=float, default=0.0)
    parser.add_argument("--no-search-penalty", type=float, default=0.0)
    parser.add_argument("--verbose-answer-penalty", type=float, default=0.0)
    parser.add_argument("--verbose-answer-token-threshold", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    """Run eval and write trajectory JSONL plus Markdown report."""
    args = parse_args()
    load_dotenv(args.env_file)

    registry = build_registry(_backend_config(args))
    examples = load_examples(args.data, limit=0)
    if args.offset < 0:
        raise ValueError("--offset must be non-negative")
    examples = examples[args.offset :]
    if args.limit > 0:
        examples = examples[: args.limit]
    if not examples:
        raise ValueError(f"no examples selected from {args.data}")
    service_client = trio.ServiceClient(api_key=os.getenv("PYTRIO_API_KEY") or None)
    sampling_client = service_client.create_sampling_client(
        base_model=args.base_model,
        model_path=args.model_path,
    )
    tokenizer = sampling_client.get_tokenizer()
    config = RolloutConfig(
        group_size=1,
        max_search_calls=args.max_search_calls,
        max_assistant_turns=args.max_assistant_turns,
        max_trajectory_tokens=args.max_trajectory_tokens,
        max_assistant_tokens=args.max_assistant_tokens,
        max_tool_response_tokens=args.max_tool_response_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        reward_shaping=_reward_shaping_config(args),
    )

    trajectories: list[Trajectory] = []
    batches = _chunks(examples, max(args.batch_size, 1))
    with tqdm(total=len(examples), desc="Eval", unit="question") as progress:
        for batch in batches:
            batch_trajectories = rollout_batch(
                sampling_client,
                tokenizer,
                registry,
                args.backend,
                batch,
                config,
                progress_callback=progress.update,
            )
            trajectories.extend(batch_trajectories)

    records = [trajectory_to_record(item, run_type="eval") for item in trajectories]
    write_trajectory_jsonl(records, args.jsonl_output)
    report = build_markdown_report(records, title=f"PyTRIO Eval Report: {args.backend}")
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(report, encoding="utf-8")

    metrics = evaluation_metrics(trajectories)
    metrics.update(registry.metrics())
    print(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"wrote {len(records)} trajectories to {args.jsonl_output}")
    print(f"wrote report to {args.report_output}")


def _backend_config(args: argparse.Namespace) -> BackendConfig:
    return BackendConfig(
        backend=args.backend,
        bm25_corpus=args.bm25_corpus,
        env_file=args.env_file,
        failure_seed=args.failure_seed,
        p_timeout=args.p_timeout,
        p_empty=args.p_empty,
        p_noise=args.p_noise,
        p_rate_limited=args.p_rate_limited,
    )


def _reward_shaping_config(args: argparse.Namespace) -> RewardShapingConfig:
    return RewardShapingConfig(
        duplicate_query_penalty=args.duplicate_query_penalty,
        empty_result_penalty=args.empty_result_penalty,
        max_search_no_answer_penalty=args.max_search_no_answer_penalty,
        bad_max_search_penalty=args.bad_max_search_penalty,
        date_granularity_penalty=args.date_granularity_penalty,
        multi_candidate_answer_penalty=args.multi_candidate_answer_penalty,
        helpful_followup_bonus=args.helpful_followup_bonus,
        no_search_penalty=args.no_search_penalty,
        verbose_answer_penalty=args.verbose_answer_penalty,
        verbose_answer_token_threshold=args.verbose_answer_token_threshold,
    )


def _chunks(examples: list, size: int) -> list[list]:
    return [examples[index : index + size] for index in range(0, len(examples), size)]


if __name__ == "__main__":
    main()
