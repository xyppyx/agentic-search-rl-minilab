"""Train Search-R1 MiniLab with PyTRIO GRPO and pluggable search backends."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any

import pytrio as trio
from dotenv import load_dotenv
from tqdm import tqdm

from search_r1_minilab.data import shuffled_examples, take_batch
from search_r1_minilab.rewards import RewardShapingConfig
from search_r1_minilab.rollout import RolloutConfig, rollout_batch, trajectory_to_record
from search_r1_minilab.tooling import BACKEND_CHOICES, BackendConfig, build_registry
from search_r1_minilab.training import (
    add_reference_logprobs,
    build_custom_forward_datums,
    build_training_datums,
    loss_input_float_lists,
    make_grpo_kl_loss_fn,
    merge_trainer_metrics,
    pack_micro_batches,
    pick_mean_loss_metric,
    rollout_metrics,
    save_checkpoint,
    TurnCreditConfig,
    weight_micro_batch_items_for_global_mean,
)
from search_r1_minilab.trajectories import build_markdown_report, write_trajectory_jsonl


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "tests" / "fixtures" / "smoke_eval.jsonl"
DEFAULT_BM25_CORPUS = ROOT / "tests" / "fixtures" / "bm25_corpus.jsonl"
DEFAULT_ENV_FILE = ROOT / ".env"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "train_pytrio"
DEFAULT_SWANLAB_PROJECT = "llm-agent-rl-lab-search-r1"
DEFAULT_ADVANTAGE_NORMALIZATION = "standardize"
DEFAULT_ADVANTAGE_CLIP = 2.0
DEFAULT_KL_COEF = 0.01
DEFAULT_POLICY_RATIO_CLIP = 0.2
DEFAULT_LEARNING_RATE = 1e-5
_SWANLAB_MODULE: Any | None = None


def parse_args() -> argparse.Namespace:
    """Parse train, rollout, backend, and logging arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--base-model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--resume-state")
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--questions-per-batch", type=int, default=1)
    parser.add_argument("--group-size", type=int, default=2)
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
    parser.add_argument(
        "--turn-credit-policy",
        choices=["none", "helpful_bridge", "evidence_bridge", "final_hop_bridge"],
        default="none",
    )
    parser.add_argument("--helpful-search-turn-bonus", type=float, default=0.0)
    parser.add_argument("--evidence-search-turn-bonus", type=float, default=0.0)
    parser.add_argument("--final-hop-search-turn-bonus", type=float, default=0.0)
    parser.add_argument("--early-answer-turn-penalty", type=float, default=0.0)
    parser.add_argument("--missing-final-hop-turn-penalty", type=float, default=0.0)
    parser.add_argument("--verbose-answer-penalty", type=float, default=0.0)
    parser.add_argument("--verbose-answer-token-threshold", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--advantage-normalization",
        choices=["center", "standardize"],
        default=DEFAULT_ADVANTAGE_NORMALIZATION,
        help="Group advantage normalization; default uses KL/std-stabilized GRPO.",
    )
    parser.add_argument("--advantage-epsilon", type=float, default=1e-6)
    parser.add_argument("--advantage-clip", type=float, default=DEFAULT_ADVANTAGE_CLIP)
    parser.add_argument("--kl-coef", type=float, default=DEFAULT_KL_COEF)
    parser.add_argument("--policy-ratio-clip", type=float, default=DEFAULT_POLICY_RATIO_CLIP)
    parser.add_argument("--reference-model-path")
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", default="search-r1-minilab")
    parser.add_argument("--swanlab-project")
    parser.add_argument(
        "--swanlab-mode",
        choices=["online", "local", "offline", "disabled"],
        default="disabled",
    )
    parser.add_argument("--trajectory-output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--backend", choices=BACKEND_CHOICES, default="local_bm25")
    parser.add_argument("--bm25-corpus", type=Path, default=DEFAULT_BM25_CORPUS)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--failure-seed", type=int, default=0)
    parser.add_argument("--p-timeout", type=float, default=0.0)
    parser.add_argument("--p-empty", type=float, default=0.0)
    parser.add_argument("--p-noise", type=float, default=0.0)
    parser.add_argument("--p-rate-limited", type=float, default=0.0)
    return parser.parse_args()


def main(args: argparse.Namespace | None = None) -> None:
    """Run the full on-policy training loop."""
    args = args or parse_args()
    load_dotenv(args.env_file)
    args.swanlab_project = _resolve_swanlab_project(args.swanlab_project)

    examples = shuffled_examples(args.data, args.seed, limit=args.max_train_samples)
    registry = build_registry(_backend_config(args))
    service_client = trio.ServiceClient(api_key=os.getenv("PYTRIO_API_KEY") or None)
    if args.resume_state:
        training_client = service_client.create_training_client_from_state(args.resume_state)
    else:
        training_client = service_client.create_lora_training_client(
            base_model=args.base_model,
            rank=args.lora_rank,
            seed=args.seed,
        )
    tokenizer = training_client.get_tokenizer()

    rollout_config = RolloutConfig(
        group_size=args.group_size,
        max_search_calls=args.max_search_calls,
        max_assistant_turns=args.max_assistant_turns,
        max_trajectory_tokens=args.max_trajectory_tokens,
        max_assistant_tokens=args.max_assistant_tokens,
        max_tool_response_tokens=args.max_tool_response_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        advantage_normalization=args.advantage_normalization,
        advantage_epsilon=args.advantage_epsilon,
        advantage_clip=args.advantage_clip,
        reward_shaping=_reward_shaping_config(args),
    )
    turn_credit_config = TurnCreditConfig(
        policy=args.turn_credit_policy,
        helpful_search_turn_bonus=args.helpful_search_turn_bonus,
        evidence_search_turn_bonus=args.evidence_search_turn_bonus,
        final_hop_search_turn_bonus=args.final_hop_search_turn_bonus,
        early_answer_turn_penalty=args.early_answer_turn_penalty,
        missing_final_hop_turn_penalty=args.missing_final_hop_turn_penalty,
    )
    adam_params = trio.AdamParams(
        learning_rate=args.learning_rate,
        beta1=args.beta1,
        beta2=args.beta2,
    )
    reference_client = None
    if args.kl_coef > 0.0:
        reference_client = service_client.create_sampling_client(
            base_model=args.base_model,
            model_path=args.reference_model_path,
        )

    run = _init_swanlab(args)

    try:
        with tqdm(total=args.max_steps, desc="Training", unit="step", position=0) as train_bar:
            for step in range(args.max_steps):
                step_started = perf_counter()
                batch = take_batch(
                    examples,
                    step * args.questions_per_batch,
                    args.questions_per_batch,
                )
                train_bar.set_postfix(phase="prepare sampler", refresh=True)
                sampling_client = training_client.save_weights_and_get_sampling_client()

                train_bar.set_postfix(phase="rollout", refresh=True)
                with tqdm(
                    total=len(batch) * args.group_size,
                    desc=f"Step {step + 1}/{args.max_steps} rollout",
                    unit="trajectory",
                    position=1,
                    leave=False,
                ) as rollout_bar:
                    trajectories = rollout_batch(
                        sampling_client,
                        tokenizer,
                        registry,
                        args.backend,
                        batch,
                        rollout_config,
                        progress_callback=rollout_bar.update,
                    )

                train_bar.set_postfix(phase="build datums", refresh=True)
                datums = build_training_datums(trajectories, turn_credit_config)
                if reference_client is not None and datums:
                    train_bar.set_postfix(phase="reference logprobs", refresh=True)
                    datums = add_reference_logprobs(datums, reference_client)
                micro_batches = pack_micro_batches(datums)

                train_bar.set_postfix(phase="backward", refresh=True)
                trainer_results = []
                for micro_batch in micro_batches:
                    weighted_items = weight_micro_batch_items_for_global_mean(
                        micro_batch,
                        total_samples=len(trajectories),
                    )
                    if args.kl_coef > 0.0:
                        result = training_client.forward_backward_custom(
                            build_custom_forward_datums(weighted_items),
                            make_grpo_kl_loss_fn(
                                sampling_logprobs_list=loss_input_float_lists(
                                    weighted_items,
                                    "logprobs",
                                ),
                                advantages_list=loss_input_float_lists(
                                    weighted_items,
                                    "advantages",
                                ),
                                reference_logprobs_list=[
                                    _require_reference_logprobs(item)
                                    for item in weighted_items
                                ],
                                kl_coef=args.kl_coef,
                                policy_ratio_clip=args.policy_ratio_clip,
                            ),
                        ).result()
                    else:
                        result = training_client.forward_backward(
                            [item.datum for item in weighted_items],
                            loss_fn="importance_sampling",
                        ).result()
                    trainer_results.append(result)

                if micro_batches:
                    train_bar.set_postfix(phase="optimizer", refresh=True)
                    training_client.optim_step(adam_params).result()

                metrics = rollout_metrics(
                    trajectories,
                    datums,
                    micro_batches,
                    len(batch),
                    turn_credit_policy=args.turn_credit_policy,
                )
                metrics.update(registry.metrics())
                metrics.update(merge_trainer_metrics(trainer_results))
                metrics["time/step_seconds"] = perf_counter() - step_started
                metrics["train/update_skipped"] = float(not micro_batches)
                _log_swanlab(args, metrics, step)
                _write_step_artifacts(args, step, trajectories, metrics)

                mean_loss = pick_mean_loss_metric(metrics)
                loss_text = (
                    "skipped"
                    if not micro_batches
                    else "missing"
                    if mean_loss is None
                    else f"{mean_loss:.4f}"
                )
                train_bar.update(1)
                train_bar.set_postfix(
                    step_s=f"{metrics['time/step_seconds']:.1f}",
                    loss_mean=loss_text,
                    reward=f"{metrics['reward/mean']:.3f}",
                    refresh=True,
                )
                tqdm.write(
                    f"step={step + 1}/{args.max_steps} "
                    f"step_time={metrics['time/step_seconds']:.1f}s "
                    f"loss_mean={loss_text} "
                    f"mean_reward={metrics['reward/mean']:.3f} "
                    f"correct_rate={metrics['reward/correct']:.3f} "
                    f"mean_search_calls={metrics['rollout/search_calls']:.2f} "
                    f"input_tokens={int(metrics['train/tokens_per_rollout_batch'])} "
                    f"loss_tokens={int(metrics['train/loss_tokens_per_rollout_batch'])} "
                    f"padded_tokens={int(metrics['train/padded_tokens_per_rollout_batch'])}"
                )

                if args.save_every > 0 and (step + 1) % args.save_every == 0:
                    train_bar.set_postfix(phase="checkpoint", refresh=True)
                    save_checkpoint(training_client, f"{args.run_name}-step-{step + 1}")

        save_checkpoint(training_client, f"{args.run_name}-final")
    finally:
        run.finish()


def serializable_config(args: argparse.Namespace) -> dict[str, Any]:
    """Return a SwanLab-safe config dictionary."""
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def _init_swanlab(args: argparse.Namespace) -> Any:
    if args.swanlab_mode == "disabled":
        return _DisabledRun()
    global _SWANLAB_MODULE
    project_env = os.environ.pop("SWANLAB_PROJECT", None)
    try:
        import swanlab

        _SWANLAB_MODULE = swanlab
        return swanlab.init(
            project=args.swanlab_project,
            name=args.run_name,
            mode=args.swanlab_mode,
            config=serializable_config(args),
        )
    finally:
        if project_env is not None:
            os.environ["SWANLAB_PROJECT"] = project_env


def _log_swanlab(args: argparse.Namespace, metrics: dict[str, float], step: int) -> None:
    if args.swanlab_mode != "disabled":
        if _SWANLAB_MODULE is None:
            raise RuntimeError("swanlab run is not initialized")
        _SWANLAB_MODULE.log(metrics, step=step)


class _DisabledRun:
    def finish(self) -> None:
        return None


def _resolve_swanlab_project(cli_project: str | None) -> str:
    project = cli_project or os.getenv("SWANLAB_PROJECT") or DEFAULT_SWANLAB_PROJECT
    return project.rsplit("/", maxsplit=1)[-1]


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


def _require_reference_logprobs(item: Any) -> list[float]:
    if item.reference_logprobs is None:
        raise ValueError("KL training requires reference logprobs on every datum")
    return item.reference_logprobs


def _write_step_artifacts(
    args: argparse.Namespace,
    step: int,
    trajectories: list,
    metrics: dict[str, float],
) -> None:
    output_dir = args.trajectory_output_dir / args.run_name
    records = [trajectory_to_record(item, run_type="train") for item in trajectories]
    jsonl_path = output_dir / f"step_{step + 1:06d}.jsonl"
    report_path = output_dir / f"step_{step + 1:06d}.md"
    write_trajectory_jsonl(records, jsonl_path)
    report = build_markdown_report(
        records,
        title=f"PyTRIO Train Step {step + 1}: {args.backend}",
    )
    report_path.write_text(
        f"{report}\n\n## Metrics\n\n```json\n"
        f"{json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True)}\n"
        "```\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
