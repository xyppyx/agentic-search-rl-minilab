# Gated OPSD v2 实现与 Smoke

记录时间：2026-08-11 CST

## 目标

基于 `gated-opsd-guardfix-5step-20260811` 的负向结果，修正 OPSD v1 “会蒸馏 wrong-valid final answer” 的问题。v2 不改变 GRPO/turn-credit 主训练目标，只把 OPSD 辅助 token mask 收窄到更可信的正向 token。

公开复盘不记录 PyTRIO 远端 sampler weights URI、真实 API key、SwanLab 私有链接或本地敏感配置。

## 改动

代码改动：

- `my-search-r1/search_r1_minilab/training.py`
  - 新增 `OPSD_POSITIVE_POLICIES`：`all`、`positive_advantage`、`positive_reward`、`exact_match`。
  - `OPSDConfig` 新增 `positive_policy`，默认 `positive_advantage`。
  - OPSD 默认 `mask_policy` 从 v1 的 `final_and_credited` 改为更保守的 `credited_turns`。
  - `build_datum()` / `build_training_datums()` 支持 `opsd_positive_policy`。
  - `_opsd_turn_selected()` 在 mask policy 命中后，再用 positive gate 过滤。
- `my-search-r1/scripts/train_pytrio.py`
  - 新增 CLI：`--opsd-positive-policy`。
  - 打开 `--opsd-coef` 时默认使用 `credited_turns + positive_advantage`。
- `my-search-r1/tests/test_rollout_training.py`
  - 覆盖 v2 CLI 默认值。
  - 覆盖 `positive_advantage` 会保留正向 credited turn、排除负向 final answer。
  - 覆盖 `exact_match` gate 只蒸馏正确轨迹。

## 语义变化

v1：

```text
mask = final_answer OR credited_turn
```

v2 默认：

```text
mask = credited_turn AND turn_effective_advantage > 0
```

可选正向 gate：

| Policy | 语义 |
| --- | --- |
| `all` | 复现 v1 无正向 gate 行为 |
| `positive_advantage` | 只蒸馏 effective advantage 为正的 turn |
| `positive_reward` | 只蒸馏最终 reward 为正的 trajectory |
| `exact_match` | 只蒸馏 exact-match 正确 trajectory |

`positive_advantage` 对 turn-credit 更合适：wrong-valid trajectory 中被 turn-credit 识别出的有用搜索 turn 仍可蒸馏；负 advantage 的 final answer 不会被蒸馏。

## 验证

代码验证：

```bash
PYTHONPATH=my-search-r1 uv run python -m unittest my-search-r1/tests/test_rollout_training.py -v
PYTHONPATH=my-search-r1 uv run python -m compileall -q my-search-r1/search_r1_minilab my-search-r1/scripts
git diff --check
```

结果：

- `test_rollout_training.py`：43 tests OK。
- `compileall`：通过。
- `git diff --check`：通过。

Local BM25 v2 default smoke：

```bash
PYTHONPATH=my-search-r1 uv run python my-search-r1/scripts/train_pytrio.py \
  --max-steps 1 \
  --questions-per-batch 1 \
  --group-size 2 \
  --backend local_bm25 \
  --swanlab-mode disabled \
  --run-name gated-opsd-v2-local-smoke-20260811 \
  --opsd-coef 0.01 \
  --opsd-mask-policy credited_turns \
  --opsd-positive-policy positive_advantage \
  --opsd-min-teacher-logprob -3.0
```

结果：

| 指标 | 值 |
| --- | ---: |
| trajectories | 2 |
| correct | 1/2 |
| format | 1.0000 |
| local BM25 success rate | 1.0000 |
| OPSD masked tokens | 0 |
| update skipped | 0 |

该 fixture 未开启 turn-credit，`credited_turns` 没有命中，因此 masked tokens 为 0；这验证了默认 v2 不会在没有正向 turn-credit 时盲目蒸馏。

Local BM25 positive-final smoke：

```bash
PYTHONPATH=my-search-r1 uv run python my-search-r1/scripts/train_pytrio.py \
  --max-steps 1 \
  --questions-per-batch 1 \
  --group-size 2 \
  --backend local_bm25 \
  --swanlab-mode disabled \
  --run-name gated-opsd-v2-positive-final-smoke-20260811 \
  --opsd-coef 0.01 \
  --opsd-mask-policy final_answer \
  --opsd-positive-policy positive_advantage \
  --opsd-min-teacher-logprob -3.0
```

结果：

| 指标 | 值 |
| --- | ---: |
| trajectories | 2 |
| correct | 1/2 |
| format | 1.0000 |
| local BM25 success rate | 1.0000 |
| OPSD masked tokens | 5 |
| OPSD mask rate | 0.0039 |
| OPSD loss mean | 0.0871 |
| student-teacher gap | 0.0032 |
| update skipped | 0 |

该 smoke 验证了非零 v2 mask 下，OPSD teacher logprobs、custom backward 和 optimizer 路径仍可运行。

## 结论

OPSD v2 工程实现已完成并通过 local PyTRIO smoke。相比 v1，默认行为从“final answer/credited turn 都可能蒸馏”改为“只蒸馏正向 advantage 的 credited turn”，避免默认强化 wrong-valid final answer。

下一步不直接扩评，应先运行真实 Zhihu 5-step v2：

```bash
PYTHONPATH=my-search-r1 uv run python my-search-r1/scripts/train_pytrio.py \
  --data my-search-r1/datasets/train.jsonl \
  --max-steps 5 \
  --questions-per-batch 2 \
  --group-size 4 \
  --backend zhihu_search \
  --env-file my-search-r1/.env \
  --seed 42 \
  --temperature 1.0 \
  --top-p 1.0 \
  --swanlab-mode disabled \
  --save-every 5 \
  --run-name gated-opsd-v2-guardfix-5step-20260811 \
  --trajectory-output-dir my-search-r1/outputs/train_pytrio \
  --turn-credit-policy final_hop_bridge \
  --evidence-search-turn-bonus 0.05 \
  --final-hop-search-turn-bonus 0.10 \
  --early-answer-turn-penalty 0.05 \
  --missing-final-hop-turn-penalty 0.08 \
  --final-answer-guard-turn-penalty 0.06 \
  --opsd-coef 0.01 \
  --opsd-mask-policy credited_turns \
  --opsd-positive-policy positive_advantage \
  --opsd-min-teacher-logprob -3.0
```

进入 dev70 的门槛：真实训练阶段 Zhihu success rate 1.0、OPSD masked tokens 非零且 mask rate 不失控、无 PyTRIO sampling 阻塞。
