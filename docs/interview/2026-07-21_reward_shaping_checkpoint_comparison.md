# Reward Shaping Checkpoint 对比复盘

复盘时间：2026-07-21 19:40 CST

## 实验目标

比较三组 Zhihu dev 评测结果：

- `base`：未训练的 `Qwen/Qwen3.5-4B`。
- `baseline_reward_ckpt`：使用原始 reward 做 5-step GRPO 训练后的 final sampler weights。
- `penalty_reward_ckpt`：使用轻量 penalty reward 做 5-step GRPO 训练后的 final sampler weights。

本轮目标是验证 reward shaping 训练后 checkpoint 的短程效果，不公开记录 PyTRIO 远端 checkpoint URI。

## 实验配置

- 模型：`Qwen/Qwen3.5-4B`
- 训练数据：`my-search-r1/datasets/train.jsonl`
- 评测数据：`my-search-r1/datasets/dev.jsonl`，70 条
- 搜索 backend：`zhihu_search`
- seed：`42`
- eval decoding：`temperature=0.0`，`top_p=1.0`
- train decoding：`temperature=1.0`，`top_p=1.0`
- 训练预算：`max_steps=5`，`questions_per_batch=2`，`group_size=4`
- SwanLab：`disabled`
- baseline run：`reward-baseline-5step-20260721`
- penalty run：`reward-penalty-5step-20260721`

Penalty 参数：

- `duplicate_query_penalty=0.05`
- `empty_result_penalty=0.03`
- `max_search_no_answer_penalty=0.05`
- `verbose_answer_penalty=0.02`
- `verbose_answer_token_threshold=8`

## 运行命令

可用性 smoke：

```bash
timeout 180s env PYTHONPATH=my-search-r1 uv run python my-search-r1/scripts/eval_pytrio.py --backend local_bm25 --limit 1 --batch-size 1 --jsonl-output /tmp/reward_compare_local_smoke.jsonl --report-output /tmp/reward_compare_local_smoke.md
timeout 600s env PYTHONPATH=my-search-r1 uv run python my-search-r1/scripts/eval_pytrio.py --data my-search-r1/datasets/dev.jsonl --backend zhihu_search --limit 5 --batch-size 1 --seed 42 --temperature 0.0 --top-p 1.0 --jsonl-output /tmp/reward_compare_zhihu_smoke.jsonl --report-output /tmp/reward_compare_zhihu_smoke.md
```

Base eval：

```bash
timeout 1800s env PYTHONPATH=my-search-r1 uv run python my-search-r1/scripts/eval_pytrio.py --data my-search-r1/datasets/dev.jsonl --backend zhihu_search --limit 0 --batch-size 1 --seed 42 --temperature 0.0 --top-p 1.0 --jsonl-output my-search-r1/eval_results/reward_train_compare_2026-07-21/base_dev.jsonl --report-output my-search-r1/eval_results/reward_train_compare_2026-07-21/base_dev.md
```

训练：

```bash
timeout 3600s env PYTHONPATH=my-search-r1 uv run python my-search-r1/scripts/train_pytrio.py --data my-search-r1/datasets/train.jsonl --max-steps 5 --questions-per-batch 2 --group-size 4 --backend zhihu_search --seed 42 --temperature 1.0 --top-p 1.0 --swanlab-mode disabled --save-every 0 --run-name reward-baseline-5step-20260721
timeout 3600s env PYTHONPATH=my-search-r1 uv run python my-search-r1/scripts/train_pytrio.py --data my-search-r1/datasets/train.jsonl --max-steps 5 --questions-per-batch 2 --group-size 4 --backend zhihu_search --seed 42 --temperature 1.0 --top-p 1.0 --duplicate-query-penalty 0.05 --empty-result-penalty 0.03 --max-search-no-answer-penalty 0.05 --verbose-answer-penalty 0.02 --verbose-answer-token-threshold 8 --swanlab-mode disabled --save-every 0 --run-name reward-penalty-5step-20260721
```

Checkpoint eval 使用训练输出的 final sampler weights，通过 `--model-path` 传入；真实远端路径不写入本文档。

## 可用性 Smoke

- `local_bm25 --limit 1`：4 秒完成，生成 1 条 trajectory，`em/macro=1.0`、`format/rate=1.0`。
- `zhihu_search --limit 5`：约 21 秒完成，生成 5 条 trajectory，搜索请求 10 次，`success_rate=1.0`、无 timeout/429/error。

结论：2026-07-21 本地 PyTRIO sampling 与 Zhihu backend 均可用，已不复现 2026-07-20 卡在首条 `sample_async` 的问题。

## 训练过程

| Run | Updates | Mean reward | Mean correct | Mean search calls | 最后一轮 datums | 最后一轮 loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline reward | 4/5 | 0.1525 | 0.2000 | 2.4000 | 8 | 0.002171 |
| penalty reward | 5/5 | 0.0778 | 0.1500 | 2.3750 | 4 | -0.001173 |

逐步指标：

| Run | Step | Reward mean | Correct | Format | Search calls | Datums | Update skipped | Loss mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 1 | 0.3250 | 0.3750 | 0.5000 | 2.5000 | 4 | 0 | -0.001429 |
| baseline | 2 | 0.2250 | 0.2500 | 0.7500 | 1.2500 | 8 | 0 | -0.004257 |
| baseline | 3 | 0.3625 | 0.3750 | 0.8750 | 2.2500 | 4 | 0 | 0.002350 |
| baseline | 4 | -0.1000 | 0.0000 | 0.0000 | 3.5000 | 0 | 1 | N/A |
| baseline | 5 | -0.0500 | 0.0000 | 0.5000 | 2.5000 | 8 | 0 | 0.002171 |
| penalty | 1 | 0.1700 | 0.2500 | 0.5000 | 2.5000 | 4 | 0 | 0.002658 |
| penalty | 2 | 0.1200 | 0.1250 | 1.0000 | 1.3750 | 8 | 0 | -0.005374 |
| penalty | 3 | 0.3675 | 0.3750 | 1.0000 | 2.1250 | 8 | 0 | 0.001404 |
| penalty | 4 | -0.1638 | 0.0000 | 0.0000 | 3.1250 | 8 | 0 | 0.000458 |
| penalty | 5 | -0.1050 | 0.0000 | 0.2500 | 2.7500 | 4 | 0 | -0.001173 |

训练输出：

- `my-search-r1/outputs/train_pytrio/reward-baseline-5step-20260721/`
- `my-search-r1/outputs/train_pytrio/reward-penalty-5step-20260721/`

## Dev Eval 结果

| 指标 | base | baseline reward ckpt | penalty reward ckpt |
| --- | ---: | ---: | ---: |
| trajectories | 70 | 70 | 70 |
| `reward/mean` | 0.2043 | 0.2829 | 0.2014 |
| `em/macro` | 0.2429 | 0.3000 | 0.2286 |
| correct count | 17 | 21 | 16 |
| `format/rate` | 0.6143 | 0.8286 | 0.7286 |
| format count | 43 | 58 | 51 |
| `rollout/search_calls` | 1.9571 | 1.7714 | 1.5143 |
| `behavior/direct_correct_rate` | 0.0143 | 0.0000 | 0.0000 |
| `behavior/searched_correct_rate` | 0.2286 | 0.3000 | 0.2286 |
| `behavior/searched_wrong_rate` | 0.3714 | 0.5286 | 0.5000 |
| `behavior/duplicate_query_rate` | 0.0143 | 0.0286 | 0.0143 |
| `behavior/empty_observation_rate` | 0.0511 | 0.0484 | 0.0472 |
| `behavior/max_search_no_answer_rate` | 0.2143 | 0.1143 | 0.1143 |
| `behavior/too_many_search_no_gain_rate` | 0.3000 | 0.2143 | 0.1571 |
| tool failures | 0 | 0 | 0 |
| Zhihu requests | 137 | 124 | 106 |
| Zhihu success rate | 1.0000 | 1.0000 | 1.0000 |

Eval 输出：

- `my-search-r1/eval_results/reward_train_compare_2026-07-21/base_dev.jsonl`
- `my-search-r1/eval_results/reward_train_compare_2026-07-21/base_dev.md`
- `my-search-r1/eval_results/reward_train_compare_2026-07-21/baseline_reward_ckpt_dev.jsonl`
- `my-search-r1/eval_results/reward_train_compare_2026-07-21/baseline_reward_ckpt_dev.md`
- `my-search-r1/eval_results/reward_train_compare_2026-07-21/penalty_reward_ckpt_dev.jsonl`
- `my-search-r1/eval_results/reward_train_compare_2026-07-21/penalty_reward_ckpt_dev.md`

## 样本级变化

相对 base：

- baseline reward checkpoint：新增答对 8 条，丢失原答对 4 条，净增 4 条；format 新增合规 18 条，丢失 3 条。
- penalty reward checkpoint：新增答对 6 条，丢失原答对 7 条，净减 1 条；format 新增合规 13 条，丢失 5 条。

baseline reward checkpoint vs penalty reward checkpoint：

- penalty checkpoint 仅新增答对 1 条，但丢失 baseline checkpoint 的 6 条正确。
- penalty checkpoint 的平均搜索次数更低，`too_many_search_no_gain_rate` 更低，但 EM 和 format 都低于 baseline checkpoint。

## 结论

- 本轮 5-step 小训练中，原始 reward checkpoint 的 dev 表现最好：EM macro 从 base 的 0.2429 提升到 0.3000，format rate 从 0.6143 提升到 0.8286。
- penalty reward checkpoint 没有带来最终效果提升：EM macro 为 0.2286，低于 base 和 baseline reward checkpoint。
- penalty checkpoint 确实减少了平均搜索次数、Zhihu 请求数和 `too_many_search_no_gain_rate`，说明轻量 penalty 改变了行为倾向；但在 5-step 小预算下，这种行为约束损伤了答对率和格式合规。
- 当前更合理的下一步不是直接扩大 penalty 训练，而是先降低 penalty 强度或延后启用，并增加对 query 改写、证据阅读和格式收束的 reward/训练信号。

## 风险与注意

- 5-step、每步 2 个问题的训练预算很小，结论只能作为方向性 smoke，不是稳定训练收益证明。
- 两个训练 run 使用同一 seed，但真实在线搜索服务和远程采样仍可能引入轻微非确定性。
- 本文档不记录真实 PyTRIO checkpoint URI；需要复跑 checkpoint eval 时应从本地终端记录或 PyTRIO 控制台查找对应 run 的 final sampler weights。
