# Penalty v2 20-Step Checkpoint 复盘

复盘时间：2026-07-21 21:50 CST

## 实验目标

在 `penalty_v2_candidate` 5-step 已满足小扩展门槛后，继续跑 20-step Zhihu GRPO 稳定性实验，验证保守 penalty 在更长训练下是否能同时保持 EM、format 和搜索行为。

本次开启 SwanLab 在线记录，但不在公开文档中记录 SwanLab 私有 run 链接或 PyTRIO 远端 checkpoint URI；复跑 checkpoint eval 时应从本地终端日志或 PyTRIO 控制台查找对应 final sampler weights。

## 实验配置

- 模型：`Qwen/Qwen3.5-4B`
- 训练数据：`my-search-r1/datasets/train.jsonl`
- 评测数据：`my-search-r1/datasets/dev.jsonl`，70 条
- 搜索 backend：`zhihu_search`
- seed：`42`
- eval decoding：`temperature=0.0`，`top_p=1.0`
- train decoding：`temperature=1.0`，`top_p=1.0`
- 训练预算：`max_steps=20`，`questions_per_batch=2`，`group_size=4`
- SwanLab：`online`
- run name：`reward-penalty-v2-20step-20260721`

Penalty v2 参数：

- `duplicate_query_penalty=0.03`
- `empty_result_penalty=0.01`
- `max_search_no_answer_penalty=0.0`
- `verbose_answer_penalty=0.0`
- `verbose_answer_token_threshold=0`

## 运行命令

训练命令使用 `.env` 中的 `PYTRIO_API_KEY`、`ZHIHU_API_KEY`、`SWANLAB_API_KEY` 和 `SWANLAB_PROJECT`。真实 key 不写入本文档。

```bash
timeout 10800s env PYTHONPATH=my-search-r1 uv run python my-search-r1/scripts/train_pytrio.py \
  --data my-search-r1/datasets/train.jsonl \
  --max-steps 20 \
  --questions-per-batch 2 \
  --group-size 4 \
  --backend zhihu_search \
  --seed 42 \
  --temperature 1.0 \
  --top-p 1.0 \
  --duplicate-query-penalty 0.03 \
  --empty-result-penalty 0.01 \
  --max-search-no-answer-penalty 0.0 \
  --verbose-answer-penalty 0.0 \
  --verbose-answer-token-threshold 0 \
  --swanlab-mode online \
  --save-every 0 \
  --run-name reward-penalty-v2-20step-20260721
```

Checkpoint eval 使用训练输出的 final sampler weights，通过 `--model-path` 传入；真实远端路径不写入本文档。

```bash
timeout 3600s env PYTHONPATH=my-search-r1 uv run python my-search-r1/scripts/eval_pytrio.py \
  --data my-search-r1/datasets/dev.jsonl \
  --backend zhihu_search \
  --limit 0 \
  --batch-size 1 \
  --seed 42 \
  --temperature 0.0 \
  --top-p 1.0 \
  --model-path '<final-sampler-weights-uri>' \
  --jsonl-output my-search-r1/eval_results/reward_train_compare_2026-07-21/penalty_v2_20step_dev.jsonl \
  --report-output my-search-r1/eval_results/reward_train_compare_2026-07-21/penalty_v2_20step_dev.md
```

## 训练过程

20 个 rollout step 全部完成，每步 2 个问题、8 条 trajectory，共 160 条训练 trajectory。有效 optimizer update 为 18/20；step 3 和 step 10 因 batch 内 group-relative advantage 全为 0 被跳过。

训练总体：

- 平均 reward：0.2337
- 平均 correct rate：0.2687
- 平均 format rate：0.6562
- 平均搜索次数：2.1187
- final step：`reward/mean=-0.0125`、`reward/correct=0.0`、`reward/format=0.8750`、`rollout/search_calls=1.75`、`trainer/loss_mean=-0.000318`
- 训练阶段最终累计 Zhihu 请求数：339，`success_rate=1.0`
- SwanLab 上传完成：1699 条 records

训练期间 SwanLab 出现一次网络或服务端上传重试，SDK 自动恢复并最终上传完成。

训练输出：

- `my-search-r1/outputs/train_pytrio/reward-penalty-v2-20step-20260721/`

## Dev Eval 结果

| 指标 | base | 5-step 原始 reward | 20-step 原始 reward | 5-step penalty v2 | 20-step penalty v2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| trajectories | 70 | 70 | 70 | 70 | 70 |
| correct count | 17 | 21 | 19 | 19 | 25 |
| `em/macro` | 0.2429 | 0.3000 | 0.2714 | 0.2714 | 0.3571 |
| format count | 43 | 58 | 60 | 58 | 51 |
| `format/rate` | 0.6143 | 0.8286 | 0.8571 | 0.8286 | 0.7286 |
| `rollout/search_calls` | 1.9571 | 1.7714 | 1.7000 | 1.7571 | 2.6143 |
| `behavior/duplicate_query_rate` | 0.0143 | 0.0286 | 0.0000 | 0.0000 | 0.0429 |
| `behavior/empty_observation_rate` | 0.0511 | 0.0484 | 0.0504 | 0.0488 | 0.0273 |
| `behavior/max_search_no_answer_rate` | 0.2143 | 0.1143 | 0.1286 | 0.1143 | 0.2714 |
| `behavior/too_many_search_no_gain_rate` | 0.3000 | 0.2143 | 0.2143 | 0.2143 | 0.4286 |
| Zhihu requests | 137 | 124 | 119 | 123 | 183 |
| Zhihu success rate | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| tool failures | 0 | 0 | 0 | 0 | 0 |

Penalty v2 20-step 分来源 EM：

| Source | EM |
| --- | ---: |
| `2wikimultihopqa` | 0.2000 |
| `bamboogle` | 0.7000 |
| `hotpotqa` | 0.3000 |
| `musique` | 0.2000 |
| `nq` | 0.3000 |
| `popqa` | 0.2000 |
| `triviaqa` | 0.6000 |

Eval 输出：

- `my-search-r1/eval_results/reward_train_compare_2026-07-21/penalty_v2_20step_dev.jsonl`
- `my-search-r1/eval_results/reward_train_compare_2026-07-21/penalty_v2_20step_dev.md`

## Offline Diagnostics

| Run | Wrong-valid | Alias | Granularity | Missing follow-up |
| --- | ---: | ---: | ---: | ---: |
| base | 26 | 10 | 0 | 0 |
| 5-step 原始 reward | 37 | 9 | 0 | 3 |
| 20-step 原始 reward | 41 | 8 | 1 | 6 |
| 5-step penalty v2 | 39 | 11 | 0 | 2 |
| 20-step penalty v2 | 26 | 8 | 0 | 1 |

Penalty v2 20-step 没有新增答案粒度风险，`missing_followup_query=1`，低于 5-step penalty v2 和 20-step 原始 reward。

输出：

- `my-search-r1/eval_results/reward_train_compare_2026-07-21/penalty_v2_20step_offline_diagnostics.jsonl`
- `my-search-r1/eval_results/reward_train_compare_2026-07-21/penalty_v2_20step_offline_diagnostics.md`

## Reward Sensitivity

对 penalty v2 20-step checkpoint 的 dev trajectory 做离线重评分：

| Config | mean_base_reward | mean_final_reward | mean_delta | penalized | correct penalized | wrong-valid penalized | missing-followup penalized | alias penalized | granularity penalized |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `penalty_v1` | 0.3300 | 0.3119 | -0.0181 | 24 | 0 | 5 | 0 | 1 | 0 |
| `penalty_v2_candidate` | 0.3300 | 0.3283 | -0.0017 | 6 | 0 | 1 | 0 | 1 | 0 |
| `penalty_v2_no_empty` | 0.3300 | 0.3287 | -0.0013 | 3 | 0 | 1 | 0 | 1 | 0 |

输出：

- `my-search-r1/eval_results/reward_train_compare_2026-07-21/penalty_v2_20step_reward_sensitivity_summary.json`
- `my-search-r1/eval_results/reward_train_compare_2026-07-21/penalty_v2_20step_reward_sensitivity.jsonl`
- `my-search-r1/eval_results/reward_train_compare_2026-07-21/penalty_v2_20step_reward_sensitivity.md`

## 结论

`penalty_v2_candidate` 的 20-step checkpoint 在 EM 上取得当前最好结果：0.3571，高于 base、5-step 原始 reward、20-step 原始 reward和 5-step penalty v2。它也降低了 offline diagnostics 中的 missing follow-up 风险，且没有新增答案粒度风险。

但它不是全面更优。20-step penalty v2 的 format rate 降到 0.7286，低于 5-step penalty v2 的 0.8286 和 20-step 原始 reward 的 0.8571；平均搜索次数升到 2.6143，`too_many_search_no_gain_rate` 升到 0.4286，Zhihu 请求数也升到 183。也就是说，更长训练带来了更多答对样本，但策略重新变得更重搜索、更容易空转或到达 max-search。

当前决策：

- 可以把 20-step penalty v2 作为“最高 EM checkpoint”保留。
- 不能把它作为默认最优策略，因为 format 和搜索效率明显退化。
- 下一步不应继续盲目加步数；更合理的是做 `penalty_v2_no_empty` ablation 或提高/重引入“max-search no-answer”约束，同时增加 follow-up query 和 final-answer format 的正向信号。
