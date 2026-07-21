# Penalty v2 5-Step Checkpoint 复盘

复盘时间：2026-07-21 21:30 CST

## 实验目标

验证 `penalty_v2_candidate` 在 5-step Zhihu GRPO 小预算下是否比 `penalty_v1` 更稳，并判断是否满足进入 20-step 训练的门槛。

本次开启 SwanLab 在线记录，但不在公开文档中记录 SwanLab 私有 run 链接或 PyTRIO 远端 checkpoint URI；复跑 checkpoint eval 时应从本地终端日志或 PyTRIO 控制台查找对应 final sampler weights。

## 实验配置

- 模型：`Qwen/Qwen3.5-4B`
- 训练数据：`my-search-r1/datasets/train.jsonl`
- 评测数据：`my-search-r1/datasets/dev.jsonl`，70 条
- 搜索 backend：`zhihu_search`
- seed：`42`
- eval decoding：`temperature=0.0`，`top_p=1.0`
- train decoding：`temperature=1.0`，`top_p=1.0`
- 训练预算：`max_steps=5`，`questions_per_batch=2`，`group_size=4`
- SwanLab：`online`
- run name：`reward-penalty-v2-5step-20260721`

Penalty v2 参数：

- `duplicate_query_penalty=0.03`
- `empty_result_penalty=0.01`
- `max_search_no_answer_penalty=0.0`
- `verbose_answer_penalty=0.0`
- `verbose_answer_token_threshold=0`

## 前置修复

本次首次打开 SwanLab online 时遇到两个环境问题：

- `swanlab==0.8.4` 被服务端拒绝，新版本项目要求 SDK `>=0.9.0`。
- `.env` 中的 `SWANLAB_PROJECT` 使用 `owner/project` 形式时，SwanLab SDK 会把它作为内部 settings 字段解析，导致 `project.name` 校验失败。

修复：

- 使用 `uv add 'swanlab>=0.9.0'` 将 SwanLab SDK 升级到 `0.9.0`。
- `train_pytrio.py` 延迟导入 SwanLab，并在初始化前将 `SWANLAB_PROJECT` 规范化为项目名，避免 `owner/` 前缀进入 SDK settings。

验证：

- `PYTHONPATH=my-search-r1 uv run python -m unittest discover -s my-search-r1/tests -v`：51 个 unittest 全部通过。
- `PYTHONPATH=my-search-r1 uv run python -m compileall -q my-search-r1/search_r1_minilab my-search-r1/scripts`：通过。
- 在已 source `.env` 的环境下运行 `train_pytrio.py --help`：通过，不再触发 SwanLab settings 解析错误。

## 运行命令

训练命令使用 `.env` 中的 `PYTRIO_API_KEY`、`ZHIHU_API_KEY`、`SWANLAB_API_KEY` 和 `SWANLAB_PROJECT`。真实 key 不写入本文档。

```bash
timeout 3600s env PYTHONPATH=my-search-r1 uv run python my-search-r1/scripts/train_pytrio.py \
  --data my-search-r1/datasets/train.jsonl \
  --max-steps 5 \
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
  --run-name reward-penalty-v2-5step-20260721
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
  --jsonl-output my-search-r1/eval_results/reward_train_compare_2026-07-21/penalty_v2_ckpt_dev.jsonl \
  --report-output my-search-r1/eval_results/reward_train_compare_2026-07-21/penalty_v2_ckpt_dev.md
```

## 训练过程

5 个 rollout step 全部完成，每步 2 个问题、8 条 trajectory，共 40 条训练 trajectory。有效 optimizer update 为 4/5；step 3 因 batch 内 group-relative advantage 全为 0 被跳过。

| Step | Reward mean | Correct | Format | Search calls | Update skipped | Loss mean |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.3250 | 0.3750 | 0.5000 | 2.5000 | 0 | -0.004472 |
| 2 | 0.2375 | 0.2500 | 0.8750 | 1.3750 | 0 | -0.001981 |
| 3 | 0.5000 | 0.5000 | 1.0000 | 2.1250 | 1 | N/A |
| 4 | -0.1075 | 0.0000 | 0.0000 | 3.0000 | 0 | 0.000125 |
| 5 | -0.0625 | 0.0000 | 0.3750 | 2.6250 | 0 | 0.000220 |

训练总体：

- 平均 reward：0.1785
- 平均 correct rate：0.2250
- 平均 format rate：0.5500
- 平均搜索次数：2.3250
- Zhihu backend eval 前训练阶段最终累计请求数：93，`success_rate=1.0`
- SwanLab 上传完成：480 条 records

训练输出：

- `my-search-r1/outputs/train_pytrio/reward-penalty-v2-5step-20260721/`

## Dev Eval 结果

| 指标 | base | 5-step 原始 reward | 5-step penalty v1 | 20-step 原始 reward | 5-step penalty v2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| trajectories | 70 | 70 | 70 | 70 | 70 |
| correct count | 17 | 21 | 16 | 19 | 19 |
| `em/macro` | 0.2429 | 0.3000 | 0.2286 | 0.2714 | 0.2714 |
| format count | 43 | 58 | 51 | 60 | 58 |
| `format/rate` | 0.6143 | 0.8286 | 0.7286 | 0.8571 | 0.8286 |
| `rollout/search_calls` | 1.9571 | 1.7714 | 1.5143 | 1.7000 | 1.7571 |
| `behavior/duplicate_query_rate` | 0.0143 | 0.0286 | 0.0143 | 0.0000 | 0.0000 |
| `behavior/empty_observation_rate` | 0.0511 | 0.0484 | 0.0472 | 0.0504 | 0.0488 |
| `behavior/max_search_no_answer_rate` | 0.2143 | 0.1143 | 0.1143 | 0.1286 | 0.1143 |
| `behavior/too_many_search_no_gain_rate` | 0.3000 | 0.2143 | 0.1571 | 0.2143 | 0.2143 |
| Zhihu requests | 137 | 124 | 106 | 119 | 123 |
| Zhihu success rate | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| tool failures | 0 | 0 | 0 | 0 | 0 |

Penalty v2 分来源 EM：

| Source | EM |
| --- | ---: |
| `2wikimultihopqa` | 0.2000 |
| `bamboogle` | 0.6000 |
| `hotpotqa` | 0.4000 |
| `musique` | 0.1000 |
| `nq` | 0.1000 |
| `popqa` | 0.2000 |
| `triviaqa` | 0.3000 |

Eval 输出：

- `my-search-r1/eval_results/reward_train_compare_2026-07-21/penalty_v2_ckpt_dev.jsonl`
- `my-search-r1/eval_results/reward_train_compare_2026-07-21/penalty_v2_ckpt_dev.md`

## Offline Diagnostics

| Run | Wrong-valid | Alias | Granularity | Missing follow-up |
| --- | ---: | ---: | ---: | ---: |
| base | 26 | 10 | 0 | 0 |
| 5-step 原始 reward | 37 | 9 | 0 | 3 |
| 5-step penalty v1 | 35 | 10 | 0 | 2 |
| 20-step 原始 reward | 41 | 8 | 1 | 6 |
| 5-step penalty v2 | 39 | 11 | 0 | 2 |

Penalty v2 没有新增答案粒度风险，`missing_followup_query=2`，低于 20-step 原始 reward 的 6，持平 5-step penalty v1。

输出：

- `my-search-r1/eval_results/reward_train_compare_2026-07-21/penalty_v2_ckpt_offline_diagnostics.jsonl`
- `my-search-r1/eval_results/reward_train_compare_2026-07-21/penalty_v2_ckpt_offline_diagnostics.md`

## Reward Sensitivity

对 penalty v2 checkpoint 的 dev trajectory 做离线重评分：

| Config | mean_base_reward | mean_final_reward | mean_delta | penalized | correct penalized | wrong-valid penalized | missing-followup penalized | alias penalized | granularity penalized |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `penalty_v1` | 0.2543 | 0.2453 | -0.0090 | 15 | 0 | 7 | 1 | 3 | 0 |
| `penalty_v2_candidate` | 0.2543 | 0.2539 | -0.0004 | 3 | 0 | 1 | 0 | 0 | 0 |
| `penalty_v2_no_empty` | 0.2543 | 0.2543 | 0.0000 | 0 | 0 | 0 | 0 | 0 | 0 |

输出：

- `my-search-r1/eval_results/reward_train_compare_2026-07-21/penalty_v2_ckpt_reward_sensitivity_summary.json`
- `my-search-r1/eval_results/reward_train_compare_2026-07-21/penalty_v2_ckpt_reward_sensitivity.jsonl`
- `my-search-r1/eval_results/reward_train_compare_2026-07-21/penalty_v2_ckpt_reward_sensitivity.md`

## 结论

`penalty_v2_candidate` 达到了进入 20-step 小扩展的最低门槛：EM 高于 base，format 达到 0.8286，平均搜索次数和 `too_many_search_no_gain_rate` 均低于 base，且 `missing_followup_query`、`answer_granularity_miss` 没有恶化。

但它没有超过 5-step 原始 reward checkpoint：EM 为 0.2714，低于原始 reward 5-step 的 0.3000；format 与原始 reward 5-step 持平，搜索次数略低。相比 `penalty_v1`，v2 的改进明显：EM 从 0.2286 回升到 0.2714，format 从 0.7286 回升到 0.8286，同时仍保留部分搜索行为收敛。

下一步可以跑 `penalty_v2_candidate` 的 20-step 版本，但叙事上应定义为“验证保守 penalty 是否在更长训练中保持稳定”，而不是预设它会超过当前最强的 5-step 原始 reward。若要定位 empty penalty 的因果作用，应补一个 `penalty_v2_no_empty` 5-step ablation。
