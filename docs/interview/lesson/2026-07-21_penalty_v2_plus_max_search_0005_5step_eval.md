# Penalty v2 + Max-Search 0.005 5-Step 复盘

复盘时间：2026-07-21 22:14 CST

## 实验目标

在 `penalty_v2_plus_max_search_001` 暴露出 `missing_followup_query=5`、`answer_granularity_miss=2` 后，验证更温和的 `max_search_no_answer_penalty=0.005` 是否能保留搜索效率和 format 收益，同时减少过早停止或答案粒度退化。

本次开启 SwanLab 在线记录，但不在公开文档中记录 SwanLab 私有 run 链接或 PyTRIO 远端 checkpoint URI。

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
- run name：`reward-penalty-v2-maxsearch0005-5step-20260721`

Reward 参数：

- `duplicate_query_penalty=0.03`
- `empty_result_penalty=0.01`
- `max_search_no_answer_penalty=0.005`
- `verbose_answer_penalty=0.0`
- `verbose_answer_token_threshold=0`

## 运行命令

训练：

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
  --max-search-no-answer-penalty 0.005 \
  --verbose-answer-penalty 0.0 \
  --verbose-answer-token-threshold 0 \
  --swanlab-mode online \
  --save-every 0 \
  --run-name reward-penalty-v2-maxsearch0005-5step-20260721
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
  --jsonl-output my-search-r1/eval_results/reward_train_compare_2026-07-21/penalty_v2_maxsearch0005_5step_dev.jsonl \
  --report-output my-search-r1/eval_results/reward_train_compare_2026-07-21/penalty_v2_maxsearch0005_5step_dev.md
```

## 训练过程

5 个 rollout step 全部完成，每步 2 个问题、8 条 trajectory，共 40 条训练 trajectory。5/5 个 step 都产生 optimizer update。

训练总体：

- 平均 reward：0.1535
- correct rate：0.2000
- format rate：0.5750
- 平均搜索次数：2.3000
- `max_search_no_answer`：12/40
- `too_many_search_no_gain`：15/40
- 工具失败数：0
- SwanLab 上传完成：483 条 records
- 训练耗时：约 3 分 12 秒

训练输出：

- `my-search-r1/outputs/train_pytrio/reward-penalty-v2-maxsearch0005-5step-20260721/`

逐步观察：

| Step | reward | correct | format | avg search | max-search no-answer | too-many search no-gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.3225 | 3/8 | 4/8 | 2.50 | 4/8 | 4/8 |
| 2 | 0.2250 | 2/8 | 6/8 | 1.38 | 0/8 | 0/8 |
| 3 | 0.3606 | 3/8 | 7/8 | 2.25 | 1/8 | 1/8 |
| 4 | -0.1150 | 0/8 | 0/8 | 3.00 | 6/8 | 6/8 |
| 5 | -0.0256 | 0/8 | 6/8 | 2.38 | 1/8 | 4/8 |

第 4、5 步质量明显下降，说明 5-step 小预算仍有较强 batch 方差；本轮不能只看训练 reward 判断配置优劣。

## Dev Eval 结果

| 指标 | base | 5-step 原始 reward | 5-step penalty v2 | 20-step penalty v2 | 5-step v2 + max-search 0.01 | 5-step v2 + max-search 0.005 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| trajectories | 70 | 70 | 70 | 70 | 70 | 70 |
| correct count | 17 | 21 | 19 | 25 | 21 | 19 |
| `em/macro` | 0.2429 | 0.3000 | 0.2714 | 0.3571 | 0.3000 | 0.2714 |
| format count | 43 | 58 | 58 | 51 | 61 | 57 |
| `format/rate` | 0.6143 | 0.8286 | 0.8286 | 0.7286 | 0.8714 | 0.8143 |
| `rollout/search_calls` | 1.9571 | 1.7714 | 1.7571 | 2.6143 | 1.4571 | 1.5714 |
| `behavior/max_search_no_answer_rate` | 0.2143 | 0.1143 | 0.1143 | 0.2714 | 0.0857 | 0.1143 |
| `behavior/too_many_search_no_gain_rate` | 0.3000 | 0.2143 | 0.2143 | 0.4286 | 0.1000 | 0.1714 |
| `behavior/duplicate_query_rate` | 0.0143 | 0.0286 | 0.0000 | 0.0429 | 0.0000 | 0.0143 |
| Zhihu requests | 137 | 124 | 123 | 183 | 102 | 110 |
| Zhihu success rate | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| tool failures | 0 | 0 | 0 | 0 | 0 | 0 |

Eval 输出：

- `my-search-r1/eval_results/reward_train_compare_2026-07-21/penalty_v2_maxsearch0005_5step_dev.jsonl`
- `my-search-r1/eval_results/reward_train_compare_2026-07-21/penalty_v2_maxsearch0005_5step_dev.md`

## Offline Diagnostics

| Run | Wrong-valid | Alias | Granularity | Missing follow-up |
| --- | ---: | ---: | ---: | ---: |
| base | 26 | 10 | 0 | 0 |
| 5-step 原始 reward | 37 | 9 | 0 | 3 |
| 5-step penalty v2 | 39 | 11 | 0 | 2 |
| 20-step penalty v2 | 26 | 8 | 0 | 1 |
| 5-step v2 + max-search 0.01 | 40 | 9 | 2 | 5 |
| 5-step v2 + max-search 0.005 | 38 | 11 | 0 | 5 |

输出：

- `my-search-r1/eval_results/reward_train_compare_2026-07-21/penalty_v2_maxsearch0005_5step_offline_diagnostics.jsonl`
- `my-search-r1/eval_results/reward_train_compare_2026-07-21/penalty_v2_maxsearch0005_5step_offline_diagnostics.md`

## Reward Sensitivity

对本 checkpoint 的 dev trajectory 做离线重评分：

| Config | mean_base_reward | mean_final_reward | mean_delta | penalized | correct penalized | wrong-valid penalized | missing-followup penalized | alias penalized | granularity penalized |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `penalty_v1` | 0.2529 | 0.2426 | -0.0103 | 14 | 0 | 6 | 1 | 1 | 0 |
| `penalty_v2_candidate` | 0.2529 | 0.2517 | -0.0011 | 6 | 0 | 2 | 0 | 0 | 0 |
| `penalty_v2_no_empty` | 0.2529 | 0.2524 | -0.0004 | 1 | 0 | 0 | 0 | 0 | 0 |

输出：

- `my-search-r1/eval_results/reward_train_compare_2026-07-21/penalty_v2_maxsearch0005_5step_reward_sensitivity_summary.json`
- `my-search-r1/eval_results/reward_train_compare_2026-07-21/penalty_v2_maxsearch0005_5step_reward_sensitivity.jsonl`
- `my-search-r1/eval_results/reward_train_compare_2026-07-21/penalty_v2_maxsearch0005_5step_reward_sensitivity.md`

## 结论

`max_search_no_answer_penalty=0.005` 没有超过 `0.01` 版本。相对 `0.01`，它的 EM 从 0.3000 降到 0.2714，format 从 0.8714 降到 0.8143，平均搜索从 1.4571 升到 1.5714，`too_many_search_no_gain_rate` 从 0.1000 升到 0.1714。

它的优点是 `answer_granularity_miss=0`，低于 `0.01` 版本的 2；但 `missing_followup_query=5` 没有改善。这说明把 max-search penalty 从 0.01 降到 0.005 并没有解决过早回答问题，反而损失了 format 和搜索效率收益。

当前决策：

- `0.005` 不作为优先扩大训练候选。
- `0.01` 仍是当前高 format/高搜索效率候选，但需要 gained/lost case review 确认 missing follow-up 与 granularity 风险。
- 下一步更有价值的是比较 `0.01`、`0.005` 和 20-step penalty v2 的 gained/lost 样本，或做 `penalty_v2_no_empty` / `group_size=8` 小预算对照。
