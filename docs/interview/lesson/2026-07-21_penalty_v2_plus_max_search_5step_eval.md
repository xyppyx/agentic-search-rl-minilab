# Penalty v2 + Max-Search 0.01 5-Step 复盘

复盘时间：2026-07-21 22:05 CST

## 实验目标

回答三个问题：

1. 当前 reward/loss 震荡是否需要通过扩大 group size 或轨迹数量缓解。
2. 在线搜索环境是否是高方差来源，以及本地检索在 Search-R1 类实验中的作用。
3. 在 `penalty_v2_candidate` 基础上轻微加回 `max_search_no_answer_penalty`，验证是否能压住 20-step v2 暴露出的 max-search 空转问题。

本次开启 SwanLab 在线记录，但不在公开文档中记录 SwanLab 私有 run 链接或 PyTRIO 远端 checkpoint URI。

## 判断

### Group size / 轨迹数量

扩大 group size 能降低同题 GRPO advantage 估计方差，因为每道题有更多候选轨迹用于比较。但它不是免费收益：

- `group_size=4`：每步 2 题、8 条 trajectory，方差大但成本低。
- `group_size=8`：每步 2 题、16 条 trajectory，advantage 更稳，但搜索/API/采样成本接近翻倍。
- 如果预算固定，扩大 group size 会减少可跑 step 数；是否更好取决于当前瓶颈是“组内比较太噪”还是“训练覆盖题目太少”。

当前 20-step v2 的每步指标确实高方差：reward mean 从 -0.1113 到 0.7250，correct 从 0 到 0.75，format 从 0 到 1.0，平均搜索从 1.0 到 3.75。直接把 50-step 当作解决方案不合适；更稳妥的是先做小预算 reward ablation，再考虑 `group_size=8` 的 5-step 对照。

### 在线搜索 vs 本地检索

在线搜索是高方差来源之一，但不是唯一来源。当前训练还同时受到 `temperature=1.0` 采样、稀疏 EM reward、小 batch、group-relative advantage、格式解析和严格 EM 的影响。

本地/固定检索环境的价值主要是：

- 固定语料和索引，减少同一 query 在不同时间返回不同结果。
- 避免 API 额度、429、服务延迟和服务端策略变化污染 reward。
- 提高吞吐和可复现性，便于做长训练和多 seed。

但本项目当前 `local_bm25` 只是 smoke/mock 级语料，完整 dev 上空结果率曾达到 56.92%，不适合作为真实搜索能力替代。要真正降低搜索方差，需要更接近论文基础设施的固定语料检索器，而不是直接使用当前小 fixture BM25。

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
- run name：`reward-penalty-v2-maxsearch001-5step-20260721`

Reward 参数：

- `duplicate_query_penalty=0.03`
- `empty_result_penalty=0.01`
- `max_search_no_answer_penalty=0.01`
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
  --max-search-no-answer-penalty 0.01 \
  --verbose-answer-penalty 0.0 \
  --verbose-answer-token-threshold 0 \
  --swanlab-mode online \
  --save-every 0 \
  --run-name reward-penalty-v2-maxsearch001-5step-20260721
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
  --jsonl-output my-search-r1/eval_results/reward_train_compare_2026-07-21/penalty_v2_maxsearch001_5step_dev.jsonl \
  --report-output my-search-r1/eval_results/reward_train_compare_2026-07-21/penalty_v2_maxsearch001_5step_dev.md
```

## 训练过程

5 个 rollout step 全部完成，每步 2 个问题、8 条 trajectory，共 40 条训练 trajectory。5/5 个 step 都产生 optimizer update。

训练总体：

- 平均 reward：0.0737
- 平均 correct rate：0.1250
- 平均 format rate：0.5500
- 平均搜索次数：2.4250
- 平均 `too_many_search_no_gain_rate`：0.4250
- SwanLab 上传完成：483 条 records

训练输出：

- `my-search-r1/outputs/train_pytrio/reward-penalty-v2-maxsearch001-5step-20260721/`

## Dev Eval 结果

| 指标 | base | 5-step 原始 reward | 5-step penalty v2 | 20-step penalty v2 | 5-step v2 + max-search 0.01 |
| --- | ---: | ---: | ---: | ---: | ---: |
| trajectories | 70 | 70 | 70 | 70 | 70 |
| correct count | 17 | 21 | 19 | 25 | 21 |
| `em/macro` | 0.2429 | 0.3000 | 0.2714 | 0.3571 | 0.3000 |
| format count | 43 | 58 | 58 | 51 | 61 |
| `format/rate` | 0.6143 | 0.8286 | 0.8286 | 0.7286 | 0.8714 |
| `rollout/search_calls` | 1.9571 | 1.7714 | 1.7571 | 2.6143 | 1.4571 |
| `behavior/max_search_no_answer_rate` | 0.2143 | 0.1143 | 0.1143 | 0.2714 | 0.0857 |
| `behavior/too_many_search_no_gain_rate` | 0.3000 | 0.2143 | 0.2143 | 0.4286 | 0.1000 |
| `behavior/duplicate_query_rate` | 0.0143 | 0.0286 | 0.0000 | 0.0429 | 0.0000 |
| Zhihu requests | 137 | 124 | 123 | 183 | 102 |
| Zhihu success rate | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| tool failures | 0 | 0 | 0 | 0 | 0 |

Eval 输出：

- `my-search-r1/eval_results/reward_train_compare_2026-07-21/penalty_v2_maxsearch001_5step_dev.jsonl`
- `my-search-r1/eval_results/reward_train_compare_2026-07-21/penalty_v2_maxsearch001_5step_dev.md`

## Offline Diagnostics

| Run | Wrong-valid | Alias | Granularity | Missing follow-up |
| --- | ---: | ---: | ---: | ---: |
| base | 26 | 10 | 0 | 0 |
| 5-step 原始 reward | 37 | 9 | 0 | 3 |
| 5-step penalty v2 | 39 | 11 | 0 | 2 |
| 20-step penalty v2 | 26 | 8 | 0 | 1 |
| 5-step v2 + max-search 0.01 | 40 | 9 | 2 | 5 |

输出：

- `my-search-r1/eval_results/reward_train_compare_2026-07-21/penalty_v2_maxsearch001_5step_offline_diagnostics.jsonl`
- `my-search-r1/eval_results/reward_train_compare_2026-07-21/penalty_v2_maxsearch001_5step_offline_diagnostics.md`

## Reward Sensitivity

对本 checkpoint 的 dev trajectory 做离线重评分：

| Config | mean_base_reward | mean_final_reward | mean_delta | penalized | correct penalized | wrong-valid penalized | missing-followup penalized | alias penalized | granularity penalized |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `penalty_v1` | 0.2871 | 0.2803 | -0.0069 | 12 | 0 | 6 | 1 | 1 | 0 |
| `penalty_v2_candidate` | 0.2871 | 0.2869 | -0.0003 | 2 | 0 | 1 | 0 | 0 | 0 |
| `penalty_v2_no_empty` | 0.2871 | 0.2871 | 0.0000 | 0 | 0 | 0 | 0 | 0 | 0 |

输出：

- `my-search-r1/eval_results/reward_train_compare_2026-07-21/penalty_v2_maxsearch001_5step_reward_sensitivity_summary.json`
- `my-search-r1/eval_results/reward_train_compare_2026-07-21/penalty_v2_maxsearch001_5step_reward_sensitivity.jsonl`
- `my-search-r1/eval_results/reward_train_compare_2026-07-21/penalty_v2_maxsearch001_5step_reward_sensitivity.md`

## 结论

`max_search_no_answer_penalty=0.01` 是一个有效的行为约束：相对 5-step penalty v2，它把平均搜索次数从 1.7571 降到 1.4571，把 `too_many_search_no_gain_rate` 从 0.2143 降到 0.1000，把 format rate 从 0.8286 提高到 0.8714，同时 EM 提升到 0.3000，追平 5-step 原始 reward。

但 diagnostics 暴露出代价：`missing_followup_query=5`、`answer_granularity_miss=2`，高于 5-step penalty v2 和 20-step penalty v2。这说明 max-search penalty 可能确实压住了空转，但也可能让一部分需要继续查的多跳题过早停止，或诱导更粗粒度答案。

当前决策：

- 这个 5-step checkpoint 是当前“format/搜索效率最干净”的候选之一。
- 不建议直接把该配置推到 50-step；应先做 gained/lost case review，确认新增 EM 与 missing-followup/granularity 风险来自哪些样本。
- 若要扩大训练，优先做 `group_size=8` 的 5-step 对照，或者做 `max_search_no_answer_penalty=0.005` 的更温和版本。
