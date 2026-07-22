# Prompt Constraints Zhihu Dev Eval

## 实验目标

验证 2026-07-22 已实现的 prompt/rollout 层多跳 follow-up 与短答案格式约束是否改善 Zhihu dev 评测表现。先跑 dev-5 smoke，确认 Zhihu backend、PyTRIO sampling 与新 prompt 链路可用且无工具失败；再跑完整 dev 70，并与 prompt 前 base 以及当前均衡候选 v3 group size 4 对照。

重点观察指标：EM、format、平均搜索次数、`missing_followup_query`、`multi_candidate_answer`、`answer_granularity_miss`。

## 数据、模型与 backend

- 数据：`my-search-r1/datasets/dev.jsonl`
- 模型：默认 `Qwen/Qwen3.5-4B`
- checkpoint：无 `--model-path`，即 base model + 当前 prompt/rollout 约束
- backend：`zhihu_search`
- seed：`42`
- batch size：`1`
- failure injection：未开启，`p_timeout/p_empty/p_noise/p_rate_limited=0`
- 远程资源：PyTRIO sampling service 与本地 `.env` 中的 Zhihu/PyTRIO 凭据；未观测费用、显存或训练 loss

## 命令

dev-5：

```bash
PYTHONPATH=my-search-r1 uv run python my-search-r1/scripts/eval_pytrio.py \
  --data my-search-r1/datasets/dev.jsonl \
  --backend zhihu_search \
  --env-file my-search-r1/.env \
  --limit 5 \
  --batch-size 1 \
  --seed 42 \
  --jsonl-output my-search-r1/eval_results/reward_train_compare_2026-07-21/prompt_constraints_dev5_20260722.jsonl \
  --report-output my-search-r1/eval_results/reward_train_compare_2026-07-21/prompt_constraints_dev5_20260722.md
```

dev 70：

```bash
PYTHONPATH=my-search-r1 uv run python my-search-r1/scripts/eval_pytrio.py \
  --data my-search-r1/datasets/dev.jsonl \
  --backend zhihu_search \
  --env-file my-search-r1/.env \
  --limit 0 \
  --batch-size 1 \
  --seed 42 \
  --jsonl-output my-search-r1/eval_results/reward_train_compare_2026-07-21/prompt_constraints_dev_20260722.jsonl \
  --report-output my-search-r1/eval_results/reward_train_compare_2026-07-21/prompt_constraints_dev_20260722.md
```

offline diagnostics：

```bash
PYTHONPATH=my-search-r1 uv run python my-search-r1/scripts/analyse_offline_diagnostics.py \
  --input my-search-r1/eval_results/reward_train_compare_2026-07-21/prompt_constraints_dev_20260722.jsonl \
  --jsonl-output my-search-r1/eval_results/reward_train_compare_2026-07-21/prompt_constraints_dev_20260722_offline_diagnostics.jsonl \
  --report-output my-search-r1/eval_results/reward_train_compare_2026-07-21/prompt_constraints_dev_20260722_offline_diagnostics.md \
  --title 'Prompt Constraints Dev Offline Diagnostics'
```

## 输出

- `my-search-r1/eval_results/reward_train_compare_2026-07-21/prompt_constraints_dev5_20260722.jsonl`
- `my-search-r1/eval_results/reward_train_compare_2026-07-21/prompt_constraints_dev5_20260722.md`
- `my-search-r1/eval_results/reward_train_compare_2026-07-21/prompt_constraints_dev5_20260722_offline_diagnostics.jsonl`
- `my-search-r1/eval_results/reward_train_compare_2026-07-21/prompt_constraints_dev5_20260722_offline_diagnostics.md`
- `my-search-r1/eval_results/reward_train_compare_2026-07-21/prompt_constraints_dev_20260722.jsonl`
- `my-search-r1/eval_results/reward_train_compare_2026-07-21/prompt_constraints_dev_20260722.md`
- `my-search-r1/eval_results/reward_train_compare_2026-07-21/prompt_constraints_dev_20260722_offline_diagnostics.jsonl`
- `my-search-r1/eval_results/reward_train_compare_2026-07-21/prompt_constraints_dev_20260722_offline_diagnostics.md`

## 观测指标

| Run | N | EM | Format | Avg search | missing_followup_query | multi_candidate_answer | answer_granularity_miss | bad_max_search_loop | Tool failures |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| old dev-5 pre-prompt | 5 | 0.0000 | 0.2000 | 2.0000 | 0 | 0 | 0 | 0 | 0 |
| prompt constraints dev-5 | 5 | 0.2000 | 0.4000 | 3.2000 | 0 | 0 | 0 | 0 | 0 |
| base pre-prompt dev 70 | 70 | 0.2429 | 0.6143 | 1.9571 | 0 | 2 | 0 | 3 | 0 |
| v3 group size 4 dev 70 | 70 | 0.3000 | 0.8571 | 1.5143 | 6 | 1 | 0 | 2 | 0 |
| prompt constraints dev 70 | 70 | 0.3429 | 0.8714 | 1.7143 | 3 | 0 | 0 | 5 | 0 |

Zhihu backend 观测：

| Run | Requests | Success rate | Empty rate | Error rate | Timeout rate | Rate limit rate | Latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| prompt constraints dev-5 | 16 | 1.0000 | 0.0625 | 0.0000 | 0.0000 | 0.0000 | 1.0701 |
| prompt constraints dev 70 | 120 | 1.0000 | 0.0250 | 0.0000 | 0.0000 | 0.0000 | 1.0598 |

分类 EM：

| Source | EM |
| --- | ---: |
| 2wikimultihopqa | 0.3000 |
| bamboogle | 0.6000 |
| hotpotqa | 0.4000 |
| musique | 0.2000 |
| nq | 0.2000 |
| popqa | 0.2000 |
| triviaqa | 0.5000 |

## 对照结论

prompt 约束有明确正向效果，但不是纯收益。

相对 prompt 前 base dev 70，prompt 约束版 EM 从 0.2429 提升到 0.3429，净增 7 条 exact match；format 从 0.6143 提升到 0.8714，净增 18 条格式合规；平均搜索从 1.9571 降到 1.7143，总搜索请求从约 137 次降到 120 次。`multi_candidate_answer` 从 2 降到 0，`answer_granularity_miss` 保持 0。不过 `missing_followup_query` 从 0 增到 3，`bad_max_search_loop` 从 3 增到 5。

相对当前均衡候选 v3 group size 4，prompt 约束版 EM 从 0.3000 提升到 0.3429，净增 3 条 exact match；format 从 0.8571 小幅提升到 0.8714；`missing_followup_query` 从 6 降到 3，`multi_candidate_answer` 从 1 降到 0。但平均搜索从 1.5143 增到 1.7143，总搜索请求增加 14 次，`bad_max_search_loop` 从 2 增到 5。

这说明 prompt 的短答案和 follow-up 提示确实改善了最终答案格式、答案唯一性和一部分多跳行为，也提升了 base 模型 dev EM；同时它可能诱导少数样本继续搜索到 `max_search_calls`，需要在下一轮训练或 rollout 约束中抑制无收益搜索循环。

## 问题与经验

- dev-5 小样本方向与完整 dev 不完全一致：dev-5 平均搜索 3.2，看起来很差；完整 dev 70 平均搜索回落到 1.7143。因此 prompt 变更不能只看 dev-5。
- 本轮没有 Zhihu API 失败，完整 dev 70 的 success rate 为 1.0，error/timeout/rate-limit 均为 0。
- prompt 约束版可以进入小预算训练候选，但训练策略应继续保留 follow-up-aware max-search/bad-loop 抑制；否则搜索循环风险可能被放大。
