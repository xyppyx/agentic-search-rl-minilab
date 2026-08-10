# Reward v4 Follow-up Bonus Offline Sensitivity

## 目标

基于 `penalty_v3_followup_aware` 没有解决必要二跳缺失的问题，验证一个默认关闭的正向 follow-up reward：

- `duplicate_query_penalty=0.03`
- `empty_result_penalty=0.01`
- `bad_max_search_penalty=0.01`
- `date_granularity_penalty=0.05`
- `multi_candidate_answer_penalty=0.02`
- `helpful_followup_bonus=0.02`

设计意图是从 penalty-only 转向正向行为信号：对 valid format、wrong answer、存在 `helpful_followup_query` 且不是 `bad_max_search_loop` 的 trajectory 给小 bonus。正确答案和 invalid format 不加 bonus。

## 输入与命令

本轮只做离线重评分，不调用模型、知乎搜索 API、PyTRIO 训练或 SwanLab。

输入为 4 份既有 Zhihu dev 70 条 eval JSONL：

- `my-search-r1/eval_results/reward_train_compare_2026-07-21/penalty_v2_20step_dev.jsonl`
- `my-search-r1/eval_results/reward_train_compare_2026-07-21/penalty_v2_maxsearch001_5step_dev.jsonl`
- `my-search-r1/eval_results/reward_train_compare_2026-07-21/penalty_v2_maxsearch0005_5step_dev.jsonl`
- `my-search-r1/eval_results/reward_train_compare_2026-07-21/penalty_v3_followup_aware_5step_dev.jsonl`

每份输入运行：

```bash
PYTHONPATH=my-search-r1 uv run python my-search-r1/scripts/analyse_reward_sensitivity.py \
  --input <eval-jsonl> \
  --summary-output <summary-json> \
  --jsonl-output <rescore-jsonl> \
  --report-output <report-md> \
  --title '<run title>'
```

## 离线结果

| Run | mean_delta | helpful_bonus | boosted | correct_boosted | wrong_valid_boosted | penalized | wrong_valid_penalized | missing_followup_boosted | missing_followup_penalized |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `v2_20` | 0.0001 | 14 | 14 | 0 | 14 | 14 | 5 | 0 | 0 |
| `max001_5step` | -0.0003 | 7 | 7 | 0 | 7 | 5 | 4 | 0 | 2 |
| `max0005_5step` | -0.0010 | 5 | 5 | 0 | 5 | 8 | 4 | 0 | 1 |
| `v3_5step` | 0.0006 | 7 | 7 | 0 | 7 | 5 | 2 | 0 | 1 |

观测结论：

- 4 份 JSONL 中 `correct_boosted=0`，v4 没有给正确样本额外加分。
- bonus 只作用在 wrong-valid 且 helpful follow-up 的样本上，符合设计目标。
- v4 的 mean delta 很小，范围约为 `-0.0010` 到 `0.0006`，属于保守 shaping。
- `missing_followup_boosted=0`，说明当前 bonus 不能直接修复已经被标为 missing follow-up 的单搜索过早回答样本；它更像是在训练中鼓励已经出现的有用 follow-up 行为。
- max-search 两个 5-step checkpoint 上仍有少量 `missing_followup_penalized`，后续训练后需要重点看是否压掉必要二跳。

## 决策

通过离线预检查，进入 5-step 小预算训练门控。

训练门槛保持保守：

- 5-step dev eval 无知乎 API 错误，`tool_failures=0`，Zhihu success rate 为 `1.0`。
- `EM >= 0.3000`。
- `format >= 0.82`。
- `missing_followup_query <= 6`。
- `answer_granularity_miss` 不高于 v3 5-step。

若 5-step 未达标，不扩大到 20-step；若出现知乎 API 错误，停止实验并只整理已有结果。

## 本地验证

- `PYTHONPATH=my-search-r1 uv run python -m unittest discover -s my-search-r1/tests -v`：59 个 unittest 通过。
- `PYTHONPATH=my-search-r1 uv run python -m compileall -q my-search-r1/search_r1_minilab my-search-r1/scripts`：通过。
- `git diff --check`：通过。
