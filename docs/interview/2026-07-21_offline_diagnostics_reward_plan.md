# Offline Diagnostics 与 Reward Plan 复盘

复盘时间：2026-07-21 21:10 CST

## 目标

完成两件事：

- 建立 reward shaping 版本记录表，避免 penalty 设计散落在对话和实验文档中。
- 补充 3 类 offline diagnostic，并在已有 base、5-step、penalty、20-step eval JSONL 上跑通。

本次没有重新训练，也没有调用模型或搜索 API；只读取已有 eval JSONL 做离线分析。

## 新增产物

- `docs/design/reward_shaping_plan.md`
- `my-search-r1/search_r1_minilab/offline_diagnostics.py`
- `my-search-r1/scripts/analyse_offline_diagnostics.py`
- `my-search-r1/tests/test_offline_diagnostics.py`

索引已补到：

- `README.md`
- `docs/README.md`
- `my-search-r1/README.md`

## Diagnostic 定义

| Diagnostic | 作用 | 示例 |
| --- | --- | --- |
| `possible_alias_match` | 标出 strict EM false negative 或近似别名/拼写变体 | `Dexter King` vs `Dexter`，`Yun Seok-ho` vs `Yoon Seok-Ho` |
| `answer_granularity_miss` | 标出 final answer 粒度不足 | gold 为 `October 2, 1869`，final answer 为 `1869` |
| `missing_followup_query` | 标出单次搜索后在多跳/实体角色题上过早作答的风险 | 找到 mother/writer/director 后没有继续搜关键中间实体 |

这些标签是启发式 diagnostic，不替代人工 case review，也不直接作为最终评测指标。

## 运行命令

Base：

```bash
PYTHONPATH=my-search-r1 uv run python my-search-r1/scripts/analyse_offline_diagnostics.py --input my-search-r1/eval_results/reward_train_compare_2026-07-21/base_dev.jsonl --jsonl-output my-search-r1/eval_results/reward_train_compare_2026-07-21/base_offline_diagnostics.jsonl --report-output my-search-r1/eval_results/reward_train_compare_2026-07-21/base_offline_diagnostics.md --title 'Base Offline Diagnostics'
```

5-step 原始 reward：

```bash
PYTHONPATH=my-search-r1 uv run python my-search-r1/scripts/analyse_offline_diagnostics.py --input my-search-r1/eval_results/reward_train_compare_2026-07-21/baseline_reward_ckpt_dev.jsonl --jsonl-output my-search-r1/eval_results/reward_train_compare_2026-07-21/baseline_reward_ckpt_offline_diagnostics.jsonl --report-output my-search-r1/eval_results/reward_train_compare_2026-07-21/baseline_reward_ckpt_offline_diagnostics.md --title '5-Step Baseline Reward Offline Diagnostics'
```

5-step penalty：

```bash
PYTHONPATH=my-search-r1 uv run python my-search-r1/scripts/analyse_offline_diagnostics.py --input my-search-r1/eval_results/reward_train_compare_2026-07-21/penalty_reward_ckpt_dev.jsonl --jsonl-output my-search-r1/eval_results/reward_train_compare_2026-07-21/penalty_reward_ckpt_offline_diagnostics.jsonl --report-output my-search-r1/eval_results/reward_train_compare_2026-07-21/penalty_reward_ckpt_offline_diagnostics.md --title '5-Step Penalty Reward Offline Diagnostics'
```

20-step 原始 reward：

```bash
PYTHONPATH=my-search-r1 uv run python my-search-r1/scripts/analyse_offline_diagnostics.py --input my-search-r1/eval_results/reward_train_compare_2026-07-21/base_20step_dev.jsonl --jsonl-output my-search-r1/eval_results/reward_train_compare_2026-07-21/base_20step_offline_diagnostics.jsonl --report-output my-search-r1/eval_results/reward_train_compare_2026-07-21/base_20step_offline_diagnostics.md --title '20-Step Baseline Reward Offline Diagnostics'
```

## 结果

| Run | wrong_valid | possible_alias_match | answer_granularity_miss | missing_followup_query |
| --- | ---: | ---: | ---: | ---: |
| base | 26 | 10 | 0 | 0 |
| 5-step 原始 reward | 37 | 9 | 0 | 3 |
| 5-step penalty | 35 | 10 | 0 | 2 |
| 20-step 原始 reward | 41 | 8 | 1 | 6 |

20-step 重点样本：

- `answer_granularity_miss`：`test_97`，final answer 从完整日期退化成 `1869`。
- `missing_followup_query`：包含 `dev_4869` 和 `dev_3412`，对应 5-step/20-step gained-lost review 中的真实退化样本。
- `possible_alias_match`：包含 `dev_2407`，对应 `Dexter King` vs `Dexter` 的 strict EM false negative。

## 解释

- `possible_alias_match` 在所有 run 中都不少，说明 strict EM 对别名、全名/短名和日期格式有持续误伤；后续报告不能只看 EM。
- `answer_granularity_miss` 当前主要集中在 20-step 的 `test_97`，与人工 review 一致。
- `missing_followup_query` 在 20-step 中最高，说明 20-step 的搜索收敛确实可能带来多跳 follow-up 不足；这支持“不要继续粗暴压低搜索次数”的决策。

## 对下一轮 Reward 的影响

1. `penalty_v1` 不应直接扩大训练，因为它已经证明会降低搜索次数但损伤 EM。
2. 下一版 penalty 应只轻罚 duplicate query 和空结果，不启用粗粒度 `max_search_no_answer_penalty`。
3. 需要新增或至少先报告 answer granularity 诊断，避免日期题只答年份。
4. 如果要加 follow-up query reward，必须只鼓励不同实体/关系的二跳搜索，不能奖励重复搜索或无限搜索。

## 验证

- `PYTHONPATH=my-search-r1 uv run python -m unittest discover -s my-search-r1/tests -v`：44 个 unittest 通过。
- `PYTHONPATH=my-search-r1 uv run python -m compileall -q my-search-r1/search_r1_minilab my-search-r1/scripts`：通过。
- `git diff --check`：通过。
- 4 份已有 eval JSONL 的 offline diagnostics 均成功生成 JSONL 与 Markdown；输出位于被 git 忽略的 `my-search-r1/eval_results/`。
