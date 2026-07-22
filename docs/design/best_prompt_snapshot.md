# Best Prompt Snapshot

记录时间：2026-07-22 13:10 CST

本文记录当前 Search-R1 MiniLab 的最优 prompt base，供后续训练、ablation 和面试复盘引用。详细实验复盘见 `docs/interview/2026-07-22_prompt_search_budget_guard_eval.md`。

## Prompt Version

名称：`prompt_search_budget_guard`

核心改动：

- 最终回答前必须先看到至少一次 search result。
- 多跳/关系题先识别 bridge entity，再搜索该实体或关系。
- 尽量 3 次搜索内完成；3 次后用最佳证据输出短答案，而不是继续请求第 5 次搜索。
- 最终答案必须是单行 `Answer: <shortest single answer span>`。

当前 system prompt：

```text
You answer factual questions with help from a search tool.
Search before giving the final answer. Use concise English queries.
Do not answer from memory before seeing at least one search result.
For multi-hop or relation questions, first identify the bridge entity, then search that entity or relation before answering.
Call search exactly once per assistant turn. Wait for the tool result before making another search call.
Do not stop after a search result that only identifies an intermediate person, work, place, date, role, or organization.
Use at most three searches when possible. After three searches, answer with the best supported short span instead of asking for another search.
When ready, output exactly one line and nothing else:
Answer: <shortest single answer span>
Do not include reasoning, markdown, citations, parentheses, alternatives, or words such as "or" after Answer:.
Do not call a tool and give the final answer in the same turn.
```

当前 tool observation reminder：

```text
Reminder: if the result only identifies a bridge entity, search that entity or relation before answering. Final output must be exactly one line: Answer: <shortest single answer span>. If you have searched three times, answer with the best supported span instead of searching again.
```

## Data And Eval Config

数据：

- Train data：`my-search-r1/datasets/train.jsonl`
- Dev data：`my-search-r1/datasets/dev.jsonl`
- Dev size：70
- Dev composition：7 个 data source，每类 10 条

评测配置：

- Base model：`Qwen/Qwen3.5-4B`
- Backend：`zhihu_search`
- Seed：42
- Temperature：0.0
- Top-p：1.0
- Batch size：1
- Max search calls：4
- Max assistant turns：6

主要产物：

- Dev-5 JSONL：`my-search-r1/eval_results/reward_train_compare_2026-07-21/prompt_search_budget_guard_dev5_20260722.jsonl`
- Dev-5 report：`my-search-r1/eval_results/reward_train_compare_2026-07-21/prompt_search_budget_guard_dev5_20260722.md`
- Dev 70 JSONL：`my-search-r1/eval_results/reward_train_compare_2026-07-21/prompt_search_budget_guard_dev_20260722.jsonl`
- Dev 70 report：`my-search-r1/eval_results/reward_train_compare_2026-07-21/prompt_search_budget_guard_dev_20260722.md`
- Offline diagnostics：`my-search-r1/eval_results/reward_train_compare_2026-07-21/prompt_search_budget_guard_dev_20260722_offline_diagnostics.md`

## Current Best Metrics

| Metric | Value |
| --- | ---: |
| `em/macro` | 0.4143 |
| `format/rate` | 0.8857 |
| `rollout/search_calls` | 1.9429 |
| `rollout/no_search_rate` | 0.0000 |
| `missing_followup_query` | 0 |
| `answer_granularity_miss` | 0 |
| `multi_candidate_answer` | 0 |
| `bad_max_search_loop` | 4 |
| Zhihu success rate | 1.0000 |
| Tool failures | 0 |

相对 `prompt_search_first`：

- EM：0.3714 -> 0.4143
- Format：0.8000 -> 0.8857
- Average search：2.0571 -> 1.9429
- Gained/lost：4 / 1
- Invalid format：14 -> 8

## Training Gate

后续训练必须以本 prompt 为 base，并用以下门槛判断是否继续扩大：

- EM 不低于 0.4143。
- Format 不低于 0.8857。
- `missing_followup_query=0`。
- `answer_granularity_miss=0`。
- no-search 不反弹。
- 平均搜索不明显高于 1.9429。
- 长训必须开启 `--save-every 10` 或更频繁。

如果训练无法超过该 prompt-only base，应优先保留 prompt-only 方案，并把训练结果作为负向或稳定性参考。
