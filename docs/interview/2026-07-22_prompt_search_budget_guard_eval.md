# Prompt Search Budget Guard 复盘

复盘时间：2026-07-22 12:35 CST

## 实验目标

在 `prompt_search_first` 已取得当前最高 EM，但 format 降到 0.8000、max-search 类 format 失败仍有 14 条的基础上，继续做低风险 prompt 优化：

1. 保留“最终回答前必须先看到一次 search result”，防止 no-search 退化。
2. 增加搜索预算提醒：尽量 3 次搜索内完成，3 次后用最好证据输出短答案，而不是继续发起第 5 次搜索。
3. 目标是在不牺牲 `missing_followup_query=0` 的前提下，把 format 拉回 0.85+，并让平均搜索次数回到 v3 group size 4 附近。

本轮没有继续启动 50-step 训练。原因是 `prompt_search_first` + v5 20-step 已显示训练会把 EM/format 拉低并诱发过度搜索；当前更优先验证 prompt-only 约束能否直接修复 format/search。

## 实现变更

代码变更：

- `my-search-r1/search_r1_minilab/protocol.py`
  - system prompt 增加：尽量最多搜索 3 次；3 次后用最佳证据短答案作答，不继续搜索。
  - tool observation reminder 同步增加 3-search 后作答提醒。
- `my-search-r1/tests/test_protocol.py`
  - 增加 prompt/reminder 中搜索预算约束的回归断言。

验证：

```bash
PYTHONPATH=my-search-r1 uv run python -m unittest my-search-r1/tests/test_protocol.py -v
PYTHONPATH=my-search-r1 uv run python -m compileall -q my-search-r1/search_r1_minilab my-search-r1/scripts
PYTHONPATH=my-search-r1 uv run python -m unittest discover -s my-search-r1/tests -v
```

结果：协议单测 3 个通过；完整 unittest 65 个通过；compileall 通过。

## Zhihu Dev-5 快筛

命令：

```bash
PYTHONPATH=my-search-r1 uv run python my-search-r1/scripts/eval_pytrio.py \
  --data my-search-r1/datasets/dev.jsonl \
  --backend zhihu_search \
  --env-file my-search-r1/.env \
  --limit 5 \
  --batch-size 1 \
  --seed 42 \
  --temperature 0.0 \
  --top-p 1.0 \
  --jsonl-output my-search-r1/eval_results/reward_train_compare_2026-07-21/prompt_search_budget_guard_dev5_20260722.jsonl \
  --report-output my-search-r1/eval_results/reward_train_compare_2026-07-21/prompt_search_budget_guard_dev5_20260722.md
```

结果：

- EM：0.4000
- format：0.6000
- 平均搜索：2.8000
- no-search rate：0.0000
- `bad_max_search_loop_rate=0.0000`
- Zhihu requests：14
- Zhihu success rate：1.0000
- tool failures：0

对比 `prompt_search_first` dev-5：EM 从 0.2000 升到 0.4000，format 从 0.4000 升到 0.6000，平均搜索从 3.2000 降到 2.8000，bad max-search loop 从 0.2000 降到 0。

## Zhihu Dev 70

命令：

```bash
PYTHONPATH=my-search-r1 uv run python my-search-r1/scripts/eval_pytrio.py \
  --data my-search-r1/datasets/dev.jsonl \
  --backend zhihu_search \
  --env-file my-search-r1/.env \
  --limit 0 \
  --batch-size 1 \
  --seed 42 \
  --temperature 0.0 \
  --top-p 1.0 \
  --jsonl-output my-search-r1/eval_results/reward_train_compare_2026-07-21/prompt_search_budget_guard_dev_20260722.jsonl \
  --report-output my-search-r1/eval_results/reward_train_compare_2026-07-21/prompt_search_budget_guard_dev_20260722.md
```

结果：

| Run | EM | Format | Avg search | no-search | missing_followup | helpful_followup | multi_candidate | granularity_miss | bad_loop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| prompt constraints base | 0.3429 | 0.8714 | 1.7143 | 0.0000 | 3 | 21 | 0 | 0 | 5 |
| prompt search-first base | 0.3714 | 0.8000 | 2.0571 | 0.0000 | 0 | 33 | 0 | 0 | 3 |
| prompt search-budget guard | 0.4143 | 0.8857 | 1.9429 | 0.0000 | 0 | 32 | 0 | 0 | 4 |
| v3 group size 4 | 0.3000 | 0.8571 | 1.5143 | 0.0429 | 6 | 15 | 1 | 0 | 2 |
| prompt+v5 20-step final | 0.3000 | 0.7286 | 2.5857 | 0.0000 | 0 | 51 | 0 | 5 | 4 |

Zhihu backend：

- requests：136
- success rate：1.0000
- empty rate：0.0515
- error rate：0.0000
- timeout rate：0.0000
- rate-limit rate：0.0000
- tool failures：0

Offline diagnostics：

```bash
PYTHONPATH=my-search-r1 uv run python my-search-r1/scripts/analyse_offline_diagnostics.py \
  --input my-search-r1/eval_results/reward_train_compare_2026-07-21/prompt_search_budget_guard_dev_20260722.jsonl \
  --jsonl-output my-search-r1/eval_results/reward_train_compare_2026-07-21/prompt_search_budget_guard_dev_20260722_offline_diagnostics.jsonl \
  --report-output my-search-r1/eval_results/reward_train_compare_2026-07-21/prompt_search_budget_guard_dev_20260722_offline_diagnostics.md \
  --title 'Prompt Search Budget Guard Offline Diagnostics'
```

结果：

- `missing_followup_query=0`
- `answer_granularity_miss=0`
- `multi_candidate_answer=0`
- `possible_alias_match=7`
- wrong-valid：33

## Gained/Lost

相对 `prompt_search_first` base：

- gained：4
- lost：1
- invalid format：14 降到 8
- max-search invalid：14 降到 8

gained 样本：

- `dev_6133`：Sophie Of France maternal grandfather，搜索次数 4 降到 2 后答对。
- `dev_7773`：导演出生更晚的电影，仍用 4 次搜索但最终收束为正确答案。
- `test_2231`：Astros league switch，搜索次数 4 降到 1 后答对。
- `test_8542`：Slivovitz fruit，保持 1 次搜索并答对。

lost 样本：

- `test_7511`：Spring Waltz director，搜索次数同为 2，但最终答案退化。

## 结论

`prompt_search_budget_guard` 是当前最强 base：

- EM 0.4143，超过此前最高 `prompt_search_first` 的 0.3714。
- format 0.8857，超过 0.85 目标线，也超过 `prompt_search_first` 的 0.8000。
- 平均搜索 1.9429，低于 `prompt_search_first` 的 2.0571，并明显低于 v5 20-step 的 2.5857。
- `missing_followup_query=0`，没有重现 prompt+v3 的 no-search/过早回答退化。
- `answer_granularity_miss=0`、`multi_candidate_answer=0`，没有重现 v5 20-step 的日期粒度退化。

下一步决策：

- 将 `prompt_search_budget_guard` 作为新的当前最佳 prompt base。
- 暂不做 50-step reward 训练；已有证据显示当前训练配置比 prompt-only 更容易退化。
- 后续若继续训练，应围绕该 prompt 做 5-step smoke，且必须 `--save-every 10` 或更频繁；门槛建议为 EM 不低于 0.4143、format 不低于 0.8857、`missing_followup_query=0`。
- 下一轮更有价值的是多 seed 或同一 prompt 下温度/搜索预算 ablation，而不是继续扩大 v5。
