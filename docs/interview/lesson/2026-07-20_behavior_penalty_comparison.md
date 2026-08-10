# 行为指标与轻量 Penalty 对照复盘

复盘时间：2026-07-20 21:21 CST

## 实验目标

基于迁移测试失败案例复盘，完成三件事：

- 在 trajectory 报告和训练/eval metrics 中补充 Search-R1 行为指标。
- 接入默认关闭的轻量 reward penalty 配置。
- 尝试跑 Zhihu dev 前后对照，并在远程采样不可用时用已有 Zhihu dev trajectory 做离线重评分 sanity check。

## 代码改造

新增和改动能力：

- 新增统一行为诊断：`direct_correct`、`searched_correct`、`searched_wrong`、`direct_wrong`、`searched_invalid_format`、`empty_observation_count`、`duplicate_query_count`、`max_search_no_answer`、`too_many_search_no_gain`、`tool_observation_count`、`pending_tool_call_count`。
- Markdown 报告新增 Summary 指标、rate、bucket 和案例字段：gold answers、extracted answer、stop reason、empty observations、duplicate queries、pending tool calls。
- 训练/eval metrics 新增：
  - `behavior/direct_correct_rate`
  - `behavior/searched_correct_rate`
  - `behavior/searched_wrong_rate`
  - `behavior/empty_observation_rate`
  - `behavior/duplicate_query_rate`
  - `behavior/max_search_no_answer_rate`
  - `behavior/too_many_search_no_gain_rate`
- `eval_pytrio.py` 和 `train_pytrio.py` 新增默认关闭的 CLI 参数：
  - `--duplicate-query-penalty`
  - `--empty-result-penalty`
  - `--max-search-no-answer-penalty`
  - `--verbose-answer-penalty`
  - `--verbose-answer-token-threshold`
- 每条训练级 trajectory record 的 `metadata.reward_components` 写入 base reward、各项 penalty 和 final reward。

## 验证结果

已运行：

```bash
PYTHONPATH=my-search-r1 uv run python -m unittest discover -s my-search-r1/tests -v
PYTHONPATH=my-search-r1 uv run python -m compileall -q my-search-r1/search_r1_minilab my-search-r1/scripts
git diff --check
PYTHONPATH=my-search-r1 uv run python my-search-r1/scripts/analyse_trajectories.py --input my-search-r1/eval_results/zhihu_dev.jsonl --output my-search-r1/eval_results/zhihu_dev_behavior_baseline_existing.md --title 'PyTRIO Eval Report: zhihu_search baseline existing trajectories'
```

结果：

- 39 个 unittest 全部通过。
- `compileall` 通过。
- `git diff --check` 通过。
- 旧 Zhihu dev JSONL 可用新版报告逻辑离线复算。

## Zhihu Dev 对照

在线重新采样尝试：

```bash
timeout 900s env PYTHONPATH=my-search-r1 uv run python my-search-r1/scripts/eval_pytrio.py --data my-search-r1/datasets/dev.jsonl --backend zhihu_search --batch-size 1 --jsonl-output my-search-r1/eval_results/zhihu_dev_behavior_baseline.jsonl --report-output my-search-r1/eval_results/zhihu_dev_behavior_baseline.md
```

实际结果：

- 命令启动后停在 `Eval: 0/70`。
- 等待 3 分 49 秒仍未返回第一条 PyTRIO sampling 结果，手动中止。
- 中止栈显示阻塞在 `pytrio/client/sampling.py` 的 `sample_async` 等待响应阶段。
- 因此本轮没有得到新的在线 Zhihu dev baseline/penalty 采样对照结果。

补充验证：

- 还尝试过两个 `local_bm25 --limit 2` 临时 eval，分别为默认 reward 和 penalty reward，均在 `Eval: 0/2` 等待 3 分钟以上未返回第一条采样结果后中止。
- 这说明当前阻塞更可能在 PyTRIO sampling 服务响应，而不是 Zhihu 搜索 backend。

## 离线重评分 Sanity Check

由于 eval 阶段的 reward shaping 不改变模型生成内容，本轮使用已有 `my-search-r1/eval_results/zhihu_dev.jsonl` 做离线重评分，用于验证 penalty 对 reward/report 的影响。

输出：

- `my-search-r1/eval_results/zhihu_dev_behavior_baseline_existing.md`
- `my-search-r1/eval_results/zhihu_dev_behavior_penalty_offline.jsonl`
- `my-search-r1/eval_results/zhihu_dev_behavior_penalty_offline.md`

Penalty 参数：

- `duplicate_query_penalty=0.05`
- `empty_result_penalty=0.03`
- `max_search_no_answer_penalty=0.05`
- `verbose_answer_penalty=0.02`
- `verbose_answer_token_threshold=8`

注意：离线重评分没有 PyTRIO tokenizer，`verbose_answer` 的 token count 使用 whitespace word count；真实 train/eval 代码使用 tokenizer token count。

| 指标 | baseline existing | penalty offline |
| --- | ---: | ---: |
| `reward/mean` | 0.1814 | 0.1673 |
| `reward/correct` | 0.2143 | 0.2143 |
| `reward/format` | 0.6714 | 0.6714 |
| `rollout/search_calls` | 1.8286 | 1.8286 |
| `behavior/direct_correct_rate` | 0.0143 | 0.0143 |
| `behavior/searched_correct_rate` | 0.2000 | 0.2000 |
| `behavior/searched_wrong_rate` | 0.4571 | 0.4571 |
| `behavior/empty_observation_rate` | 0.0547 | 0.0547 |
| `behavior/duplicate_query_rate` | 0.0286 | 0.0286 |
| `behavior/max_search_no_answer_rate` | 0.1714 | 0.1714 |
| `behavior/too_many_search_no_gain_rate` | 0.2143 | 0.2143 |

Reward components 分布：

- 15 条正确样本保持 `final_reward=1.0`。
- 22 条错误但未触发 penalty 的样本保持 `final_reward=0.0`。
- 11 条 invalid format 未触发额外 penalty，保持 `final_reward=-0.1`。
- 12 条 `max_search_no_answer` 触发 `0.05` penalty，其中 2 条同时有空结果，2 条同时有重复 query。
- 10 条 verbose wrong answer 触发 `0.02` penalty，其中 1 条同时有空结果。

## 结论

- 行为指标和报告 bucket 已完成，能从旧 JSONL 复算，也能进入新 eval/train metrics。
- 轻量 penalty 配置已完成，默认关闭；显式启用后会写入 `reward_components`，并在最终 reward 中体现。
- 本轮没有完成新的在线 Zhihu dev 前后采样对照；原因是 PyTRIO sampling 在第一条样本处长时间无响应。
- 后续重新跑在线对照时，应优先确认 PyTRIO sampling 服务可用，再跑全量 dev。
