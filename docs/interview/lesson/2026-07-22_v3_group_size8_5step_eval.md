# Penalty v3 Group Size 8 5-step Eval

## 目标

在 `reward_v4_followup_bonus` 未通过扩大训练门槛后，验证一个不改 reward 的稳定性对照：保持 `penalty_v3_followup_aware` 参数不变，只把 `group_size` 从 4 提高到 8，观察更大 rollout group 是否改善 5-step 小样本方差。

## 配置

- 数据：`my-search-r1/datasets/train.jsonl` 训练，`my-search-r1/datasets/dev.jsonl` 评测。
- 模型：默认 `Qwen/Qwen3.5-4B`，LoRA rank 默认值。
- 训练预算：`max_steps=5`，`questions_per_batch=2`，`group_size=8`。
- 后端：`zhihu_search`。
- seed：`42`。
- reward 参数：
  - `duplicate_query_penalty=0.03`
  - `empty_result_penalty=0.01`
  - `bad_max_search_penalty=0.01`
  - `date_granularity_penalty=0.05`
  - `multi_candidate_answer_penalty=0.02`
  - `helpful_followup_bonus=0.0`

真实 PyTRIO checkpoint URI、SwanLab 链接和本地运行日志路径不写入公开文档。

## 训练结果

- 生成 80 条训练 trajectory。
- 5/5 step 都执行 optimizer 阶段。
- 训练阶段 Zhihu requests `205`，success rate `1.0`。
- 训练 trajectory tool failures 为 `0`。
- 训练平均 reward `0.0954`，correct rate `0.1500`，format rate `0.4750`，平均搜索 `2.5625`。

后两步训练明显退化：

| Step | reward mean | correct rate | mean search calls |
| --- | ---: | ---: | ---: |
| 1 | 0.3250 | 0.3750 | 2.5000 |
| 2 | 0.1188 | 0.1250 | 1.4375 |
| 3 | 0.2313 | 0.2500 | 2.3750 |
| 4 | -0.1100 | 0.0000 | 3.5000 |
| 5 | -0.0881 | 0.0000 | 3.0000 |

## Dev 70 结果

| Metric | v3 group 4 | v3 group 8 |
| --- | ---: | ---: |
| `em/macro` | 0.3000 | 0.2714 |
| `format/rate` | 0.8571 | 0.6857 |
| `rollout/search_calls` | 1.5143 | 2.1143 |
| `behavior/helpful_followup_query_rate` | 0.2143 | 0.4429 |
| `behavior/max_search_no_answer_rate` | 0.1143 | 0.2429 |
| `behavior/too_many_search_no_gain_rate` | 0.1286 | 0.3143 |
| `behavior/bad_max_search_loop_rate` | 0.0286 | 0.0429 |
| `tool/zhihu_search/success_rate` | 1.0 | 1.0 |

Offline diagnostics：

| Metric | Value |
| --- | ---: |
| `wrong_valid` | 29 |
| `possible_alias_match` | 9 |
| `answer_granularity_miss` | 0 |
| `missing_followup_query` | 3 |
| `helpful_followup_query` | 31 |
| `bad_max_search_loop` | 3 |
| `multi_candidate_answer` | 3 |

## Gained/Lost Review

相对 v3 group 4：

- gained 3：`dev_7774`、`test_29`、`dev_3412`
- lost 5：`test_108`、`dev_2407`、`dev_4889`、`dev_2223`、`test_494`

相对 max001：

- gained 4：`dev_7774`、`test_29`、`test_97`、`test_4020`
- lost 6：`test_108`、`dev_2407`、`dev_4889`、`dev_2223`、`test_2231`、`test_8542`

## 结论

`group_size=8` 提高了 helpful follow-up 行为率，并降低了 `missing_followup_query` 到 3，但代价是 format 和搜索效率明显退化，EM 也低于 v3 group 4 和 max001。

本轮不继续扩大 `group_size=8` 训练。当前最值得保留的结论是：单纯增加 rollout group 不能解决 reward 对 format 和最终答案唯一性的约束不足；下一步应优先改 prompt/rollout 结构或做 final-answer 约束，而不是继续加 step 或增大 group。
