# Reward v4 Follow-up Bonus 5-step Eval

## 目标

验证 `reward_v4_followup_bonus` 是否能在不牺牲 format 和搜索效率的情况下，鼓励必要 follow-up query。

本轮使用 5-step 小预算门控，不直接扩大到 20/50-step。

## 配置

- 数据：`my-search-r1/datasets/train.jsonl` 训练，`my-search-r1/datasets/dev.jsonl` 评测。
- 模型：默认 `Qwen/Qwen3.5-4B`，LoRA rank 默认值。
- 训练预算：`max_steps=5`，`questions_per_batch=2`，`group_size=4`。
- 后端：`zhihu_search`。
- seed：训练 `42`，最终 dev eval `42`。
- reward 参数：
  - `duplicate_query_penalty=0.03`
  - `empty_result_penalty=0.01`
  - `bad_max_search_penalty=0.01`
  - `date_granularity_penalty=0.05`
  - `multi_candidate_answer_penalty=0.02`
  - `helpful_followup_bonus=0.02`

真实 PyTRIO checkpoint URI、SwanLab 链接和本地运行日志路径不写入公开文档。

## 运行记录

先运行 dev-5 smoke，确认 Zhihu backend 可用：

- 5 条 dev trajectory。
- Zhihu requests `10`，success rate `1.0`。
- error、timeout、rate-limit 均为 `0.0`。

随后运行 5-step GRPO：

- 生成 40 条训练 trajectory。
- 5/5 step 都执行 optimizer 阶段。
- 训练阶段 Zhihu requests `94`，success rate `1.0`。
- 训练 trajectory tool failures 为 `0`。
- 训练平均 reward `0.1293`，correct rate `0.1750`，format rate `0.5500`，平均搜索 `2.3500`。

第一次完整 dev eval 在第 12/70 条附近遇到 PyTRIO sampling run `409 conflict`，不是知乎 API 错误；按停止条件未归类为 Zhihu API failure。重新创建 sampling client 后，用 seed 42 完成 dev 70 eval。

## Dev 70 结果

| Metric | Value |
| --- | ---: |
| `em/macro` | 0.2714 |
| `format/rate` | 0.7714 |
| `rollout/search_calls` | 1.6286 |
| `behavior/helpful_followup_query_rate` | 0.2714 |
| `behavior/max_search_no_answer_rate` | 0.1571 |
| `behavior/too_many_search_no_gain_rate` | 0.1857 |
| `behavior/bad_max_search_loop_rate` | 0.0286 |
| `tool/zhihu_search/requests` | 114 |
| `tool/zhihu_search/success_rate` | 1.0 |
| `tool/zhihu_search/error_rate` | 0.0 |
| `tool/zhihu_search/timeout_rate` | 0.0 |
| `tool/zhihu_search/rate_limit_rate` | 0.0 |

Offline diagnostics：

| Metric | Value |
| --- | ---: |
| `wrong_valid` | 35 |
| `possible_alias_match` | 9 |
| `answer_granularity_miss` | 0 |
| `missing_followup_query` | 4 |
| `helpful_followup_query` | 19 |
| `bad_max_search_loop` | 2 |
| `multi_candidate_answer` | 2 |

Reward sensitivity on this checkpoint：

- `reward_v4_followup_bonus` 的 `helpful_followup_bonus_count=8`。
- `correct_boosted=0`，`wrong_valid_boosted=8`。
- `mean_delta=0.0009`，仍然是很轻的 shaping。

## 门控判断

本轮不进入 20-step 或 50-step。

原因：

- EM `0.2714`，低于门槛 `0.3000`。
- format `0.7714`，低于门槛 `0.82`。
- 虽然 `missing_followup_query=4` 低于 v3 的 6，且 `answer_granularity_miss=0`，但整体 format 退化太明显。
- Zhihu API 本身没有失败，停止扩大训练是 reward/策略门控决策，不是 API stop。

## Gained/Lost Review

相对 `penalty_v3_followup_aware_5step`：

- gained 1：`dev_3412`
- lost 3：`dev_6748`、`test_494`、`test_4020`

相对 `penalty_v2_plus_max_search_001`：

- gained 1：`test_97`
- lost 3：`dev_6748`、`test_2231`、`test_8542`

关键案例：

- `dev_3412`：v4 用第二跳 `Lari White talent competition winner` 答对 `You Can Be a Star`，说明 follow-up bonus 的方向有局部收益。
- `test_97`：v4 通过 `Mahatma Gandhi birth date` 修复了 max001 的年份粒度问题，答出 `October 2, 1869`。
- `dev_6748`：v3/max001 都能比较两人出生日期，v4 直接 invalid format 且无搜索，是 format/遵循退化。
- `test_494`：v3 答对 `David Villa`，v4 单次搜索后答 `Alvaro Morata`，是证据读取或搜索结果鲁棒性退化。
- `test_4020`：v4 输出 `The Buddha (or Shakyamuni Buddha)`，严格 EM 判错，属于多候选/答案唯一性问题。

## 结论

`reward_v4_followup_bonus` 证明了正向 follow-up 信号可以修复少量多跳样本，但 5-step 结果没有超过 v3/max001，也没有达到扩大训练门槛。当前不应把 v4 推到 20/50-step。

下一步更合理的是把 follow-up 信号从单纯 reward bonus 前移到 prompt/rollout 约束，要求模型在多跳题中先锁定中间实体，再输出最终答案；或者做 `group_size=8` 稳定性对照，降低 5-step 小样本方差。
