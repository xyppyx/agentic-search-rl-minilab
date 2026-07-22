# Project TODO

本文件记录进行中任务、下一步、验收条件、阻塞项和未解决风险。完成且验证后移入 `PROJECT_COMPLETED.md`。

## 近期任务

- 设计下一轮 reward shaping 训练策略。
  - 验收条件：基于 2026-07-21 至 2026-07-22 的 checkpoint 对比结果、offline diagnostics、reward sensitivity 和 `docs/design/reward_shaping_plan.md`，明确下一轮是否做 no-empty/lower-penalty ablation、必要搜索/follow-up 正向门控或多 seed 稳定性重复实验。
  - 当前状态：2026-07-21 至 2026-07-22 已完成 reward 设计记录表、base/5-step/penalty/20-step Zhihu dev 70 对比、5-step/20-step gained-lost case review、三类 offline diagnostics、reward sensitivity/rescore，`penalty_v2_candidate` 的 5-step/20-step SwanLab 在线训练，`penalty_v2_plus_max_search_001`、`penalty_v2_plus_max_search_0005` 的 5-step 小实验，三个关键 checkpoint 的 gained/lost case review，`penalty_v3_followup_aware` 的实现、离线 sensitivity、5-step 在线训练和 dev 评测，`reward_v4_followup_bonus` 的实现、离线 sensitivity、5-step 在线训练和 dev 评测，v3 `group_size=8` 稳定性对照，prompt/rollout 层多跳 follow-up 与短答案格式约束的本地实现，prompt 约束版 Zhihu dev-5/dev 70 效果对照，prompt 约束基础上的 v3 follow-up-aware/bad-loop 5-step 小预算训练，search-first prompt dev 70 对照，`reward_v5_no_search_guard` 实现、离线 sensitivity、一次 50-step 中断复盘和一次 20-step 训练/eval，`prompt_search_budget_guard` 的 dev-5/dev 70 对照，以及 GRPO KL/std 训练稳定性改造和 5-step Zhihu 小实验。当前最高 checkpoint 证据是 `prompt_budget_kl_std_5step` 的 EM 0.4286、format 0.9429、平均搜索 1.9429、`missing_followup_query=0`、`answer_granularity_miss=0`；当前最高 prompt-only base 是 `prompt_search_budget_guard` 的 EM 0.4143、format 0.8857。2026-07-22 已开始单因素 ablation：`std+clip only` 完成 5-step 训练和 dev-5，dev-5 EM 0.4000、format 0.8000、平均搜索 3.0000；`std+clip only` full dev、`KL only` train 和 base dev-1 smoke 均卡在 PyTRIO sampling await，已中断并记录到 `docs/interview/2026-07-22_grpo_kl_std_ablation_partial.md`。
  - 下一步：不要扩 v4、v3 group size 8、prompt+v3 5-step 或 prompt+v5 20-step 到 50-step。保留 `prompt_search_budget_guard` 作为最强公开 prompt base，并保留 `prompt_budget_kl_std_5step` 作为当前最强 checkpoint 证据；PyTRIO sampling 恢复后，优先补 `std+clip only` full dev，再跑 `KL only` 5-step train/dev-5/full dev，确认收益来源。任何 20-step 扩大前必须以 EM 不低于 0.4286、format 不低于 0.9429、`missing_followup_query=0`、`answer_granularity_miss=0` 作为门槛。所有后续长训必须开启 `--save-every 5` 或更频繁。若知乎 API 出现 429、timeout、credential/http error、`tool_failures > 0` 或 success rate 低于 1.0，停止实验并总结；若 PyTRIO sampling await 超过 2 分钟无进度，停止当前 run 并记录为外部 sampling 阻塞。

## 未解决风险

- `my-search-r1/` 当前已有搜索工具层、trajectory JSONL、报告能力、训练级 rollout、PyTRIO train/eval CLI、完整数据集、一次非退化 1-step GRPO 更新证据、一次 5-step reward shaping checkpoint 对照、一次原始 reward 20-step checkpoint 对照、offline diagnostics、reward sensitivity，`penalty_v2_candidate` 的 5-step/20-step SwanLab 在线训练，v2+max-search 0.01/0.005 小实验，关键 checkpoint gained/lost case review，v3 follow-up-aware penalty 实现和小实验，v4 正向 follow-up bonus 实现/离线 sensitivity/小实验，v3 group size 8 对照，prompt/rollout 层 follow-up 约束的本地实现和 Zhihu dev 对照，prompt 约束基础上的小预算训练，v5 no-search guard 小实验，`prompt_search_budget_guard` prompt-only 最强 base，以及 KL/std 稳定化 GRPO 5-step 正向结果；KL/std 单因素 ablation 进行中但被 PyTRIO sampling 阻塞，尚未完成 full dev 归因、多 seed 稳定性重复实验或 20-step 扩展。
- 真实知乎搜索 API、PyTRIO 远程训练和 SwanLab 记录依赖外部凭据与服务状态，后续实验需要 mock baseline 与真实 backend 指标分开记录。
- 当前 dev 失败复盘显示 local BM25 只适合 smoke/mock；完整 dev 上空结果率 56.92%，不宜用它代表真实搜索能力。Zhihu dev 主要失败不在工具异常，而在格式收束、query 改写、证据阅读和严格 EM 对冗长答案/日期格式的误伤。2026-07-21 的 5-step 对比进一步显示，轻量 penalty 能减少过度搜索但可能损伤答对率，需要更谨慎的权重或训练日程。
- 当前已有 `03-search-r1/train.py` 本地改动，后续修改基线文件前需要继续保护这部分改动。
