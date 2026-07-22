# Project TODO

本文件记录进行中任务、下一步、验收条件、阻塞项和未解决风险。完成且验证后移入 `PROJECT_COMPLETED.md`。

## 近期任务

- 设计下一轮 reward shaping 训练策略。
  - 验收条件：基于 2026-07-21 的 5-step 与 20-step checkpoint 对比结果、offline diagnostics、reward sensitivity 和 `docs/design/reward_shaping_plan.md`，完成下一轮保守 penalty 小预算训练，并明确是否继续做 no-empty ablation、answer-granularity/follow-up reward 或多 seed。
  - 当前状态：2026-07-21 至 2026-07-22 已完成 reward 设计记录表、base/5-step/penalty/20-step Zhihu dev 70 对比、5-step/20-step gained-lost case review、三类 offline diagnostics、reward sensitivity/rescore，`penalty_v2_candidate` 的 5-step/20-step SwanLab 在线训练，`penalty_v2_plus_max_search_001`、`penalty_v2_plus_max_search_0005` 的 5-step 小实验，三个关键 checkpoint 的 gained/lost case review，`penalty_v3_followup_aware` 的实现、离线 sensitivity、5-step 在线训练和 dev 评测，`reward_v4_followup_bonus` 的实现、离线 sensitivity、5-step 在线训练和 dev 评测，以及 v3 `group_size=8` 稳定性对照。20-step penalty v2 的 EM 0.3571 为当前最高，但 format 和搜索效率退化；v3 group size 4 仍是均衡候选，EM 0.3000、format 0.8571、平均搜索 1.5143；v4 和 v3 group size 8 都未达到扩大训练门槛。
  - 下一步：不要扩 v4 或 v3 group size 8 到 20/50-step。若继续做方法改进，优先把 follow-up 信号前移到 prompt/rollout 约束，要求多跳题先锁定中间实体再回答，并增加 final-answer 格式/唯一性约束；在线训练前先做 prompt smoke 与离线 case review。若知乎 API 出现 429、timeout、credential/http error、`tool_failures > 0` 或 success rate 低于 1.0，停止实验并总结。

## 未解决风险

- `my-search-r1/` 当前已有搜索工具层、trajectory JSONL、报告能力、训练级 rollout、PyTRIO train/eval CLI、完整数据集、一次非退化 1-step GRPO 更新证据、一次 5-step reward shaping checkpoint 对照、一次原始 reward 20-step checkpoint 对照、offline diagnostics、reward sensitivity，`penalty_v2_candidate` 的 5-step/20-step SwanLab 在线训练，v2+max-search 0.01/0.005 小实验，关键 checkpoint gained/lost case review，v3 follow-up-aware penalty 实现和小实验，v4 正向 follow-up bonus 实现/离线 sensitivity/小实验，以及 v3 group size 8 对照；尚未做 no-empty ablation、多 seed 稳定性重复实验或 prompt/rollout 层 follow-up 约束。
- 真实知乎搜索 API、PyTRIO 远程训练和 SwanLab 记录依赖外部凭据与服务状态，后续实验需要 mock baseline 与真实 backend 指标分开记录。
- 当前 dev 失败复盘显示 local BM25 只适合 smoke/mock；完整 dev 上空结果率 56.92%，不宜用它代表真实搜索能力。Zhihu dev 主要失败不在工具异常，而在格式收束、query 改写、证据阅读和严格 EM 对冗长答案/日期格式的误伤。2026-07-21 的 5-step 对比进一步显示，轻量 penalty 能减少过度搜索但可能损伤答对率，需要更谨慎的权重或训练日程。
- 当前已有 `03-search-r1/train.py` 本地改动，后续修改基线文件前需要继续保护这部分改动。
