# Project TODO

本文件只记录当前活跃任务、验收条件、停止条件和未解决风险。切换前快照位于 `docs/status/archive/2026-10-08_pre_skill_opsd/`，仅供历史追溯。

## Active Track: Skill 条件 OPSD 实现与审计

目标：保留旧 final route 作为已验证 baseline，完成 Skill teacher 的固定 checkpoint 审计，再判断是否启动新方法训练。代码与本地测试结果见 `docs/interview/lesson/2026-10-08_skill_opsd_code_integration.md`。

### 1. 固定 checkpoint 离线审计

- 用训练 split 既有轨迹核查 Skill Bank 不含答案、测试题或未来 observation；对同一已采样动作比较普通上下文与 Skill 上下文 logprob。
- 按桥接搜索、最终属性搜索、过早回答及格式 token 统计 signed gap、gate、teacher 调用量和耗时，并人工复核 gained/lost case。
- 停止条件：真实 Qwen chat template 下 token 对齐失败、Skill 主要抬高错误动作、gap 近零且无有用区分，或额外 teacher 成本不可接受。未通过不启动新训练。

### 2. 训练与评测门槛

- 审计通过后，从同一 guard-fix checkpoint、同 seed/数据/backend/预算对照 GRPO、显式旧 same-context 门控辅助和 Skill OPSD；先做 local BM25 小步 smoke，再进入真实 backend。
- 每组记录 loss/reward、EM/correct、format、avg search、tool success、missing follow-up、重复 query、bad max-search loop、Skill/gap/gate 分布、token/时间/费用。
- dev70 只做 health gate；至少一个 clean targeted eval 后再谈收益。工具成功率不足、格式下降、搜索次数失控、gate 饱和或对齐失败时停止扩大训练。新方法指标不能沿用旧门控结果。

### 3. 旧路线验证补强

- `guardfix20-resume-opsd-v2-5step-20260811` 仍缺 alias80、第二 seed/非重叠采样和 bridge350 gained/lost case review；有资源时补 MuSiQue/bridge500 边界分析。
- 小预算训练覆盖不足，暂不盲目增加到 20+ step；扩量前先判断问题来自训练覆盖、reward 还是 teacher 设计。

## Parked Tracks

- 不可靠搜索工具鲁棒训练：已搁置；failure injection 只保留为 smoke、回归测试和评测可信度工具。
- Penalty-only reward 路线：已停放；简单 duplicate/empty/max-search penalty 可能压掉必要 follow-up。
- Base 起点 same-context OPSD：收益不足；除非引入本质不同 teacher 或 preference-filtered replay，否则不继续扩。
- Evidence-v2 50-step、guard-fix 独立 full bridge150、KL/std 单因素消融：可作补充验证，但不是当前下一步。

## 未解决风险

- 真实 Zhihu Search API、PyTRIO 远程训练和 SwanLab 依赖外部服务；实验前必须 health check，并分开记录工具失败和模型策略失败。
- 小预算 RL 方差较大；当前训练覆盖远小于 `train.jsonl`，不能包装成充分训练。
- `bridge_eval_350` 是 test-only targeted subset，但不是完整无偏 test；去除 MuSiQue 的边界必须继续说明。
- alias/granularity 对最终路线尚未验证；如果面试或展示中提泛化收益，需要先补证据。
- `docs/interview/future/baoyan-resume-template/main.tex` 的 Skill 条件 OPSD 仅完成代码接入；结果数字仍是原门控辅助训练的占位数据。对外使用前须在新方法训练评测后替换，或继续明确旧路线归因。
- 当前工作区已有多处代码改动；提交前必须复查 `git diff --cached`、敏感信息和大文件边界。
