# Project TODO

本文件只记录当前活跃任务、下一步验收条件、停止条件和未解决风险。压缩前快照位于 `docs/status/archive/2026-09-16_pre_next_experiments/`，仅供历史追溯。

## Active Track: Next-Experiment Prep

目标：在不改变当前最终路线的前提下，把 Git、文档、评测门槛和后续实验入口整理干净，便于继续做验证补强。

### 1. Git / 文档卫生

- 当前状态：2026-09-16 已开始压缩当前 status，并为压缩前状态建立 archive 快照。
- 验收条件：`PROJECT_COMPLETED.md` 只保留当前基线、已验证能力和公开边界；`PROJECT_TODO.md` 只保留下一步实验；`PROJECT_LOG.md` 只保留长期决策。
- 后续动作：复查工作区已有代码 diff，区分用户既有改动、状态压缩改动和后续实验需要提交的改动。
- 停止条件：发现代码 diff 中含密钥、私有路径、checkpoint、SwanLab 私有链接或大文件路径时，先处理公开边界，不进入实验。

### 2. Final Route Validation

- 目标：补强 `guardfix20-resume-opsd-v2-5step-20260811` 的可信度，而不是继续盲目加训练步数。
- 优先级：
  1. 跑 final route 的 `alias80`，检查答案别名、粒度和非 ASCII 风险。
  2. 做 second seed 或 non-overlap offset 小规模复现实验，验证不是 40 个 question slot 的偶然收益。
  3. 对 bridge_eval_350 做 gained/lost case review，确认收益来自 bridge/final-hop 搜索和停止策略，而不是只靠 format 收束。
  4. 若资源允许，补 bridge500 / MuSiQue source-level 分析，明确复杂多跳短板。
- 验收指标：EM/correct、format、avg search、tool success rate、missing follow-up、bad max-search loop、alias/granularity diagnostics。
- 停止条件：tool success rate < 1.0、format 明显下降、平均搜索失控、missing follow-up 增加，或 OPSD mask/token 占比异常。

### 3. Training Scale Gate

- 当前决策：不优先扩大训练步数；先做验证、case review 和更可靠 teacher/replay 设计。
- 允许重启扩大训练的条件：final route 在 alias80/second-seed/case review 中没有明显副作用，且有明确问题指向“训练覆盖不足”而非 reward/teacher 设计问题。
- 推荐扩展方式：小步试验 30/50 step、不同 seed、不同 offset 或 replay/teacher 过滤；每次必须有 dev70 health gate 和至少一个 targeted eval。
- 不推荐：直接把 OPSD resume 从 5 step 盲目拉长到 20+ step，或把 success rate < 1.0 的结果包装成正式提升。

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
- 当前工作区已有多处代码改动；提交前必须复查 `git diff --cached`、敏感信息和大文件边界。
