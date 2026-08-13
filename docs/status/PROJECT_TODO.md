# Project TODO

本文件只记录当前活跃任务、下一步、验收条件、阻塞项和未解决风险。旧 reward shaping 与 targeted eval 流水账已压缩到 `PROJECT_COMPLETED.md` 的历史阶段索引；压缩前全文位于 `docs/status/archive/2026-08-10_pre_cleanup/`，仅供追溯参考。

## Active Track: Gated OPSD

目标：在不破坏现有 turn-level credit 主线的前提下，评估是否能加入 gated on-policy self-distillation 作为辅助 loss，为 Search-R1 Agent 提供更密集的 token-level 训练信号。

### 1. Freeze Pre-OPSD Baseline

- 验收条件：明确后续 OPSD 对照使用的 baseline、checkpoint、评测集、参数和公开指标口径。
- 当前状态：2026-08-11 已冻结 v1 对照口径；以 `prompt_search_budget_guard`、`turn_credit_evidence_bridge_20step`、`turn-credit-final-hop-guardfix-20step-20260806` 作为三类基线，bridge150 的 guard-fix 结果必须标注 patched protocol。
- 停止条件：如果无法定位可用 sampler weights 或必要评测 JSONL，先补状态记录，不进入 OPSD 实现。

### 2. OPSD Feasibility Smoke

- 验收条件：在 local/mock 小样本上验证 teacher/self-teacher logprob 能与 student rollout target tokens 对齐，并记录 token 数、mask 数、缺失 logprob 数和 loss 数值范围。
- 当前状态：2026-08-11 已完成 local BM25 1-step PyTRIO OPSD train smoke；`TrainingDatum`、shifted-target logprob 对齐、OPSD mask、custom loss metrics、reference logprobs、OPSD teacher logprobs、custom backward 和 checkpoint 路径均已跑通。
- 技术方向：复用现有 `compute_reference_logprobs()`、`TrainingDatum`、custom loss 路径；v1 teacher context policy 固定为 `same_context`，不构造 gold-answer teacher。
- 停止条件：teacher logprob 长度无法稳定对齐、tokenizer 不兼容、tool observation token 被错误纳入训练，或 smoke loss 出现 NaN/inf。

### 3. Gated OPSD Auxiliary Loss

- 验收条件：实现默认关闭的 `--opsd-coef`、`--opsd-context-policy`、`--opsd-mask-policy`；GRPO/turn-credit 仍为主 loss，OPSD 只作为小系数辅助项。
- 当前状态：2026-08-11 已完成 `gated-opsd-guardfix-5step-20260811` 真实 Zhihu 5-step 训练；OPSD v1 训练链路可用，但 dev70 结果未超过当前 turn-credit 主线。随后已实现 OPSD v2：新增 `--opsd-positive-policy`，默认 `credited_turns + positive_advantage`，local PyTRIO smoke 已通过。
- 推荐初始策略：只在 final-answer/turn-credit 命中 turn 上启用 OPSD，不做全序列蒸馏，不蒸馏 tool observation tokens。
- 停止条件：naive full-sequence OPSD、gold answer teacher 诱导少搜/早答、或 distillation token 占比失控。

### 4. Evaluation Ladder

- 验收条件：依次完成 local BM25 1-step smoke、Zhihu dev-5 health、dev70 小预算对照；只有 dev70 不明显退化时才考虑 bridge150 或 alias80。
- 当前状态：2026-08-11 OPSD v1 evaluation ladder 已完成；local smoke、Zhihu dev-5 health、5-step train、训练后 dev-5 和 dev70 均完成且工具 success rate 为 1.0。Base 起点 OPSD v2 已完成 5-step 有效评测和用户指定的 20-step 压力测试，但未超过 turn-credit 主线。随后完成从 `turn-credit-final-hop-guardfix-20step-20260806` final state 恢复的 OPSD v2 5-step/20-step 对照；其中 5-step dev70 clean EM 0.4857、format 0.9857、平均搜索 1.7286，成为当前最高 clean dev70。
- bridge150 状态：`guardfix20-resume-opsd-v2-5step-20260811` 已完成 10 个 clean chunk 合并的 full eval，EM 0.5242、correct 87/150、format 0.9067、平均搜索 3.1400、tool failures 0。该结果超过此前 guard-fix 20-step patched EM/correct，但 format 仍弱于 evidence-v2 20-step。
- 20-step 方差对照：`guardfix20-resume-opsd-v2-20step-seed43-20260811` 已完成 clean 训练/dev70/bridge150；dev70 EM 0.4571、correct 32/70、format 1.0000，bridge150 EM 0.5317、correct 81/150、format 0.8133、平均搜索 3.2533。宏平均 bridge EM 略高，但 correct/format/search 综合弱于 5-step。
- 关键指标：EM、correct、format、平均搜索、`missing_followup_query`、`bad_max_search_loop`、tool success rate、OPSD mask 命中率、teacher-student logprob gap。
- 停止条件：Zhihu API 出现 429、timeout、credential/http error、`tool_failures > 0` 或 success rate < 1.0；PyTRIO sampling await 超过 2 分钟无进度；OPSD 明显降低 format 或增加 missing follow-up。

### 5. OPSD v2 Decision

- 验收条件：运行真实 Zhihu 5-step v2，确认训练阶段 tool success rate 1.0、OPSD mask 非零且不失控，再跑 dev70 小预算对照。
- 当前状态：2026-08-11 base 起点 v2 训练与评测已完成；5-step dev70 EM 0.4143、format 0.9143、平均搜索 1.9714，20-step dev70 reference EM 0.4429、format 1.0000、平均搜索 2.0286 但工具门槛未过。随后从 guardfix-20step final state 恢复并使用 guardfix final weights 作为 KL/reference 与 OPSD teacher：5-step dev70 clean EM 0.4857、correct 34/70、format 0.9857、平均搜索 1.7286；20-step dev70 reference EM 0.4571、format 0.9714、平均搜索 2.0571，但训练和 dev70 都有工具失败。
- 推荐命令要点：`--opsd-coef 0.01 --opsd-mask-policy credited_turns --opsd-positive-policy positive_advantage --opsd-min-teacher-logprob -3.0`，其余参数沿用 guard-fix 5-step。
- 停止条件：训练阶段 Zhihu success rate < 1.0、OPSD masked tokens 长期为 0、dev70 format 低于 guard-fix 5-step、平均搜索继续上升、`missing_followup_query` 增加，或 OPSD mask/token 占比失控。

### 6. OPSD Stop/Park

- 验收条件：明确 OPSD 分支是否停放，以及后续若重启需要满足什么设计变化。
- 当前决策：固定最终路线为 `turn-credit-final-hop-guardfix-20step-20260806 -> guardfix20-resume-opsd-v2-5step-20260811`。停放 base 起点 same-context OPSD，不再继续做同类 base 起点训练；20-step seed43 方差对照不支持替代 5-step。后续只补验证，不再改主路线。
- 允许重启条件：引入本质不同的 teacher 或监督信号，例如离线 correct trajectory teacher、answer-span 级别蒸馏、preference-filtered replay，且先在 local/mock smoke 证明不会蒸馏错误 final answer 或 tool observation tokens。

## Parked Historical Tracks

- 早期“面向不可靠搜索工具/故障注入鲁棒训练”方向已暂时搁置。保留 failure injection、tool failure 记录和 patched/clean 边界是为了 smoke、回归测试和评测可信度；后续不把“训练模型适应失败工具”作为当前主线验收目标。
- Reward penalty v1/v2/v3/v4、max-search penalty、follow-up bonus、no-search guard 等旧 ablation 暂停，不再作为活跃 TODO。
- KL/std 单因素消融优先级下调；当前默认训练配置继续保留 KL/std 稳定化组合。
- Evidence-v2 50-step 和 guard-fix 独立 full bridge150 可作为后续严谨评测补充，但不是 gated OPSD 前置条件。
- `guardfix20-resume-opsd-v2-5step-20260811` 的 alias80 尚未评测；这是最终路线的下一步验证，不是路线选择前置条件。
- `bridge_eval_350.jsonl` 已完成 base+prompt、guard-fix 20-step、最终 OPSD v2 5-step clean 三方对照。当前可支撑的公开口径是：最终路线在较明确多跳场景上相对 base 有效，guard-fix 20-step 提供中间增益，OPSD v2 5-step 在其上继续提升 format/EM/search efficiency。后续若要扩展泛化结论，应补失败 case review。

## 未解决风险

- 真实 Zhihu Search API、PyTRIO 远程训练和 SwanLab 依赖外部服务状态；任何真实搜索实验都必须先做 health check，并分开记录工具失败与模型策略失败。
- 当前 bridge150 最强 guard-fix 指标是 patched protocol，不应包装成论文式独立 full run。
- OPSD/OPD 类方法容易把 teacher 的短答案偏好转化成少搜/早答，需要 gate、mask 和 stop condition 约束。
- Same-context OPSD v1/v2 在 base 起点收益不足；在 guardfix checkpoint 上 5-step 有明确 clean dev70/bridge150 综合增益。20-step seed43 的 bridge EM macro 略高，但 correct/format/search 更弱，因此不应把“更多步数”包装成最终路线。
- Alias/granularity eval 尚未验证 guard-fix 20-step，后续如果宣称泛化收益必须补跑。
