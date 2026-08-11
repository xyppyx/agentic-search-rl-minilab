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
- 当前状态：2026-08-11 OPSD v1 evaluation ladder 已完成；local smoke、Zhihu dev-5 health、5-step train、训练后 dev-5 和 dev70 均完成且工具 success rate 为 1.0。Dev70 EM 0.4286、format 0.9286、平均搜索 1.9143，相对 guard-fix 5-step EM 持平但 format/search 退化，因此不进入 bridge150 或 alias80。
- 关键指标：EM、correct、format、平均搜索、`missing_followup_query`、`bad_max_search_loop`、tool success rate、OPSD mask 命中率、teacher-student logprob gap。
- 停止条件：Zhihu API 出现 429、timeout、credential/http error、`tool_failures > 0` 或 success rate < 1.0；PyTRIO sampling await 超过 2 分钟无进度；OPSD 明显降低 format 或增加 missing follow-up。

### 5. OPSD v2 Decision

- 验收条件：运行真实 Zhihu 5-step v2，确认训练阶段 tool success rate 1.0、OPSD mask 非零且不失控，再跑 dev70 小预算对照。
- 当前状态：v2 代码与 local smoke 已完成；尚未运行真实 Zhihu v2 训练。
- 推荐命令要点：`--opsd-coef 0.01 --opsd-mask-policy credited_turns --opsd-positive-policy positive_advantage --opsd-min-teacher-logprob -3.0`，其余参数沿用 guard-fix 5-step。
- 停止条件：训练阶段 Zhihu success rate < 1.0、OPSD masked tokens 长期为 0、dev70 format 低于 guard-fix 5-step、平均搜索继续上升、`missing_followup_query` 增加，或 OPSD mask/token 占比失控。

## Parked Historical Tracks

- Reward penalty v1/v2/v3/v4、max-search penalty、follow-up bonus、no-search guard 等旧 ablation 暂停，不再作为活跃 TODO。
- KL/std 单因素消融优先级下调；当前默认训练配置继续保留 KL/std 稳定化组合。
- Evidence-v2 50-step 和 guard-fix 独立 full bridge150 可作为后续严谨评测补充，但不是 gated OPSD 前置条件。

## 未解决风险

- 真实 Zhihu Search API、PyTRIO 远程训练和 SwanLab 依赖外部服务状态；任何真实搜索实验都必须先做 health check，并分开记录工具失败与模型策略失败。
- 当前 bridge150 最强 guard-fix 指标是 patched protocol，不应包装成论文式独立 full run。
- OPSD/OPD 类方法容易把 teacher 的短答案偏好转化成少搜/早答，需要 gate、mask 和 stop condition 约束。
- Alias/granularity eval 尚未验证 guard-fix 20-step，后续如果宣称泛化收益必须补跑。
