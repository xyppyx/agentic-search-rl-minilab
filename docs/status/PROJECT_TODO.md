# Project TODO

本文件只记录当前活跃任务、下一步、验收条件、阻塞项和未解决风险。旧 reward shaping 与 targeted eval 流水账已压缩到 `PROJECT_COMPLETED.md` 的历史阶段索引；压缩前全文位于 `docs/status/archive/2026-08-10_pre_cleanup/`，仅供追溯参考。

## Active Track: Gated OPSD

目标：在不破坏现有 turn-level credit 主线的前提下，评估是否能加入 gated on-policy self-distillation 作为辅助 loss，为 Search-R1 Agent 提供更密集的 token-level 训练信号。

### 1. Freeze Pre-OPSD Baseline

- 验收条件：明确后续 OPSD 对照使用的 baseline、checkpoint、评测集、参数和公开指标口径。
- 当前建议：以 `prompt_search_budget_guard`、`turn_credit_evidence_bridge_20step`、`turn-credit-final-hop-guardfix-20step-20260806` 作为三类基线；bridge150 的 guard-fix 结果必须标注 patched protocol。
- 停止条件：如果无法定位可用 sampler weights 或必要评测 JSONL，先补状态记录，不进入 OPSD 实现。

### 2. OPSD Feasibility Smoke

- 验收条件：在 local/mock 小样本上验证 teacher/self-teacher logprob 能与 student rollout target tokens 对齐，并记录 token 数、mask 数、缺失 logprob 数和 loss 数值范围。
- 技术方向：复用现有 `compute_reference_logprobs()`、`TrainingDatum`、custom loss 路径；新增 teacher context builder 和 OPSD mask。
- 停止条件：teacher logprob 长度无法稳定对齐、tokenizer 不兼容、tool observation token 被错误纳入训练，或 smoke loss 出现 NaN/inf。

### 3. Gated OPSD Auxiliary Loss

- 验收条件：实现默认关闭的 `--opsd-coef`、`--opsd-context-policy`、`--opsd-mask-policy`；GRPO/turn-credit 仍为主 loss，OPSD 只作为小系数辅助项。
- 推荐初始策略：只在 final-answer/guard 命中 turn 上启用 OPSD，不做全序列蒸馏，不蒸馏 tool observation tokens。
- 停止条件：naive full-sequence OPSD、gold answer teacher 诱导少搜/早答、或 distillation token 占比失控。

### 4. Evaluation Ladder

- 验收条件：依次完成 local BM25 1-step smoke、Zhihu dev-5 health、dev70 小预算对照；只有 dev70 不明显退化时才考虑 bridge150 或 alias80。
- 关键指标：EM、correct、format、平均搜索、`missing_followup_query`、`bad_max_search_loop`、tool success rate、OPSD mask 命中率、teacher-student logprob gap。
- 停止条件：Zhihu API 出现 429、timeout、credential/http error、`tool_failures > 0` 或 success rate < 1.0；PyTRIO sampling await 超过 2 分钟无进度；OPSD 明显降低 format 或增加 missing follow-up。

## Parked Historical Tracks

- Reward penalty v1/v2/v3/v4、max-search penalty、follow-up bonus、no-search guard 等旧 ablation 暂停，不再作为活跃 TODO。
- KL/std 单因素消融优先级下调；当前默认训练配置继续保留 KL/std 稳定化组合。
- Evidence-v2 50-step 和 guard-fix 独立 full bridge150 可作为后续严谨评测补充，但不是 gated OPSD 前置条件。

## 未解决风险

- 真实 Zhihu Search API、PyTRIO 远程训练和 SwanLab 依赖外部服务状态；任何真实搜索实验都必须先做 health check，并分开记录工具失败与模型策略失败。
- 当前 bridge150 最强 guard-fix 指标是 patched protocol，不应包装成论文式独立 full run。
- OPSD/OPD 类方法容易把 teacher 的短答案偏好转化成少搜/早答，需要 gate、mask 和 stop condition 约束。
- Alias/granularity eval 尚未验证 guard-fix 20-step，后续如果宣称泛化收益必须补跑。
