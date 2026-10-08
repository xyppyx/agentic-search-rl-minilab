# Project Log

本文件只记录需要长期追溯的重要决策和方向变化。压缩前快照位于 `docs/status/archive/2026-09-16_pre_next_experiments/`；更早流水账快照位于 `docs/status/archive/`，仅供历史追溯，不作为当前事实源。

## 2026-07-19

- 将仓库协作规范迁移到 Agentic RL/Search-R1 MiniLab 语境。
- 确认 `my-search-r1/` 为自有改进实现目录，完成工具层、trajectory JSONL、Markdown report 和最小 PyTRIO rollout smoke。

## 2026-07-22

- 决策：默认训练配置切换为 KL/std 稳定化组合：standardized advantage、advantage clip 2.0、KL-style reference drift penalty 0.01、policy ratio clip 0.2、learning rate 1e-5。
- 决策依据：简单 penalty 和盲目扩步数可能压缩必要 follow-up；后续不假设步数越长越好。

## 2026-07-23

- Targeted bridge 与 alias/granularity eval 形成关键诊断：turn-level evidence credit 能改善 format 和 correct 数，但 bridge EM macro 与 alias/granularity EM 未全面超过 prompt-only base。
- 长期方向：优化 final-hop follow-up、属性 query 和短答案格式，而不是只优化平均搜索次数。

## 2026-08-05 至 2026-08-06

- 完成 final-hop bridge guard 与 guard fix，新增 final-hop attribute search credit、missing-final-hop penalty 和 final-answer/max-search guard。
- 完成 guard-fix 20-step 训练与 dev70 retry 有效评测；bridge150 历史强结果采用 patched protocol，不能包装成独立 clean full run。

## 2026-08-10

- 创建 `Backup` 分支归档上游教学目录；`main` 聚焦 Robust Search-R1 MiniLab 自有实现和公开项目文档。
- 决策：后续若接 OPD/OPSD，不做 naive full-sequence distillation，只考虑 gated auxiliary objective，并让 GRPO/turn-level credit 继续作为主训练信号。
- 建立 status archive/压缩规则：切换新主线、阶段完成、status 超阈值或同一 active track 累计多个 run/retry 时先压缩 status。

## 2026-08-11

- OPSD v1 和 base 起点 OPSD v2 工程链路跑通，但 dev70 未超过 turn-credit 主线；base 起点 same-context OPSD 停放。
- 从 `turn-credit-final-hop-guardfix-20step-20260806` 恢复做 OPSD v2 5-step/20-step 对照。5-step clean dev70 成为当前最高；20-step 对照不支持“更多步数更好”。
- 完成 final route 的 clean bridge150 分片评测；固定最终路线为 `turn-credit-final-hop-guardfix-20step-20260806 -> guardfix20-resume-opsd-v2-5step-20260811`。
- 长期表述：guard-fix 20-step 是搜索策略基座，OPSD v2 5-step 是强 checkpoint 上的 gated conservative refinement；后续 alias80/second-seed 属于验证补充，不改变当前主路线。

## 2026-08-12 至 2026-08-13

- 项目定位从“不可靠搜索工具鲁棒训练”收敛为“面向多跳问答/真实搜索环境的搜索型 LLM Agent 强化学习训练与评测框架”。
- 完成 `bridge_eval_350` 的 base+prompt、guard-fix 20-step、final route 三方 clean 对照；结论限定为较明确 bridge-hop 场景有效，MuSiQue/复杂多跳仍是短板。
- README 与公开设计文档已更新到最终路线、数据 split、评测集和 clean/patched 口径。

## 2026-09-03 至 2026-09-07

- 补充保研/面试材料：简历项目表述、Search-R1 面试准备、turn-level credit、Gated OPSD、数据指标 QA、GRPO/turn-credit/OPSD QA。
- 长期表达边界：主动说明小预算 POC、实际训练 trajectory 数、dev70 health gate、bridge350 targeted subset 和 final route 尚未验证 alias80。

## 2026-09-16

- 按用户要求整理 Git 与项目文档，准备后续实验前的状态压缩。
- 在 `docs/status/archive/2026-09-16_pre_next_experiments/` 保存压缩前 `PROJECT_COMPLETED.md`、`PROJECT_TODO.md`、`PROJECT_LOG.md` 和根 README 快照。
- 当前 status 收敛为：最终路线与 baseline、下一步验证补强、训练扩量门槛和公开边界；旧 run 流水账继续只在 archive 中追溯。
