# Project Log

本文件记录需要长期追溯的重要事件、方向变化、问题解决和阶段复盘。旧实验长流水账已在 2026-08-10 压缩，压缩前全文快照位于 `docs/status/archive/2026-08-10_pre_cleanup/`，仅供历史追溯，不作为当前事实源。

## 2026-07-19

- 将仓库协作规范从旧医疗后训练项目语境迁移到 Agentic RL/Search-R1 MiniLab 语境。
- 确认项目边界：`my-search-r1/` 是后续自有改进实现目录，`docs/design/idea.md` 是近期路线依据。
- 完成工具层、trajectory JSONL、Markdown report 和最小 PyTRIO rollout smoke，奠定后续可观测 Search-R1 MiniLab 基础。

## 2026-07-22

- 用户决策：后续训练默认采用 KL/std 稳定化组合，`train_pytrio.py` 默认值切换为 `advantage_normalization=standardize`、`advantage_clip=2.0`、`kl_coef=0.01`、`policy_ratio_clip=0.2`、`learning_rate=1e-5`；reward behavior penalty 默认仍关闭。
- 决策依据：简单 penalty 和部分长步数扩展存在压缩必要 follow-up 或被 PyTRIO 外部 sampling 阻塞的问题；后续不再假设步数越长越好。

## 2026-07-23

- Targeted bridge 与 alias/granularity eval 形成当前重要诊断结论：turn-level evidence credit 能明显提升 format 和 correct 数，但 bridge EM macro 与 alias/granularity EM 未全面超过 prompt-only base。
- 长期方向：不要只优化平均搜索次数；应优先保护 final-hop follow-up、属性 query 和短答案格式。

## 2026-08-05

- 完成 final-hop bridge guard 与 guard fix。新增 final-hop attribute search credit、missing-final-hop penalty 和 final-answer/max-search guard，确认这些信号能在真实失败轨迹中命中。
- 长期方向：final-hop credit 有正向价值，但 naive 扩步数和简单 penalty 都不能替代 case-driven guard 设计。

## 2026-08-06

- 完成 guard-fix 20-step 训练与 dev70 retry 有效评测；bridge150 full run 因 Zhihu 外部工具错误未过 success rate 1.0 门槛，随后生成 patched bridge150 作为分析口径。
- 长期边界：patched bridge150 可作为当前最高 bridge EM/correct 探索证据，但不能冒充独立全量 success rate 1.0 run。

## 2026-08-10

- 按用户要求创建 `Backup` 分支，归档上游 `00-loss-function/`、`01-grpo/`、`02-opd/`、`03-search-r1/` 教学目录；`main` 聚焦 Robust Search-R1 MiniLab 自有实现和面试/实验文档。
- 决策：后续若接 OPD/OPSD，不做 naive full-sequence distillation；只考虑 gated auxiliary objective，并让 GRPO/turn-level credit 继续作为主训练信号。
- 状态维护决策：在 `docs/status/archive/2026-08-10_pre_cleanup/` 保存压缩前 status 快照；archive 只供历史追溯，不作为当前进度、TODO、基线或实验结论。当前状态仍只读取 `PROJECT_COMPLETED.md`、`PROJECT_TODO.md`、`PROJECT_LOG.md`。
- 创建 OPSD 实验分支 `exp/gated-opsd`。按用户要求将状态压缩触发规则写入 `AGENTS.md` 与 `docs/AGENTS.md`，并分别提交到 `main` 与 `exp/gated-opsd`，后续切新主线、阶段完成、status 超过阈值或同一 active track 累计多个 run/retry 时先压缩 status 再继续推进。
