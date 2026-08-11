# Project Completed

本文件记录当前可引用的已完成事实、产物、结果和最终决策。压缩前的历史全文快照位于 `docs/status/archive/2026-08-10_pre_cleanup/`，仅供追溯参考，不作为当前项目状态事实源。

## 当前可引用结论快照

截至 2026-08-11，项目主线是 Robust Search-R1 MiniLab：在 Qwen3.5-4B、PyTRIO GRPO 和真实/可模拟搜索工具环境下，构建可观测、可诊断、可复盘的搜索型 Agentic RL 实验框架。

当前最重要结论：

- `prompt_search_budget_guard` 是当前最强 prompt-only base：Zhihu dev70 EM 0.4143、format 0.8857。
- `turn_credit_evidence_bridge_20step` 是当前最高 format checkpoint 证据：dev70 EM 0.4429、format 1.0000、平均搜索 1.7714；bridge150 EM 0.4583、correct 81/150、format 0.9400、平均搜索 3.0933。
- `turn-credit-final-hop-guardfix-20step-20260806` 是当前最高 EM/correct 探索证据：dev70 retry EM 0.4571、correct 32/70、format 0.9571、平均搜索 1.9000；bridge150 patched EM 0.5142、correct 83/150、format 0.8267、平均搜索 3.2000。
- bridge150 的 guard-fix 20-step 最强结果采用 patched protocol，由 full run 中非工具失败记录加失败样本 retry 合成；它可用于项目分析，但不等同一次独立全量 success rate 1.0 run。
- `gated-opsd-guardfix-5step-20260811` 已完成真实 PyTRIO OPSD 训练与 dev70 有效评测：dev70 EM 0.4286、correct 30/70、format 0.9286、平均搜索 1.9143、Zhihu success rate 1.0。结论是 OPSD v1 工程链路跑通，但 `opsd_coef=0.05` 未超过当前 turn-credit 主线，不扩到 bridge150/alias80。
- `alias_granularity_eval_80` 已完成 prompt-only base 与 evidence-v2 20-step 对比：base EM 0.4500、correct 36/80、format 0.9250；evidence-v2 20-step EM 0.4375、correct 35/80、format 0.9625。guard-fix 20-step 尚未在该 eval 集上验证。
- turn-level credit 的主要正向收益是改善 evidence bridge search、final-hop attribute search、停止策略和部分 format；主要短板仍是 bridge 场景下的 format/max-search no-answer，以及部分 early answer/final-hop follow-up 被压缩。

## 已验证核心能力

- 统一搜索工具层已完成，支持 `mock_search`、`local_bm25`、`zhihu_search`、多 key 解析、错误脱敏、failure injection 和 backend registry。
- trajectory JSONL 与 Markdown report 已完成，支持正确/错误/格式错误/工具失败/重复搜索/行为 bucket/group comparison 等复盘视图。
- Search-R1 rollout、PyTRIO train/eval CLI、数据准备、checkpoint 分析、offline diagnostics、reward sensitivity、turn-credit analysis 和 gained/lost case review 已完成。
- Gated OPSD v1 已完成实现和真实训练验证，支持 `--opsd-coef`、`--opsd-context-policy same_context`、`--opsd-mask-policy final_and_credited`、teacher logprob 对齐、OPSD mask 指标和 custom loss metrics。
- 训练默认稳定化配置已切换为 standardized advantage、advantage clip 2.0、KL-style reference drift penalty 0.01、policy ratio clip 0.2、learning rate 1e-5；reward behavior penalty 默认仍关闭。
- Reward shaping 已验证过多轮路线：简单 duplicate/empty/max-search penalty 能降低部分坏行为但可能损伤必要 follow-up；prompt/rollout 约束和 turn-level credit 更适合当前 Search-R1 MiniLab。
- Zhihu backend 的 parse/url/rate-limit 等工具异常已进入 trajectory 与报告，项目评测规则要求模型策略问题和外部工具失败分开记录。

## 当前 Baseline 表

| 场景 | 模型/策略 | EM macro | Correct | Format | Avg search | 备注 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| dev70 | prompt-only best | 0.4143 | - | 0.8857 | - | `prompt_search_budget_guard` |
| dev70 | evidence-v2 20-step | 0.4429 | - | 1.0000 | 1.7714 | 高 format checkpoint |
| dev70 | gated OPSD guard-fix 5-step | 0.4286 | 30/70 | 0.9286 | 1.9143 | OPSD v1 有效评测，未超过 turn-credit 主线 |
| dev70 | guard-fix 20-step retry | 0.4571 | 32/70 | 0.9571 | 1.9000 | 当前最高 dev70 EM |
| bridge150 | prompt-only base | 0.4750 | 74/150 | 0.7200 | 3.3067 | independent full run |
| bridge150 | evidence-v2 20-step | 0.4583 | 81/150 | 0.9400 | 3.0933 | independent full run |
| bridge150 | guard-fix 20-step patched | 0.5142 | 83/150 | 0.8267 | 3.2000 | patched protocol，不等同独立 full run |
| alias80 | prompt-only base | 0.4500 | 36/80 | 0.9250 | 1.6500 | independent full run |
| alias80 | evidence-v2 20-step | 0.4375 | 35/80 | 0.9625 | 1.5125 | independent full run |

## 历史阶段索引

旧实验细节不再写入当前 status；需要追溯时读取下列复盘文档或 archive 快照。

- 2026-07-19 至 2026-07-20：工具层、trajectory report、rollout smoke、PyTRIO train/eval 迁移、数据准备和首次端到端验证。
- 2026-07-21：reward penalty、offline diagnostics、reward sensitivity、penalty v2/v3/v4、多组 5-step/20-step 对比与 gained/lost review。
- 2026-07-22：prompt constraints、search budget guard、KL/std 稳定化、turn-level evidence credit v2、小预算与 20-step/50-step 尝试。
- 2026-07-23：bridge150 与 alias80 targeted eval、parse error 可观测性修复、base vs 20-step case review。
- 2026-08-05 至 2026-08-06：final-hop bridge guard、guard-fix、5-step/20-step 训练、dev70/bridge150 full 与 patched 评测。
- 2026-08-06 至 2026-08-10：面向保研/实习简历的项目材料整理、Backup 分支归档上游教学目录、main 聚焦自有实现。
- 2026-08-11：Gated OPSD v1 local smoke、Zhihu dev-5 health、5-step 真实训练、dev70 eval 与诊断完成；详见 `docs/interview/lesson/2026-08-11_gated_opsd_5step_train.md`。

## 公开边界

- 本项目是个人学习型 POC，不代表 PyTRIO、SwanLab、知乎开放平台、Search-R1 官方实现或原作者参与、委托或认可。
- 公开文档不得记录真实 API key、远程 sampler weights URI、SwanLab 私有链接、模型权重、checkpoint、私有服务器地址或账号。
- success rate < 1.0 的真实搜索 full run 不进入正式模型效果表；patched protocol 必须显式标注组成和边界。
