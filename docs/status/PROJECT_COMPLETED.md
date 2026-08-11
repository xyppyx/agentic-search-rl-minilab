# Project Completed

本文件记录当前可引用的已完成事实、产物、结果和最终决策。压缩前的历史全文快照位于 `docs/status/archive/2026-08-10_pre_cleanup/`，仅供追溯参考，不作为当前项目状态事实源。

## 当前可引用结论快照

截至 2026-08-11，项目主线是 Robust Search-R1 MiniLab：在 Qwen3.5-4B、PyTRIO GRPO 和真实/可模拟搜索工具环境下，构建可观测、可诊断、可复盘的搜索型 Agentic RL 实验框架。

当前最重要结论：

- 最终路线已固定为 `turn-credit-final-hop-guardfix-20step-20260806 -> guardfix20-resume-opsd-v2-5step-20260811`：先用 guard-fix 20-step 学 final-hop/bridge 搜索策略，再从其 final state 恢复做 OPSD v2 5-step gated conservative refinement。公开参数与选择理由见 `docs/interview/lesson/2026-08-11_final_route_guardfix20_opsd_v2_5step.md`。
- `prompt_search_budget_guard` 是当前最强 prompt-only base：Zhihu dev70 EM 0.4143、format 0.8857。
- `turn_credit_evidence_bridge_20step` 是当前最高 format checkpoint 证据：dev70 EM 0.4429、format 1.0000、平均搜索 1.7714；bridge150 EM 0.4583、correct 81/150、format 0.9400、平均搜索 3.0933。
- `guardfix20-resume-opsd-v2-5step-20260811` 是当前最高 dev70 clean checkpoint：从 `turn-credit-final-hop-guardfix-20step-20260806` final state 恢复，使用 guardfix final weights 作为 KL/reference 与 OPSD teacher，dev70 EM 0.4857、correct 34/70、format 0.9857、平均搜索 1.7286、Zhihu success rate 1.0。
- `turn-credit-final-hop-guardfix-20step-20260806` 是此前最高 EM/correct 探索证据：dev70 retry EM 0.4571、correct 32/70、format 0.9571、平均搜索 1.9000；bridge150 patched EM 0.5142、correct 83/150、format 0.8267、平均搜索 3.2000。
- bridge150 的 guard-fix 20-step 最强结果采用 patched protocol，由 full run 中非工具失败记录加失败样本 retry 合成；它可用于项目分析，但不等同一次独立全量 success rate 1.0 run。
- `gated-opsd-guardfix-5step-20260811` 已完成真实 PyTRIO OPSD 训练与 dev70 有效评测：dev70 EM 0.4286、correct 30/70、format 0.9286、平均搜索 1.9143、Zhihu success rate 1.0。结论是 OPSD v1 工程链路跑通，但 `opsd_coef=0.05` 未超过当前 turn-credit 主线，不扩到 bridge150/alias80。
- `gated-opsd-v2-guardfix-5step-20260811` 已完成真实 PyTRIO 训练与 dev70 有效评测：dev70 EM 0.4143、correct 29/70、format 0.9143、平均搜索 1.9714、Zhihu success rate 1.0。结论是正向 gate 收窄了 mask，但没有修复 OPSD 对 format/search 的干扰，不扩到 20-step、bridge150 或 alias80。
- `gated-opsd-v2-guardfix-20step-20260811` 已按用户要求完成真实 PyTRIO 20-step 压力测试：训练阶段 160 条轨迹、correct 52/160、format 0.9000、平均搜索 2.0188、OPSD mask rate 0.0088、Zhihu success rate 1.0；最终 dev5 EM 0.4000、format 1.0000。dev70 reference 为 EM 0.4429、correct 31/70、format 1.0000、平均搜索 2.0286，但因 1 次 Zhihu parse error 导致 success rate 0.9930，未进入正式 baseline 表。
- `guardfix20-resume-opsd-v2-20step-20260811` 已完成从 guardfix final state 恢复的 20-step reference run：训练阶段 correct 61/160、format 0.9438、平均搜索 1.9563、OPSD mask rate 0.0098，但训练阶段有 1 次工具错误；最终 dev70 reference EM 0.4571、correct 32/70、format 0.9714、平均搜索 2.0571，因 dev70 success rate 0.9931 未进入正式 baseline 表。
- `guardfix20-resume-opsd-v2-5step-20260811` 已完成 clean bridge150 分片 full eval：EM 0.5242、correct 87/150、format 0.9067、平均搜索 3.1400、tool failures 0；这是当前最高 bridge150 clean/patched 口径结果，但 format 低于 evidence-v2 20-step。
- `guardfix20-resume-opsd-v2-20step-seed43-20260811` 已按用户要求完成 20-step 重训、dev70 和 bridge150 clean 分片评测：dev70 EM 0.4571、correct 32/70、format 1.0000；bridge150 EM 0.5317、correct 81/150、format 0.8133、平均搜索 3.2533。结论是 20-step 宏平均 bridge EM 略高，但 correct/format/search 综合弱于 5-step，不替代最终候选。
- `alias_granularity_eval_80` 已完成 prompt-only base 与 evidence-v2 20-step 对比：base EM 0.4500、correct 36/80、format 0.9250；evidence-v2 20-step EM 0.4375、correct 35/80、format 0.9625。guard-fix 20-step 尚未在该 eval 集上验证。
- turn-level credit 的主要正向收益是改善 evidence bridge search、final-hop attribute search、停止策略和部分 format；主要短板仍是 bridge 场景下的 format/max-search no-answer，以及部分 early answer/final-hop follow-up 被压缩。

## 已验证核心能力

- 统一搜索工具层已完成，支持 `mock_search`、`local_bm25`、`zhihu_search`、多 key 解析、错误脱敏、failure injection 和 backend registry。
- trajectory JSONL 与 Markdown report 已完成，支持正确/错误/格式错误/工具失败/重复搜索/行为 bucket/group comparison 等复盘视图。
- Search-R1 rollout、PyTRIO train/eval CLI、数据准备、checkpoint 分析、offline diagnostics、reward sensitivity、turn-credit analysis 和 gained/lost case review 已完成。
- Gated OPSD v1 已完成实现和真实训练验证，支持 `--opsd-coef`、`--opsd-context-policy same_context`、`--opsd-mask-policy final_and_credited`、teacher logprob 对齐、OPSD mask 指标和 custom loss metrics。
- Gated OPSD v2 已完成实现与 local PyTRIO smoke：新增 `--opsd-positive-policy`，默认 `credited_turns + positive_advantage`，避免默认蒸馏 wrong-valid final answer；local nonzero-mask smoke 验证 OPSD teacher logprobs、custom backward 和 optimizer 路径可用。
- `eval_pytrio.py` 已增加 `--offset` 数据选择参数，用于长评测被外部 sampling session 中断时做分片恢复；默认 `--offset 0` 保持原评测行为。
- 最终路线公开训练参数已冻结：guard-fix 20-step 使用 `final_hop_bridge`、`evidence_search_turn_bonus=0.05`、`final_hop_search_turn_bonus=0.10`、`early_answer_turn_penalty=0.05`、`missing_final_hop_turn_penalty=0.08`、`final_answer_guard_turn_penalty=0.06`；OPSD v2 5-step 在其 final state 上恢复，使用 `opsd_coef=0.01`、`opsd_mask_policy=credited_turns`、`opsd_positive_policy=positive_advantage`、`opsd_min_teacher_logprob=-3.0`。
- 公开展示材料已按最终路线更新：根 `README.md`、`docs/interview/one_page_project_pitch.md`、`docs/interview/interview_qa_quick_reference.md`、`docs/learning/project/project_experience_star.md` 和 `docs/learning/project/technical_implementation_details.md` 均切换到 guard-fix 20-step + OPSD v2 5-step 口径；新增 `docs/learning/basic/rl/opd_opsd_interview_notes.md` 解释 OPD/OPSD、gated objective、mask 和 5-step 选择理由。
- 训练默认稳定化配置已切换为 standardized advantage、advantage clip 2.0、KL-style reference drift penalty 0.01、policy ratio clip 0.2、learning rate 1e-5；reward behavior penalty 默认仍关闭。
- Reward shaping 已验证过多轮路线：简单 duplicate/empty/max-search penalty 能降低部分坏行为但可能损伤必要 follow-up；prompt/rollout 约束和 turn-level credit 更适合当前 Search-R1 MiniLab。
- Zhihu backend 的 parse/url/rate-limit 等工具异常已进入 trajectory 与报告，项目评测规则要求模型策略问题和外部工具失败分开记录。

## 当前 Baseline 表

| 场景 | 模型/策略 | EM macro | Correct | Format | Avg search | 备注 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| dev70 | prompt-only best | 0.4143 | - | 0.8857 | - | `prompt_search_budget_guard` |
| dev70 | evidence-v2 20-step | 0.4429 | - | 1.0000 | 1.7714 | 高 format checkpoint |
| dev70 | gated OPSD guard-fix 5-step | 0.4286 | 30/70 | 0.9286 | 1.9143 | OPSD v1 有效评测，未超过 turn-credit 主线 |
| dev70 | gated OPSD v2 guard-fix 5-step | 0.4143 | 29/70 | 0.9143 | 1.9714 | OPSD v2 有效评测，弱于 v1 和 guard-fix 主线 |
| dev70 | guard-fix 20-step retry | 0.4571 | 32/70 | 0.9571 | 1.9000 | 此前最高 dev70 EM |
| dev70 | guardfix20 resume OPSD v2 5-step | 0.4857 | 34/70 | 0.9857 | 1.7286 | 当前最高 clean dev70 |
| dev70 | guardfix20 resume OPSD v2 20-step seed43 | 0.4571 | 32/70 | 1.0000 | 1.8857 | clean 方差对照，未超过 5-step |
| bridge150 | prompt-only base | 0.4750 | 74/150 | 0.7200 | 3.3067 | independent full run |
| bridge150 | evidence-v2 20-step | 0.4583 | 81/150 | 0.9400 | 3.0933 | independent full run |
| bridge150 | guard-fix 20-step patched | 0.5142 | 83/150 | 0.8267 | 3.2000 | patched protocol，不等同独立 full run |
| bridge150 | guardfix20 resume OPSD v2 5-step | 0.5242 | 87/150 | 0.9067 | 3.1400 | 10 个 clean chunks 合并，tool failures 0 |
| bridge150 | guardfix20 resume OPSD v2 20-step seed43 | 0.5317 | 81/150 | 0.8133 | 3.2533 | clean 方差对照，宏平均高但综合弱于 5-step |
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
- 2026-08-11：Gated OPSD v1 local smoke、Zhihu dev-5 health、5-step 真实训练、dev70 eval 与诊断完成；随后完成 OPSD v2 正向 gate 实现、local smoke、真实 5-step 训练、dev70 eval、用户指定的 v2 20-step 压力测试，以及从 guardfix-20step final state 恢复的 OPSD v2 5-step/20-step 对照。bridge150 经分片重试完成 clean full eval，并完成 20-step seed43 重训方差对照；最终路线固定为 guard-fix 20-step + OPSD v2 5-step。详见 `docs/interview/lesson/2026-08-11_gated_opsd_5step_train.md`、`docs/interview/lesson/2026-08-11_gated_opsd_v2_implementation.md`、`docs/interview/lesson/2026-08-11_gated_opsd_v2_5step_train.md`、`docs/interview/lesson/2026-08-11_gated_opsd_v2_20step_train.md`、`docs/interview/lesson/2026-08-11_guardfix20_resume_opsd_v2_train.md`、`docs/interview/lesson/2026-08-11_guardfix20_resume_opsd_v2_bridge150_attempt.md`、`docs/interview/lesson/2026-08-11_guardfix20_resume_opsd_v2_20step_seed43.md` 和 `docs/interview/lesson/2026-08-11_final_route_guardfix20_opsd_v2_5step.md`。

## 公开边界

- 本项目是个人学习型 POC，不代表 PyTRIO、SwanLab、知乎开放平台、Search-R1 官方实现或原作者参与、委托或认可。
- 公开文档不得记录真实 API key、远程 sampler weights URI、SwanLab 私有链接、模型权重、checkpoint、私有服务器地址或账号。
- success rate < 1.0 的真实搜索 full run 不进入正式模型效果表；patched protocol 必须显式标注组成和边界。
