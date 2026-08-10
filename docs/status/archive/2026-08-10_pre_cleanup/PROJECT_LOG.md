# Project Log

本文件记录需要长期追溯的重要事件、方向变化、问题解决和阶段复盘。

## 2026-07-19

- 将仓库协作规范从旧医疗后训练项目语境迁移到 Agentic RL/Search-R1 MiniLab 语境。
- 确认项目边界：`03-search-r1/` 作为上游基线参考，`my-search-r1/` 作为后续改进实现目录，`docs/design/idea.md` 作为近期路线依据。
- 完成 `my-search-r1` 首版搜索工具层：mock、Zhihu、local BM25、failure wrapper 和 registry。默认测试不访问真实网络，使用 unittest 覆盖 12 个用例。
- 完成 trajectory JSONL 与 Markdown 报告基础能力：可从工具 smoke 或已有 JSONL 生成报告，支持关键分类案例与 group comparison 区块；默认测试扩展到 16 个 unittest。
- 新增从 `03-search-r1/` 迁移而来的最小 PyTRIO rollout/eval smoke 代码：协议解析、`Answer:` reward、`ToolRegistry` 调度、公开 MiniLab 小数据和 Markdown 报告入口已实现；本地 fake sampler 单元测试通过。随后配置 `PYTRIO_API_KEY` 后完成真实 smoke：2 条 fixture、EM 0.5、format rate 1.0、平均搜索次数 0.5、工具失败率 0.0，输出位于 `my-search-r1/outputs/rollout_smoke/`。

## 2026-07-22

- 用户决策：由于 `std+clip only` 与 `KL only` 消融实验被 PyTRIO sampling 阻塞，先将 KL/std 组合作为项目合理必备技术手段，优先尝试 `prompt_search_budget_guard` + KL/std 的 20-step 扩展。首次 20-step 尝试在 step 1 rollout 的 `Step 1/20 rollout: 0/8` 超过 2 分钟无进展，栈停在 PyTRIO `sample_async -> await response`，按项目规则中断；未产生 checkpoint、trajectory 或 eval 结果。复盘记录见 `docs/interview/lesson/2026-07-22_kl_std_20step_sampling_blocked.md`。
- 用户决策：后续训练默认采用 KL/std 组合，`train_pytrio.py` 默认值切换为 `advantage_normalization=standardize`、`advantage_clip=2.0`、`kl_coef=0.01`、`policy_ratio_clip=0.2`、`learning_rate=1e-5`；reward penalty 默认仍保持关闭。
- 用户决策：`turn_credit_evidence_bridge_v2` 20-step 达到 dev 70 EM 0.4429、format 1.0000、平均搜索 1.7714 后，直接尝试扩大到 50-step。50-step run 在 step 0 `prepare sampler` 阶段因 PyTRIO actor event submit retryable SSL EOF 失败，未进入 rollout、未生成 trajectory、optimizer update 或 checkpoint；本次记录为远程 actor/网络侧初始化阻塞，不计入 reward 机制效果。复盘记录见 `docs/interview/lesson/2026-07-22_turn_level_evidence_credit_v2_50step_attempt.md`。

## 2026-07-23

- Targeted bridge 三模型对比推进：prompt-only base 与 `turn_credit_evidence_bridge_20step` 在 `bridge_eval_150.jsonl` 上均达到 Zhihu success rate 1.0；20-step overall correct 81/150，高于 prompt-only base 的 74/150，但 EM macro 0.4583 低于 prompt-only base 的 0.4750。`turn_credit_evidence_bridge_50step` 已运行但出现 1 次 Zhihu `parse_error: TypeError`，success rate 0.9979，未通过 targeted eval 每个 run success rate 1.0 的门槛，因此暂不作为正式三模型结论。复盘记录见 `docs/interview/lesson/2026-07-23_bridge_targeted_three_way_eval.md`。
- Alias/granularity targeted eval 首次 prompt-only base 尝试生成 80 条 trajectory，但 `dev_8490` 出现 1 次 Zhihu `parse_error: TypeError`，success rate 0.9924，未通过 success rate 1.0 门槛；按项目规则停止，没有继续运行 `turn_credit_evidence_bridge_20step`。复盘记录见 `docs/interview/lesson/2026-07-23_alias_targeted_base_attempt.md`。
- 按用户要求修复 Zhihu `parse_error` 可观测性：后续 parse error 会在 metadata 中记录截断后的 API response 摘录和状态码，但不记录 Authorization/API key。修复后重跑 alias prompt-only base 达到 success rate 1.0，并完成 alias base vs 20-step 对比：20-step format 更高、搜索更少，但 EM macro/overall correct 从 0.4500/36 降到 0.4375/35，gained 1/lost 2。复盘记录见 `docs/interview/lesson/2026-07-23_alias_base_vs_20step_eval.md`。
- 完成两个 targeted eval 集的 base vs 20-step case review。Bridge 的 12 个 gained 全部是 base max-search/invalid format 被 20-step 修复，5 个 lost 主要来自 early answer、final-hop follow-up 被压缩或比较绑定错误；alias/granularity 的 1 个 gained 同样是格式收束收益，2 个 lost 是国籍/实体绑定早答错。后续方向应优先做 final-hop guard、属性 query 保护和 answer normalization diagnostics，而不是默认继续增加训练步数。复盘记录见 `docs/interview/lesson/2026-07-23_targeted_base_vs_20step_case_review.md`。

## 2026-08-05

- 创建 `exp/final-hop-bridge-guard` 分支并完成 `turn_credit_final_hop_bridge_v3`。本轮实现 final-hop 属性 search credit 与 missing-final-hop final-answer penalty，完成本地测试、local smoke、Zhihu dev-5 health、5-step Zhihu 训练、dev70 和 bridge150 eval。Bridge150 是有效 run，Zhihu success rate 1.0，EM macro 0.4767、correct 77/150，略高于 prompt-only base 的 0.4750、74/150；但 format 只有 0.7600，未达到预设 0.9000 门槛，也弱于 evidence 20-step 的 81/150、format 0.9400。长期决策：final-hop 属性 credit 保留为有价值方向，但当前版本只算 partial positive；不自动扩到 20-step，下一步先修 format/max-search/final-answer guard，并让 missing-final-hop detector 在真实 wrong-valid 样本上产生可解释命中。复盘记录见 `docs/interview/lesson/2026-08-05_final_hop_bridge_5step_eval.md`。
- 完成 final-hop v3 guard fix：新增默认关闭的 `--final-answer-guard-turn-penalty`，用于惩罚搜过后 max-search 不答或 invalid final answer；同时收紧 `missing_final_hop_attribute` 覆盖判断，使 date 类不再被 observation 中任意年份误判为已覆盖。对既有 bridge150 JSONL 离线分析后，`final_answer_guard_penalty_records=34`，`missing_final_hop_penalty_records=1`，说明两类训练信号已能在真实失败轨迹中命中。长期决策：这仍只是训练信号修正，不代表指标已提升；必须重新跑 5-step Zhihu train/dev70/bridge150 eval 后才能更新模型效果结论。复盘记录见 `docs/interview/lesson/2026-08-05_final_hop_guard_fix.md`。
- 按 guard fix 参数完成 `turn-credit-final-hop-guardfix-5step-20260805` 真实 Zhihu 5-step 训练。训练阶段工具事件 86/86 成功，记录 `final_answer_guard=6`、`missing_final_hop_attribute=2`，说明修正后的训练信号进入真实 rollout。Dev70 有效，Zhihu success rate 1.0，EM 0.4286、format 0.9857、平均搜索 1.8000。Bridge150 eval 运行完成但出现 7 个 Zhihu `url_error`，success rate 0.9855，未通过 targeted eval 工具门槛；参考值 EM macro 0.5042、correct 79/150、format 0.7933 不能作为正式比较结论。长期决策：不扩 20-step，若要证明 guard fix 是否超过 bridge base，先重跑 bridge150 eval 并要求 success rate 1.0。复盘记录见 `docs/interview/lesson/2026-08-05_final_hop_guardfix_5step_train_eval.md`。
- 按用户要求强制只用第二个 Zhihu key 尝试 guard-fix bridge rerun，先跑 bridge-5 health。该 health run 生成 5 条 trajectory，但 14 次 Zhihu 请求全部 rate-limited，success rate 0.0、rate-limit rate 1.0；因此没有继续跑完整 bridge150。长期决策：当前两个 key 都不能支撑正式 targeted bridge rerun，需等待 quota 恢复或提供新的可用 key。复盘记录继续写入 `docs/interview/lesson/2026-08-05_final_hop_guardfix_5step_train_eval.md`。
- 按用户要求再用原 Zhihu key 重跑 guard-fix bridge150。bridge-5 health 通过，完整 bridge150 运行完成，但 `dev_7742` 出现 1 个 `parse_error`，Zhihu success rate 0.9979，仍未通过 success rate 1.0 的正式工具门槛；参考值 EM macro 0.4817、correct 79/150、format 0.7867 不能作为正式比较结论。长期决策：当前 guard-fix checkpoint 仍无正式 bridge150 结果，不能用参考 EM 包装简历指标。
- 按用户要求验证“忽略唯一工具错误并单独补跑错误 case”的可行性。`eval_pytrio.py` 暂无 `--example-id` 参数，但可从 `bridge_eval_150.jsonl` 抽取第 20 行 `dev_7742` 生成单样本 JSONL，并使用同一 final sampler weights 补跑。单样本补跑 Zhihu requests 4、success rate 1.0、exact match 1、format 1.0；替换原失败记录后的 patched bridge150 为 EM macro 0.4842、correct 80/150、format 0.7933、tool failures 0。长期决策：patched 结果可用于分析外部工具噪声影响，说明 guard-fix bridge EM/correct 小幅超过 prompt-only base，但必须显式标注 patched 协议；主要短板仍是 format/max-search no-answer，不应继续把精力放在全量重跑上。

## 2026-08-10

- 按用户要求创建 `Backup` 分支，用于保留 `00-loss-function/`、`01-grpo/`、`02-opd/`、`03-search-r1/` 四个原教学复现目录；随后在 `main` 删除这些目录和本地残留，并更新入口文档。长期决策：`main` 聚焦 Robust Search-R1 MiniLab 自有实现和面试/实验文档，原教学内容只作为备份分支历史参考，不再作为主分支结构的一部分。
