# Project Log

本文件记录需要长期追溯的重要事件、方向变化、问题解决和阶段复盘。

## 2026-07-19

- 将仓库协作规范从旧医疗后训练项目语境迁移到 Agentic RL/Search-R1 MiniLab 语境。
- 确认项目边界：`03-search-r1/` 作为上游基线参考，`my-search-r1/` 作为后续改进实现目录，`docs/design/idea.md` 作为近期路线依据。
- 完成 `my-search-r1` 首版搜索工具层：mock、Zhihu、local BM25、failure wrapper 和 registry。默认测试不访问真实网络，使用 unittest 覆盖 12 个用例。
- 完成 trajectory JSONL 与 Markdown 报告基础能力：可从工具 smoke 或已有 JSONL 生成报告，支持关键分类案例与 group comparison 区块；默认测试扩展到 16 个 unittest。
- 新增从 `03-search-r1/` 迁移而来的最小 PyTRIO rollout/eval smoke 代码：协议解析、`Answer:` reward、`ToolRegistry` 调度、公开 MiniLab 小数据和 Markdown 报告入口已实现；本地 fake sampler 单元测试通过。随后配置 `PYTRIO_API_KEY` 后完成真实 smoke：2 条 fixture、EM 0.5、format rate 1.0、平均搜索次数 0.5、工具失败率 0.0，输出位于 `my-search-r1/outputs/rollout_smoke/`。

## 2026-07-22

- 用户决策：由于 `std+clip only` 与 `KL only` 消融实验被 PyTRIO sampling 阻塞，先将 KL/std 组合作为项目合理必备技术手段，优先尝试 `prompt_search_budget_guard` + KL/std 的 20-step 扩展。首次 20-step 尝试在 step 1 rollout 的 `Step 1/20 rollout: 0/8` 超过 2 分钟无进展，栈停在 PyTRIO `sample_async -> await response`，按项目规则中断；未产生 checkpoint、trajectory 或 eval 结果。复盘记录见 `docs/interview/2026-07-22_kl_std_20step_sampling_blocked.md`。
- 用户决策：后续训练默认采用 KL/std 组合，`train_pytrio.py` 默认值切换为 `advantage_normalization=standardize`、`advantage_clip=2.0`、`kl_coef=0.01`、`policy_ratio_clip=0.2`、`learning_rate=1e-5`；reward penalty 默认仍保持关闭。
- 用户决策：`turn_credit_evidence_bridge_v2` 20-step 达到 dev 70 EM 0.4429、format 1.0000、平均搜索 1.7714 后，直接尝试扩大到 50-step。50-step run 在 step 0 `prepare sampler` 阶段因 PyTRIO actor event submit retryable SSL EOF 失败，未进入 rollout、未生成 trajectory、optimizer update 或 checkpoint；本次记录为远程 actor/网络侧初始化阻塞，不计入 reward 机制效果。复盘记录见 `docs/interview/2026-07-22_turn_level_evidence_credit_v2_50step_attempt.md`。

## 2026-07-23

- Targeted bridge 三模型对比推进：prompt-only base 与 `turn_credit_evidence_bridge_20step` 在 `bridge_eval_150.jsonl` 上均达到 Zhihu success rate 1.0；20-step overall correct 81/150，高于 prompt-only base 的 74/150，但 EM macro 0.4583 低于 prompt-only base 的 0.4750。`turn_credit_evidence_bridge_50step` 已运行但出现 1 次 Zhihu `parse_error: TypeError`，success rate 0.9979，未通过 targeted eval 每个 run success rate 1.0 的门槛，因此暂不作为正式三模型结论。复盘记录见 `docs/interview/2026-07-23_bridge_targeted_three_way_eval.md`。
- Alias/granularity targeted eval 首次 prompt-only base 尝试生成 80 条 trajectory，但 `dev_8490` 出现 1 次 Zhihu `parse_error: TypeError`，success rate 0.9924，未通过 success rate 1.0 门槛；按项目规则停止，没有继续运行 `turn_credit_evidence_bridge_20step`。复盘记录见 `docs/interview/2026-07-23_alias_targeted_base_attempt.md`。
- 按用户要求修复 Zhihu `parse_error` 可观测性：后续 parse error 会在 metadata 中记录截断后的 API response 摘录和状态码，但不记录 Authorization/API key。修复后重跑 alias prompt-only base 达到 success rate 1.0，并完成 alias base vs 20-step 对比：20-step format 更高、搜索更少，但 EM macro/overall correct 从 0.4500/36 降到 0.4375/35，gained 1/lost 2。复盘记录见 `docs/interview/2026-07-23_alias_base_vs_20step_eval.md`。
- 完成两个 targeted eval 集的 base vs 20-step case review。Bridge 的 12 个 gained 全部是 base max-search/invalid format 被 20-step 修复，5 个 lost 主要来自 early answer、final-hop follow-up 被压缩或比较绑定错误；alias/granularity 的 1 个 gained 同样是格式收束收益，2 个 lost 是国籍/实体绑定早答错。后续方向应优先做 final-hop guard、属性 query 保护和 answer normalization diagnostics，而不是默认继续增加训练步数。复盘记录见 `docs/interview/2026-07-23_targeted_base_vs_20step_case_review.md`。
