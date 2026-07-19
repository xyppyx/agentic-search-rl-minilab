# Project Log

本文件记录需要长期追溯的重要事件、方向变化、问题解决和阶段复盘。

## 2026-07-19

- 将仓库协作规范从旧医疗后训练项目语境迁移到 Agentic RL/Search-R1 MiniLab 语境。
- 确认项目边界：`03-search-r1/` 作为上游基线参考，`my-search-r1/` 作为后续改进实现目录，`docs/design/idea.md` 作为近期路线依据。
- 完成 `my-search-r1` 首版搜索工具层：mock、Zhihu、local BM25、failure wrapper 和 registry。默认测试不访问真实网络，使用 unittest 覆盖 12 个用例。
- 完成 trajectory JSONL 与 Markdown 报告基础能力：可从工具 smoke 或已有 JSONL 生成报告，支持关键分类案例与 group comparison 区块；默认测试扩展到 16 个 unittest。
- 新增从 `03-search-r1/` 迁移而来的最小 PyTRIO rollout/eval smoke 代码：协议解析、`Answer:` reward、`ToolRegistry` 调度、公开 MiniLab 小数据和 Markdown 报告入口已实现；本地 fake sampler 单元测试通过。随后配置 `PYTRIO_API_KEY` 后完成真实 smoke：2 条 fixture、EM 0.5、format rate 1.0、平均搜索次数 0.5、工具失败率 0.0，输出位于 `my-search-r1/outputs/rollout_smoke/`。
