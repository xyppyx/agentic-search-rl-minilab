# Project Log

本文件记录需要长期追溯的重要事件、方向变化、问题解决和阶段复盘。

## 2026-07-19

- 将仓库协作规范从旧医疗后训练项目语境迁移到 Agentic RL/Search-R1 MiniLab 语境。
- 确认项目边界：`03-search-r1/` 作为上游基线参考，`my-search-r1/` 作为后续改进实现目录，`docs/design/idea.md` 作为近期路线依据。
- 完成 `my-search-r1` 首版搜索工具层：mock、Zhihu、local BM25、failure wrapper 和 registry。默认测试不访问真实网络，使用 unittest 覆盖 12 个用例。
- 完成 trajectory JSONL 与 Markdown 报告基础能力：可从工具 smoke 或已有 JSONL 生成报告，支持关键分类案例与 group comparison 区块；默认测试扩展到 16 个 unittest。
