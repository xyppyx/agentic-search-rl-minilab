# Project Completed

本文件记录已完成且有验证证据的事实、产物、结果和最终决策。未验证的代码或实验不得写入完成项。

## 2026-07-19

- 项目定位已明确为基于上游 `llm-agent-rl-lab` 的 Search-R1/Agentic RL 学习与改进项目；当前主线是 `docs/design/idea.md` 中的 Robust Search-R1 MiniLab。
  - 相关路径：`AGENTS.md`、`README.md`、`docs/status/`、`my-search-r1/`
  - 验证方式：读取仓库结构与 `docs/design/idea.md`；检查所有一级目录均具备 `AGENTS.md` 和 `README.md`；运行 `git diff --check` 无空白错误。
  - 关键结果：协作规范已从旧医疗后训练项目语境迁移到 Search-R1 MiniLab 项目语境，状态事实源和 `my-search-r1/` 目录规则已建立。

- `my-search-r1/` 首版搜索工具层已实现。
  - 相关路径：`my-search-r1/search_r1_minilab/tools/`、`my-search-r1/tests/test_tools.py`、`my-search-r1/tests/fixtures/bm25_corpus.jsonl`
  - 验证方式：运行 `PYTHONPATH=my-search-r1 uv run python -m unittest discover -s my-search-r1/tests -v`。
  - 关键结果：12 个 unittest 全部通过；已覆盖 mock fixture/空结果、Zhihu key 解析和错误脱敏、local BM25 排序/空 query/无命中、failure wrapper 固定 seed 与失败类型指标、registry 分发，以及 mock/local BM25 smoke record schema 一致性。

- `my-search-r1/` trajectory JSONL 保存与 Markdown 报告已实现。
  - 相关路径：`my-search-r1/search_r1_minilab/trajectories/`、`my-search-r1/scripts/analyse_trajectories.py`、`my-search-r1/scripts/tool_smoke.py`、`my-search-r1/tests/test_trajectories.py`
  - 验证方式：运行 `PYTHONPATH=my-search-r1 uv run python -m unittest discover -s my-search-r1/tests -v`；运行 `tool_smoke.py` 分别验证 `mock_search` 和 `local_bm25` 输出 JSONL 与 Markdown；运行 `analyse_trajectories.py` 从已有 JSONL 重新生成 Markdown。
  - 关键结果：16 个 unittest 全部通过；`/tmp/search-r1-tool-smoke.jsonl`、`/tmp/search-r1-tool-smoke.md`、`/tmp/search-r1-bm25-smoke.jsonl`、`/tmp/search-r1-bm25-smoke.md` 和 `/tmp/search-r1-tool-smoke-reanalyse.md` 均成功生成。报告支持 summary、correct、wrong、invalid_format、tool_failure、repeated_search 和 group comparison 区块。
