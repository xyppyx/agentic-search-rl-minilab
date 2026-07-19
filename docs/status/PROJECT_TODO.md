# Project TODO

本文件记录进行中任务、下一步、验收条件、阻塞项和未解决风险。完成且验证后移入 `PROJECT_COMPLETED.md`。

## 近期任务

- 从 `03-search-r1/` 基线迁移最小可运行 Search-R1 smoke/eval 链路。
  - 验收条件：在不依赖真实付费搜索服务的 mock backend 下，可生成 trajectory JSONL 和 Markdown 报告。
  - 当前状态：待开始。

- 将 `my-search-r1` 工具层接入真实 rollout。
  - 验收条件：rollout 通过 `ToolRegistry` 调用 `mock_search`、`local_bm25` 或 `zhihu_search`，不直接依赖具体搜索 client，并输出同一 trajectory JSONL schema。
  - 当前状态：待开始。

- 增加公开 smoke 配置与命令入口。
  - 验收条件：无需真实 API key，可用 mock 或 local BM25 跑一个端到端 rollout smoke，并输出固定路径产物。
  - 当前状态：工具 smoke 已完成，模型 rollout smoke 待开始。

## 未解决风险

- `my-search-r1/` 当前已有搜索工具层、trajectory JSONL 和报告能力，但尚未接入模型 rollout、训练或评测。
- 真实知乎搜索 API、PyTRIO 远程训练和 SwanLab 记录依赖外部凭据与服务状态，后续实验需要 mock baseline 与真实 backend 指标分开记录。
- 当前已有 `03-search-r1/train.py` 本地改动，后续修改基线文件前需要继续保护这部分改动。
