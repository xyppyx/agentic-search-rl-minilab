# Project TODO

本文件记录进行中任务、下一步、验收条件、阻塞项和未解决风险。完成且验证后移入 `PROJECT_COMPLETED.md`。

## 近期任务

- 将 `my-search-r1` 工具层接入真实 rollout。
  - 验收条件：rollout 通过 `ToolRegistry` 调用 `mock_search`、`local_bm25` 或 `zhihu_search`，不直接依赖具体搜索 client，并输出同一 trajectory JSONL schema。
  - 当前状态：最小 smoke/eval 状态机已通过真实 PyTRIO smoke 验证；训练 rollout/PyTRIO GRPO 路径尚未接入。

- 增加训练前 eval 对照和失败案例复盘。
  - 验收条件：基于 `my-search-r1/outputs/rollout_smoke/` 或后续更大 eval 产物，记录模型直接答对、搜索后答错、空结果、工具失败、重复 query 等案例，并明确下一步 reward/penalty 改造依据。
  - 当前状态：待开始。

## 未解决风险

- `my-search-r1/` 当前已有搜索工具层、trajectory JSONL、报告能力和最小 PyTRIO rollout/eval smoke，但训练 rollout、GRPO 更新和更大规模评测尚未接入。
- 真实知乎搜索 API、PyTRIO 远程训练和 SwanLab 记录依赖外部凭据与服务状态，后续实验需要 mock baseline 与真实 backend 指标分开记录。
- 当前已有 `03-search-r1/train.py` 本地改动，后续修改基线文件前需要继续保护这部分改动。
