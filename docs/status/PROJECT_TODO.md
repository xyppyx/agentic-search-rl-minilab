# Project TODO

本文件记录进行中任务、下一步、验收条件、阻塞项和未解决风险。完成且验证后移入 `PROJECT_COMPLETED.md`。

## 近期任务

- 初始化 `my-search-r1/` 改进实现骨架。
  - 验收条件：明确包结构、配置入口、工具 backend 接口、trajectory schema 和 smoke run 命令。
  - 当前状态：待开始。

- 从 `03-search-r1/` 基线迁移最小可运行 Search-R1 smoke/eval 链路。
  - 验收条件：在不依赖真实付费搜索服务的 mock backend 下，可生成 trajectory JSONL 和 Markdown 报告。
  - 当前状态：待开始。

- 实现搜索工具封装与 mock backend。
  - 验收条件：rollout 只依赖统一工具接口，可切换 Zhihu/mock backend，并记录工具调用指标。
  - 当前状态：待开始。

- 实现 trajectory JSONL 保存与基础报告。
  - 验收条件：训练或评测可输出 correct、wrong、invalid_format、tool_failure 等分类案例报告。
  - 当前状态：待开始。

## 未解决风险

- `my-search-r1/` 当前尚未放入可运行代码，需要先定义从基线复制、迁移或重写的边界。
- 真实知乎搜索 API、PyTRIO 远程训练和 SwanLab 记录依赖外部凭据与服务状态，后续实验需要 mock baseline 与真实 backend 指标分开记录。
- 当前已有 `03-search-r1/train.py` 本地改动，后续修改基线文件前需要继续保护这部分改动。
