# Project TODO

本文件记录进行中任务、下一步、验收条件、阻塞项和未解决风险。完成且验证后移入 `PROJECT_COMPLETED.md`。

## 近期任务

- 设计下一轮 reward shaping 训练策略。
  - 验收条件：基于 2026-07-21 的 5-step 与 20-step checkpoint 对比结果，提出更保守的 penalty 参数、延后启用策略或新增格式/证据阅读 reward，并明确下一轮训练的对照配置、验收指标和停止条件。
  - 当前状态：2026-07-21 已完成 base、5-step 原始 reward、5-step penalty reward、20-step 原始 reward 的 Zhihu dev 70 条对比，以及 5-step/20-step gained-lost case review；5-step 原始 reward EM 最好，20-step 原始 reward format 最好且搜索次数更低，但 EM 低于 5-step。20-step lost 4 条中有 2 条是真实多跳/实体绑定退化，2 条主要是严格 EM/答案粒度问题；下一轮应优先区分必要 follow-up query 与无收益重复搜索，并补 answer granularity 诊断。

## 未解决风险

- `my-search-r1/` 当前已有搜索工具层、trajectory JSONL、报告能力、训练级 rollout、PyTRIO train/eval CLI、完整数据集、一次非退化 1-step GRPO 更新证据、一次 5-step reward shaping checkpoint 对照，以及一次原始 reward 20-step checkpoint 对照；尚未做多 seed 稳定性重复实验或 20-step 以上训练。
- 真实知乎搜索 API、PyTRIO 远程训练和 SwanLab 记录依赖外部凭据与服务状态，后续实验需要 mock baseline 与真实 backend 指标分开记录。
- 当前 dev 失败复盘显示 local BM25 只适合 smoke/mock；完整 dev 上空结果率 56.92%，不宜用它代表真实搜索能力。Zhihu dev 主要失败不在工具异常，而在格式收束、query 改写、证据阅读和严格 EM 对冗长答案/日期格式的误伤。2026-07-21 的 5-step 对比进一步显示，轻量 penalty 能减少过度搜索但可能损伤答对率，需要更谨慎的权重或训练日程。
- 当前已有 `03-search-r1/train.py` 本地改动，后续修改基线文件前需要继续保护这部分改动。
