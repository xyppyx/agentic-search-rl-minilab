# Search-R1 基线目录规则

- 本目录保留上游 PyTRIO + 知乎搜索 Search-R1 教学复现，是 `my-search-r1/` 改进实现的基线参考。
- 除非任务明确要求修改基线，优先在 `my-search-r1/` 新增或重构改进能力。
- 修改本目录训练、rollout、reward 或 eval 行为时，必须记录与基线行为的差异，并同步更新状态文件。
- 不提交 `.env`、真实 API key、远程训练凭据、SwanLab 私有链接、数据下载缓存、eval 输出、checkpoint 或模型权重。
