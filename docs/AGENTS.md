# Docs 目录规则

- 设计、状态、学习记录和实验复盘放在本目录。
- `docs/status/PROJECT_COMPLETED.md`、`docs/status/PROJECT_TODO.md`、`docs/status/PROJECT_LOG.md` 是项目状态唯一事实源，不新增平行状态文件。
- `docs/status/archive/` 只保存历史快照，不能当作当前进度、当前 TODO、当前基线或当前实验结论。
- status 文件只保存当前快照、活跃 TODO 和关键决策；完整实验过程、命令、详细指标、排查和 case review 写入 `docs/interview/lesson/`。
- 切换新方法主线/实验分支、阶段完成、`PROJECT_COMPLETED.md` 超过约 120 行、`PROJECT_TODO.md` 超过约 80 行、同一 active track 累计 5 个以上 run/retry，或 TODO 混入历史解释时，先压缩 status 再继续推进。
- 压缩前可归档到 `docs/status/archive/YYYY-MM-DD_reason/`，但 archive 必须声明仅供追溯；压缩后当前三份 `PROJECT_*.md` 仍是唯一事实源。
- 记录实验时只写可公开信息；不得写入真实密钥、私有账号、服务器凭据、SwanLab 私有链接或付费 API key。
- 未实际观测到的指标、费用、显存、耗时或评测结果不得补写或猜测。
