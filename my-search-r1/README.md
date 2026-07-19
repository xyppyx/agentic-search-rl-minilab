# my-search-r1

本目录用于实现 Robust Search-R1 MiniLab，是当前项目区别于上游教学复现的主要工作区。

近期目标：

```text
让 Search-R1 smoke/5-step run 产生可复查的 trajectory JSONL 和报告。
```

计划模块：

- `tools/`：统一搜索工具接口、backend registry、Zhihu backend、mock backend 和 failure wrapper。
- `trajectories/`：trajectory schema、JSONL 序列化和报告生成。
- `rewards/`：正确性、格式、工具调用与重复 query 等 reward/penalty 组件。
- `backend/`：PyTRIO 训练后端薄封装。
- `configs/`：可公开的 smoke/eval/train 配置模板。
- `scripts/`：数据准备、rollout、训练、评测和报告命令入口。

实现前以 [../03-search-r1/](../03-search-r1/) 为基线参考，以 [../docs/design/idea.md](../docs/design/idea.md) 为路线依据。真实凭据写入本地 `.env`，公开模板写入 `.env.example`。
