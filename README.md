# LLM Agent RL Lab 改进项目

本仓库基于原作者 `KMnO4-zx/llm-agent-rl-lab` 的教学复现代码，用于学习和改造 LLM/Agent 强化学习算法。当前个人主线是 Search-R1 改进项目：

```text
Robust Search-R1 MiniLab:
低成本、多工具、不可靠环境下的搜索型 Agent RL 实验框架
```

设计草案见 [docs/design/idea.md](docs/design/idea.md)，reward 设计版本与下一轮实验计划见 [docs/design/reward_shaping_plan.md](docs/design/reward_shaping_plan.md)。正式改进实现放在 [my-search-r1/](my-search-r1/)；原教学复现目录已归档到 `Backup` 分支，`main` 只保留当前项目主线。

## 项目边界

- `my-search-r1/`：本项目改进实现区，优先承载工具封装、trajectory 记录、鲁棒工具环境、reward 组件化和后续训练后端封装。
- `docs/`：设计、状态事实源、学习记录和实验复盘。
- `Backup` 分支：归档 `00-loss-function/`、`01-grpo/`、`02-opd/`、`03-search-r1/` 四个原教学目录，用于历史参考。

本项目是个人学习型 POC，不代表 PyTRIO、SwanLab、知乎开放平台、Search-R1 官方实现或原作者参与、委托或认可。

## 当前路线

近期最小版本目标：

```text
让 Search-R1 smoke/5-step run 产生可复查的 trajectory JSONL 和报告。
```

优先级来自 [docs/design/idea.md](docs/design/idea.md) 和 [docs/design/reward_shaping_plan.md](docs/design/reward_shaping_plan.md)：

1. 搜索工具封装 + 轨迹可视化。
2. reward / penalty 组件化。
3. 概率失败工具环境。
4. PyTRIO 后端薄封装。
5. 阶段化训练。

## 快速启动

依赖使用 `uv` 管理：

```bash
uv sync
```

`my-search-r1/` 是当前主分支的正式实现区。新增脚本应优先提供 smoke run 参数和可复现的输出路径。需要查看或恢复原教学目录时，切换到 `Backup` 分支。

## 协作规范

- 仓库规则见 [AGENTS.md](AGENTS.md)。
- 项目状态事实源在 [docs/status/](docs/status/)。
- 开始任务前读取适用的 `AGENTS.md`、`PROJECT_COMPLETED.md` 和 `PROJECT_TODO.md`。
- 完成可独立验证的产物、实验结果或关键决策后同步更新状态文件。

## 上游来源

原项目：

- GitHub: <https://github.com/KMnO4-zx/llm-agent-rl-lab>
- Search-R1 论文: <https://arxiv.org/abs/2503.09516>
- Search-R1 官方实现: <https://github.com/PeterGriffinJin/Search-R1>

## License

See [LICENSE](LICENSE).
