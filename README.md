# LLM Agent RL Lab 改进项目

本仓库基于原作者 `KMnO4-zx/llm-agent-rl-lab` 的教学复现代码，用于学习和改造 LLM/Agent 强化学习算法。当前个人主线是 Search-R1 改进项目：

```text
Robust Search-R1 MiniLab:
低成本、多工具、不可靠环境下的搜索型 Agent RL 实验框架
```

设计草案见 [docs/design/idea.md](docs/design/idea.md)。后续正式改进实现默认放在 [my-search-r1/](my-search-r1/)；上游 Search-R1 教学复现保留在 [03-search-r1/](03-search-r1/) 作为基线参考。

## 项目边界

- `00-loss-function/`、`01-grpo/`、`02-opd/`、`03-search-r1/`：原教学复现内容，用于理解算法和对照基线。
- `my-search-r1/`：本项目改进实现区，优先承载工具封装、trajectory 记录、鲁棒工具环境、reward 组件化和后续训练后端封装。
- `docs/`：设计、状态事实源、学习记录和实验复盘。

本项目是个人学习型 POC，不代表 PyTRIO、SwanLab、知乎开放平台、Search-R1 官方实现或原作者参与、委托或认可。

## 当前路线

近期最小版本目标：

```text
让 Search-R1 smoke/5-step run 产生可复查的 trajectory JSONL 和报告。
```

优先级来自 [docs/design/idea.md](docs/design/idea.md)：

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

运行现有上游 Search-R1 基线前，需要进入 `03-search-r1/` 并按该目录说明配置 `.env`。真实 API key、远程训练凭据、SwanLab 私有链接、模型权重和 checkpoint 不得提交。

```bash
cd 03-search-r1
uv run python prepare_data.py
```

`my-search-r1/` 目前用于承接改进实现；新增脚本应优先提供 smoke run 参数和可复现的输出路径。

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
