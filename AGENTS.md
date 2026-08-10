# 项目协作规范

## 规则分层

- 本文件适用于整个仓库；进入子目录后，还必须遵守距离目标文件最近的 `AGENTS.md`。
- `AGENTS.md` 只写必要规则，模块背景、结构和使用说明写入同级 `README.md`。
- 一级子目录应维护精简的 `AGENTS.md` 和说明模块信息的 `README.md`；嵌套目录仅在存在独立规则时增设。
- 子目录规则不得重复整份根规则，只补充或收紧本模块约束。

## 项目背景

本项目基于原作者 `KMnO4-zx/llm-agent-rl-lab` 的教学复现仓库，围绕 Search-R1、GRPO、OPD 等 LLM/Agent 强化学习方法做学习、复现和改造。当前个人项目主线是 `docs/design/idea.md` 中的 Robust Search-R1 MiniLab：在低成本、多工具、不可靠工具环境下，构建可控、可观测、可扩展的搜索型 Agentic RL 实验框架。

上游教学复现目录 `00-loss-function/`、`01-grpo/`、`02-opd/`、`03-search-r1/` 已归档到 `Backup` 分支；`main` 只保留本项目自有实现、设计和复盘材料。改进实现默认放在 `my-search-r1/`，设计、复盘和长期记录放在 `docs/`。除非任务明确要求恢复或对照历史基线，不把备份分支中的既有教学能力包装为本项目新增成果。

本项目是个人学习型 POC，不代表 PyTRIO、SwanLab、知乎开放平台、Search-R1 官方实现或原作者参与、委托或认可。在线搜索 API、模型服务、实验日志和远程训练凭据都按敏感运行资源处理。

## 上下文维护

以下三个文件是项目状态的唯一事实源：

- `docs/status/PROJECT_COMPLETED.md`：已完成且有验证证据的事实、产物、结果和最终决策。
- `docs/status/PROJECT_TODO.md`：进行中任务、下一步、验收条件、阻塞项和未解决风险。
- `docs/status/PROJECT_LOG.md`：需要长期追溯的重要事件、方向变化、问题解决和阶段复盘。

`docs/status/archive/` 仅保存历史快照，供追溯压缩前状态使用；其中内容不得作为当前进度、当前 TODO、当前基线或当前实验结论。需要旧实验细节时优先读取 `docs/interview/lesson/` 的复盘文档，再按需查看 archive。

开始任务前读取适用的 `AGENTS.md`、`PROJECT_COMPLETED.md` 和 `PROJECT_TODO.md`；仅在需要追溯时读取 `PROJECT_LOG.md`，并按需读取相关 `README.md`。不要仅依赖对话历史。

完成可独立验证的产物、取得实验或评测结果、作出关键决策、发现阻塞或准备实质性提交时更新状态文件。代码写完但未验证不能记为完成；完成项需注明时间、相关路径、验证方式和关键结果。TODO 完成后移入 COMPLETED，不创建 `TODO.md`、`STATUS.md` 等平行状态文件。

状态文件采用“当前快照 + 活跃 TODO + 决策日志”写法，不写成长实验流水账。`PROJECT_COMPLETED.md` 写当前可引用结论、基线表、已验证能力和历史阶段索引；`PROJECT_TODO.md` 只写活跃任务、验收条件、停止条件和未解决风险；`PROJECT_LOG.md` 只写长期决策和方向变化。训练、评测、排查过程、完整命令、详细指标和 case review 默认写入 `docs/interview/lesson/`，status 只链接或概括其结论。

满足任一条件时，先压缩 status，再进入下一阶段实质实验或提交准备：

- 切换到新的方法主线、实验分支或 active track 前。
- 一个实验阶段完成，已经形成明确 baseline、最优 checkpoint、失败结论和下一步时。
- `PROJECT_COMPLETED.md` 超过约 120 行，或 `PROJECT_TODO.md` 超过约 80 行时。
- 同一 active track 下累计 5 个以上训练、评测、重跑或 health/retry 记录时。
- TODO 中已完成、暂停、当前任务和历史解释开始混杂时。

压缩 status 时可先在 `docs/status/archive/YYYY-MM-DD_reason/` 保存当前三份 `PROJECT_*.md` 快照，并在 archive README 中声明其仅供历史追溯。压缩后当前三份 `PROJECT_*.md` 必须仍是唯一事实源，且不能丢失当前 baseline、验收条件、停止条件和公开边界。

## 目录索引

- `my-search-r1/`：本项目 Search-R1 MiniLab 的正式改进实现目录。
- `docs/`：设计文档、学习记录、状态事实源和实验复盘。
- `images/`：根 README 使用的公开图片资产。
- `.env`：本地及远程运行配置和敏感变量；`.env.example` 只保留可公开的键名与示例。
- `Backup` 分支：归档上游教学复现目录 `00-loss-function/`、`01-grpo/`、`02-opd/`、`03-search-r1/`，用于历史参考和必要时恢复。

更详细的模块信息见根目录及各一级目录的 `README.md`。

## 开发约定

- 在本机编写和检查代码；需要 GPU、远程采样或 PyTRIO 训练时，通过明确的脚本、配置和 Git 同步复现。
- 当前仓库按公开仓库处理。所有提交、文档和 manifest 默认公开可读；不得提交密钥、令牌、服务器凭据、付费 API key、未授权数据全文、大体积训练数据、模型权重、LoRA checkpoint、SwanLab 私有链接或包含敏感路径/账号的日志。
- Python 项目统一使用 `uv` 管理依赖与虚拟环境，优先使用 `uv sync`、`uv add` 和 `uv run`；例外必须记录原因。
- 运行配置写入 `.env`，可公开的配置模板同步维护到 `.env.example`；不得提交真实密钥、令牌、账号、服务器地址或计费项凭据。
- 项目新增实现、配置、数据清单和实验产物必须放在项目自有目录中，默认优先放入 `my-search-r1/`、`docs/` 或后续明确的新目录。
- 修改前检查工作区已有内容，保留用户改动；验证强度与变更风险相匹配。

## Agentic RL 实验约定

- 每次运行训练、评测、推理、rollout 或搜索工具实验后，必须在 `docs/interview/` 或 `docs/status/` 记录可复盘材料，包含实验目标、数据与模型版本、关键参数、loss/reward/评测指标、工具调用统计、失败类型、耗时与费用、遇到的问题、排查过程和可复用经验；未观测到的指标不得补写或猜测。
- 实验链路必须可复现：脚本、配置、命令、输入输出路径、环境变量名、随机种子、模型 revision、数据 revision、搜索 backend 和 failure 注入配置应清晰落盘；实验参数必须可通过配置文件或命令行覆盖，便于 smoke run、缩放步数或做对照实验。
- Search-R1 改进必须优先保证可观测性：新增工具封装、reward、penalty 或训练逻辑时，同步考虑 trajectory JSONL、指标命名、失败案例抽样和报告输出。
- 在线搜索环境必须与模型策略问题区分记录。训练和评测指标中应尽量分开记录工具失败、空结果、超时、重复 query、格式错误、最终答案正确性和 token/费用消耗。
- GPU 或远程训练实验完成后，同步检查 `PROJECT_COMPLETED.md` 与 `PROJECT_TODO.md`；只有已验证的结果才能写入完成记录，未跑通或待补充的实验继续保留在 TODO 或复盘中。

## Git 约定

- 当且仅当用户明确要求整理 Git，或确实需要同步到远程环境运行实验时，才执行 `git add`、`commit`、`push` 等写操作；普通本地修改不自动提交。
- push 前必须复查 `git status`、`git diff --cached` 和敏感信息/大文件边界，确保只发布适合公开展示的代码、配置模板、统计、hash、manifest 和文档。
- 操作前确认目标 Git 仓库并检查 `status` 与 `diff`，不混入无关改动，不提交 `.env`、模型权重、checkpoint、未批准的大文件、私有日志或参考仓库内容。
- 提交信息必须使用中文，推荐格式为 `模块：中文说明`，例如 `工具：增加搜索后端注册表`。
- 未经用户明确授权，不改写历史、不强推、不执行破坏性清理。

## 每次运行后的固定汇报

每次完成代码、数据处理、文档、实验配置、评测或生成脚本运行后，最终回复必须明确包含：

- 目前进展是什么。
- 接下来做什么。
- 距离 LLM 应用算法面试项目还差什么。

同时简要说明实际运行的验证及结果；未运行的训练、测试或评测不得暗示已经完成。
