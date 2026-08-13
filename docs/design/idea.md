# Robust Search-R1 MiniLab Design

本文是公开版系统设计概览，包括上层目标、模块边界和路线判断。

## 项目定位

Robust Search-R1 MiniLab 是一个面向搜索型 LLM Agent 的小预算 Agentic RL 实验框架。当前更具体的应用导向是多跳搜索场景：模型需要先找到中间线索，再基于 observation 继续 follow-up，最后输出短答案。项目关注的不是单次分数，而是把以下链路做成可训练、可评测、可诊断的工程闭环：

```text
question -> search action -> tool observation -> follow-up search -> final answer
```

当前公开路线可以概括为：

```text
turn-level search credit
  -> gated self-distillation refinement
```

核心判断：搜索型 Agent 的提升不能只看最终答案正确率；还要同时观察搜索行为、工具可靠性、格式收束、follow-up 是否充分，以及不同路线的取舍。

路线边界：早期曾设想把“不可靠搜索工具/故障注入鲁棒训练”作为主问题，但后续正式训练和评测基本要求工具成功率达标，以便隔离外部 API 波动。因此该 idea 已暂时搁置；failure injection 保留为工具层 smoke、回归测试和评测边界验证能力，当前主线不把“适应失败工具”作为算法目标。

## 系统分层

| 层         | 作用                                                              |
| ---------- | ----------------------------------------------------------------- |
| 数据层     | 准备 Search-R1 风格问答样本和 targeted eval 集                    |
| 工具层     | 抽象搜索 backend，隔离真实搜索 API 与可复现 mock/local 环境       |
| Rollout 层 | 执行模型-工具交互并保存完整 trajectory                            |
| 训练层     | 支持 group rollout、policy optimization、reference 约束和辅助目标 |
| Reward 层  | 从最终答案 reward 演进到 turn-level credit                        |
| 评测层     | 同时报告答案、格式、搜索效率和工具失败                            |
| 诊断层     | 支持离线失败分类、gained/lost 对比和 route decision               |

## 设计原则

1. 先可观测，再优化。
   Agentic RL 的平均 reward 不足以解释行为变化，必须保存 trajectory 并区分答案错误、格式错误、工具失败和搜索策略失败。
2. 工具失败不等于模型失败。
   真实搜索服务会出现超时、空结果、限流或解析失败；当前做法是把这些外部问题从正式模型效果中剥离，而不是训练模型专门适应故障工具。
3. 不把“少搜索”当作直接目标。
   过强搜索惩罚可能减少无效搜索，也可能压掉必要 follow-up。更可靠的方向是奖励关键搜索动作，并约束早答和搜满不答。
4. 辅助蒸馏必须 gated。
   搜索轨迹中包含错误答案、工具 observation 和无效 query；自蒸馏不能全序列套用，只能在被筛选的正向 action tokens 上作为辅助信号。
5. 最终路线看综合指标。
   更长训练或更高单项均值不一定更好；最终选择需要同时看 correctness、format、search cost 和 tool reliability。

## 数据与评测

公开 README 介绍当前主线需要的 train/dev/test，以及 bridge150、bridge_eval_350 等多跳 targeted eval。评测集构造的公开说明见 [evaluation_design.md](evaluation_design.md)。

数据目录被 `.gitignore` 忽略，公开仓库不提交完整数据文件、真实评测 JSONL、模型权重或远程 checkpoint。

## 路线概览

项目路线已经从早期 penalty 调参收敛为两段：

1. Turn-level search credit：解决多跳搜索中的 credit assignment，让训练信号更接近“哪一次搜索有用”。
2. Gated self-distillation refinement：在已有搜索策略基础上做保守辅助优化，避免把错误轨迹或工具 observation 蒸馏进模型。

最新 `bridge_eval_350` 三方对照显示，base+prompt、GRPO guard-fix 20-step、GRPO+OPSD v2 5-step 在同一多跳评测集上呈递进改善。这支持当前项目导向：训练目标不是让模型盲目多搜，而是改善中间线索搜索、follow-up 和停止策略。

公开文档只保留这一级设计。具体 reward 权重、mask 策略、训练参数和多轮实验表放在本地私有材料中。

## 后续方向

- 继续强化 trajectory-level case review，而不是只扩大训练步数。
- 重点检查 bridge/final-hop 场景下的格式收束和搜满不答问题。
- 若重启自蒸馏，只考虑更可靠的 teacher 或 filtered replay，不重复无筛选的 same-context 蒸馏。
