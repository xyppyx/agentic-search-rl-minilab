# Robust Search-R1 MiniLab 项目经历 STAR(T) 版

## Situation

Search-R1 类搜索型 Agent 在真实工具环境下不仅要答对问题，还要稳定地调用搜索工具、处理空结果和外部 API 错误，并在多跳问题中避免过早回答。上游教学仓库已经提供 Search-R1、GRPO、OPD 等复现材料，但缺少一个低成本、可观测、可复盘的 Agentic RL 实验框架。

我基于 `KMnO4-zx/llm-agent-rl-lab` 搭建个人学习型 POC，将主线定义为 Robust Search-R1 MiniLab：围绕 Qwen3.5-4B、PyTRIO GRPO 和 Zhihu Search backend，研究如何在不可靠搜索工具和小预算训练下改进搜索型 LLM Agent。

## Task

项目目标不是单纯跑通训练，而是形成一套可以面向算法面试讲清楚的闭环：

- 构建统一搜索工具层，支持 mock、local BM25、Zhihu backend 和 failure injection。
- 保存完整 trajectory JSONL，记录每轮 search query、tool observation、final answer、reward、format 和工具失败。
- 设计可解释 reward shaping，缓解重复搜索、空结果、过早回答、缺少 follow-up query、final-hop 属性未查和 max-search 不答等问题。
- 建立 dev70、`bridge_eval_150`、`alias_granularity_eval_80` 等评测集，分开记录模型策略问题和外部搜索工具问题。
- 通过 gained/lost case review 和 offline diagnostics 解释指标变化，而不是只报告单个 EM。

## Action

我先迁移并重构 Search-R1 最小链路，形成 `my-search-r1/` 的正式实现区：

- 实现 `ToolRegistry`、`MockSearchBackend`、`LocalBM25Backend`、`ZhihuSearchBackend` 和 `FailureWrapperBackend`。
- 实现训练级 rollout、PyTRIO GRPO train/eval CLI、trajectory JSONL、Markdown report、checkpoint 分析脚本。
- 将训练默认稳定化为 standardized advantage、advantage clip 2.0、KL-style reference drift penalty、ratio clip 0.2 和 learning rate 1e-5。

随后围绕 reward shaping 做多轮实验：

- 从 duplicate/empty/max-search penalty 入手，发现简单 penalty 会降低过度搜索但容易损伤 EM。
- 加入 prompt/rollout 约束，要求多跳题先识别 bridge entity，再做 follow-up search，最终输出短答案。
- 设计 offline diagnostics，标注 `possible_alias_match`、`answer_granularity_miss`、`missing_followup_query`、`bad_max_search_loop` 等失败类型。
- 设计 turn-level credit：对有用搜索轮次赋予训练信号，而不是只在最终答案上给 reward。
- 最终形成 `final_hop_bridge` / guard-fix 策略：奖励 evidence bridge search 和 final-hop attribute search，惩罚 early answer、missing final-hop attribute 和 final-answer/max-search no-answer。

当前主 checkpoint 是 `turn-credit-final-hop-guardfix-20step-20260806`，核心配置为：

| 组件 | 值 |
| --- | ---: |
| evidence search turn bonus | 0.05 |
| final-hop search turn bonus | 0.10 |
| early-answer turn penalty | 0.05 |
| missing-final-hop turn penalty | 0.08 |
| final-answer guard turn penalty | 0.06 |
| max steps | 20 |
| questions per batch | 2 |
| group size | 4 |

## Result

基础能力已经完成并验证：

- 工具层、trajectory 报告、rollout、train/eval CLI、offline diagnostics、reward sensitivity、turn-credit analysis 均已实现。
- 训练阶段可生成可复盘 JSONL，并能区分模型错误、format 错误、搜索空结果、工具失败和重复 query。
- `turn-credit-final-hop-guardfix-20step-20260806` 训练 20/20 step 完成，160 条训练 trajectory，训练阶段 tool failures 为 0。

关键指标如下：

| Eval | Baseline | 当前最好结果 | 结论 |
| --- | ---: | ---: | --- |
| dev70 | evidence-v2 20-step EM 0.4429 | guard-fix 20-step retry EM 0.4571，format 0.9571 | EM 略优，但 format 低于 evidence-v2 的 1.0000 |
| `bridge_eval_150` | prompt-only base EM 0.4750，74/150，format 0.7200 | guard-fix 20-step patched EM 0.5142，83/150，format 0.8267 | EM/correct/format 均优于 prompt base |
| `bridge_eval_150` | evidence-v2 20-step EM 0.4583，81/150，format 0.9400 | guard-fix 20-step patched EM 0.5142，83/150，format 0.8267 | EM/correct 更高，但 format 更低 |
| `alias_granularity_eval_80` | prompt-only base EM 0.4500，36/80，format 0.9250 | evidence-v2 20-step EM 0.4375，35/80，format 0.9625 | format 提升，但 EM/correct 未超过 base；guard-fix 20-step 尚未跑该集 |

需要如实说明：`bridge_eval_150` 的 guard-fix 20-step 最强结果采用 patched 协议，由 147 条 full run 记录和 3 条失败样本 retry 记录合成。它可作为当前最高 bridge EM/correct 证据，但不等同于一次独立全量 success rate 1.0 run。

## Takeaway

这个项目最有价值的部分不是单个 checkpoint，而是形成了一个可解释的 Agentic RL 实验方法：

- 搜索型 Agent 的提升需要同时看 answer correctness、format、search efficiency 和 tool reliability。
- 只做最终 reward 容易把必要 follow-up 搜索压掉；turn-level credit 能更直接地引导“该搜什么、何时停止”。
- 外部工具不稳定会污染评测，必须把 tool failure 与模型策略错误分开记录。
- 对面试表达而言，最稳妥的结论是：我构建了可观测 Search-R1 MiniLab，提出 final-hop turn-level reward shaping，在 dev70 和 bridge targeted eval 上取得 EM/correct 正向证据，同时定位到 format/max-search no-answer 仍是下一阶段瓶颈。

## 简历表述

可放入简历的一版：

> 构建 Robust Search-R1 MiniLab，基于 Qwen3.5-4B、PyTRIO GRPO 与 Zhihu Search backend 复现并改造搜索型 Agent RL 链路；实现统一搜索工具层、trajectory JSONL、offline diagnostics、reward sensitivity 与 turn-credit analysis。设计 evidence bridge / final-hop attribute turn-level reward shaping，在 dev70 上将 EM 提升至 45.7%；在 bridge targeted eval 上将 EM 从 prompt-only base 的 47.5% 提升到 51.4%（patched protocol），correct 从 74/150 提升到 83/150，并定位 format/max-search no-answer 为主要瓶颈。

更保守的一版：

> 搭建 Search-R1 Agentic RL 实验框架，围绕真实搜索工具不稳定、多跳 follow-up 和格式收束问题设计可观测训练与评测链路；通过 turn-level final-hop reward shaping 在 dev70 与 bridge targeted eval 上取得 EM/correct 正向证据，并完成 gained/lost case review 与失败类型诊断。
