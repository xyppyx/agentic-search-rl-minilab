# 一页纸项目讲稿

本文用于面试开场 2 到 3 分钟介绍项目。建议先背“精简版”，被追问时再展开指标和边界。

## 30 秒精简版

我做的是 Robust Search-R1 MiniLab，一个面向搜索型 Agentic RL 的个人 POC。项目基于 Qwen3.5-4B、PyTRIO GRPO 和真实 Zhihu Search backend，目标不是单纯跑通 Search-R1，而是把搜索型 Agent 的训练和评测做得可观测、可诊断、可复盘。

我实现了统一搜索工具层、trajectory JSONL、Markdown report、offline diagnostics、reward sensitivity 和 turn-level credit。算法上，我从简单行为 penalty 迭代到 final-hop turn-level reward shaping，针对多跳问题里的 bridge search、final-hop attribute search、early answer 和 max-search no-answer 做定向信号。最终在 dev70 和 bridge targeted eval 上取得 EM/correct 正向证据，同时也如实保留 patched protocol 和 format 仍待提升的边界。

## 2 分钟完整版

这个项目的背景是：Search-R1 类方法把搜索工具接入 LLM，让模型在回答问题时可以主动 search、读取 observation、继续 follow-up 或输出最终答案。但真实环境下有两个问题。第一，搜索工具不稳定，会出现空结果、限流、解析错误和超时。第二，多跳问题的关键不只是最终答对，而是中间每一步该搜什么、是否需要 follow-up、什么时候停止。只看最终 EM 很难解释模型到底哪里变好。

所以我把项目目标定成 Robust Search-R1 MiniLab：在低成本真实搜索环境下，构建一套可观测的 Agentic RL 实验框架。工程上，我做了四块：

- 第一是工具层，统一 mock、local BM25、Zhihu Search 和 failure injection，让 rollout 不直接依赖某一个搜索 API。
- 第二是轨迹层，保存完整 trajectory JSONL，记录每轮 query、observation、final answer、reward、format、search calls 和 tool failures。
- 第三是训练层，迁移 PyTRIO GRPO train/eval 链路，并加入 advantage standardization、ratio clip 和 KL-style reference 约束，提升小预算训练稳定性。
- 第四是分析层，实现 offline diagnostics、reward sensitivity、turn-credit analysis 和 gained/lost review，用来解释 checkpoint 行为变化。

算法上，我的主要经验是：简单 penalty 不够稳。比如惩罚重复搜索、空结果或搜满不答，确实能减少无效搜索，但也可能让模型少做必要 follow-up，导致 EM 下降。后来我把方向改成 turn-level credit：奖励 evidence bridge search 和 final-hop attribute search，惩罚 early answer、missing final-hop attribute，以及搜过后仍然 max-search no-answer 或格式非法。

最终结果上，dev70 的 guard-fix 20-step retry 有效评测达到 EM 0.4571、correct 32/70、format 0.9571、平均搜索 1.9000，Zhihu success rate 1.0。bridge_eval_150 上，guard-fix 20-step patched 达到 EM 0.5142、correct 83/150、format 0.8267，相对 prompt-only base 的 EM 0.4750、correct 74/150、format 0.7200 有提升。

但这里我会明确说明边界：bridge 的最强结果是 patched protocol，由 147 条 full run 记录加 3 条工具失败样本 retry 合成，不等同一次独立全量 success rate 1.0 run。它说明在去除外部工具失败污染后，策略有正向证据；如果追求严格论文式结论，还需要等 Zhihu 稳定后重跑独立全量 bridge150。另外，guard-fix 20-step 的 bridge format 0.8267 仍低于 evidence-v2 20-step 的 0.9400，所以后续短板是 format/max-search no-answer。

这个项目我最想强调的不是某个单点指标，而是完整方法论：Agentic RL 不能只跑 reward 曲线，要把工具环境、trajectory、reward shaping、失败诊断和指标边界一起做清楚。

## 简历项目口述版

我搭建了一个 Search-R1 Agentic RL 实验框架，基于 Qwen3.5-4B、PyTRIO GRPO 和 Zhihu Search backend，复现并改造搜索型 LLM Agent 的训练评测链路。工程上实现了统一搜索工具层、trajectory JSONL、offline diagnostics、reward sensitivity 和 turn-credit analysis。方法上针对多跳搜索中的 follow-up 缺失和 final-hop 早答问题，设计 evidence bridge / final-hop attribute turn-level reward shaping。最终在 dev70 上达到 EM 45.7%；在 bridge targeted eval 的 patched protocol 下，EM 从 prompt-only base 的 47.5% 提升到 51.4%，correct 从 74/150 提升到 83/150，同时定位 format/max-search no-answer 为下一阶段主要瓶颈。

## 指标边界口径

面试中不要只说“我把 EM 提到 51.4%”，要补一句边界：

```text
这个 51.4% 是 bridge_eval_150 的 patched protocol 结果，由 147 条 full run 记录加 3 条失败样本 retry 合成；它是当前最高 bridge EM/correct 证据，但不是一次独立全量 success rate 1.0 run。
```

如果想保守一点：

```text
dev70 上有一次独立有效 retry，Zhihu success rate 1.0，EM 45.7%；bridge targeted eval 上 patched 结果显示 EM/correct 正向，但我在文档里明确标注它不等同严格全量 run。
```

## 高频追问速答

**为什么不用普通 RAG？**

普通 RAG 多数是固定检索再生成，检索不是模型策略的一部分；我的项目里模型在 rollout 中主动决定搜不搜、搜什么、是否 follow-up 和何时回答，训练信号作用在这些决策上。

**为什么不用 SFT？**

SFT 需要高质量搜索轨迹标注，成本高且容易学固定模式。这个项目更关注真实搜索环境下的策略行为，所以用 GRPO 和规则 reward 做小预算 RL，再通过 trajectory 和 diagnostics 解释行为。

**最大的失败经验是什么？**

简单惩罚搜索行为会带来副作用。它可能减少无效搜索，但也可能压掉必要二跳搜索，导致 EM 下降。所以后面我从“惩罚坏行为”转向“奖励必要搜索轮次”，也就是 turn-level credit。

**怎么保证不是工具 API 波动造成的？**

所有 trajectory 都记录 tool success rate、tool failures、empty result 和错误类型。success rate 低于 1.0 的 full run 不进入正式表；patched 协议单独标注，不冒充独立全量评测。

**下一步做什么？**

先重跑一次独立全量 bridge150，要求 Zhihu success rate 1.0；同时针对 format/max-search no-answer 做更细的 guard，把 guard-fix 的 EM/correct 优势和 evidence-v2 的高 format 结合起来。

## 最后 15 秒总结

我认为这个项目最有价值的地方是把搜索型 Agent RL 做成了可解释闭环：不只训练一个 checkpoint，而是能回答模型为什么搜索、为什么答错、工具有没有污染结果，以及某个 reward shaping 到底改善了什么、牺牲了什么。
