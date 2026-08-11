# 一页纸项目讲稿

本文用于面试开场 2 到 3 分钟介绍项目。建议先背“精简版”，被追问时再展开指标和边界。

## 30 秒精简版

我做的是 Robust Search-R1 MiniLab，一个面向搜索型 Agentic RL 的个人 POC。项目基于 Qwen3.5-4B、PyTRIO GRPO 和真实 Zhihu Search backend，目标不是单纯跑通 Search-R1，而是把搜索型 Agent 的训练和评测做得可观测、可诊断、可复盘。

我实现了统一搜索工具层、trajectory JSONL、Markdown report、offline diagnostics、reward sensitivity、turn-level credit 和 gated OPSD v2。算法上，我从简单行为 penalty 迭代到 final-hop turn-level reward shaping，再在最强 guard-fix checkpoint 上做 5-step OPSD v2 保守微调。最终路线固定为 guard-fix 20step + OPSD v2 5step：dev70 clean EM 0.4857、correct 34/70；bridge150 clean EM 0.5242、correct 87/150、format 0.9067。

## 2 分钟完整版

这个项目的背景是：Search-R1 类方法把搜索工具接入 LLM，让模型在回答问题时可以主动 search、读取 observation、继续 follow-up 或输出最终答案。但真实环境下有两个问题。第一，搜索工具不稳定，会出现空结果、限流、解析错误和超时。第二，多跳问题的关键不只是最终答对，而是中间每一步该搜什么、是否需要 follow-up、什么时候停止。只看最终 EM 很难解释模型到底哪里变好。

所以我把项目目标定成 Robust Search-R1 MiniLab：在低成本真实搜索环境下，构建一套可观测的 Agentic RL 实验框架。工程上，我做了四块：

- 第一是工具层，统一 mock、local BM25、Zhihu Search 和 failure injection，让 rollout 不直接依赖某一个搜索 API。
- 第二是轨迹层，保存完整 trajectory JSONL，记录每轮 query、observation、final answer、reward、format、search calls 和 tool failures。
- 第三是训练层，迁移 PyTRIO GRPO train/eval 链路，并加入 advantage standardization、ratio clip 和 KL-style reference 约束，提升小预算训练稳定性。
- 第四是分析层，实现 offline diagnostics、reward sensitivity、turn-credit analysis 和 gained/lost review，用来解释 checkpoint 行为变化。

算法上，我的主要经验是：简单 penalty 不够稳。比如惩罚重复搜索、空结果或搜满不答，确实能减少无效搜索，但也可能让模型少做必要 follow-up，导致 EM 下降。后来我把方向改成 turn-level credit：奖励 evidence bridge search 和 final-hop attribute search，惩罚 early answer、missing final-hop attribute，以及搜过后仍然 max-search no-answer 或格式非法。最后我尝试 OPD/OPSD 思路，但没有做 naive 全序列自蒸馏，而是做 gated OPSD v2：只在正向 advantage 且被 turn-credit 命中的 assistant turn 上启用辅助 distillation loss。

最终路线是两阶段：第一阶段 `turn-credit-final-hop-guardfix-20step-20260806` 学到更好的 bridge/final-hop 搜索策略；第二阶段 `guardfix20-resume-opsd-v2-5step-20260811` 从它的 final state 恢复，用 guardfix final weights 作为 KL/reference 和 OPSD teacher 做 5-step 保守微调。最终 dev70 clean EM 0.4857、correct 34/70、format 0.9857、平均搜索 1.7286；bridge150 10 个 clean chunks 合并后 EM 0.5242、correct 87/150、format 0.9067、平均搜索 3.1400。

我还做了 20step OPSD v2 的方差对照。20step seed43 在 bridge150 的 EM macro 是 0.5317，看似更高，但 correct 只有 81/150、format 0.8133、平均搜索 3.2533，综合弱于 5step 的 87/150、format 0.9067。因此最终选择 5step 不是因为省成本，而是因为更多步数没有带来更可靠的综合收益。

这个项目我最想强调的不是某个单点指标，而是完整方法论：Agentic RL 不能只跑 reward 曲线，要把工具环境、trajectory、reward shaping、失败诊断和指标边界一起做清楚。

## 简历项目口述版

我搭建了一个 Search-R1 Agentic RL 实验框架，基于 Qwen3.5-4B、PyTRIO GRPO 和 Zhihu Search backend，复现并改造搜索型 LLM Agent 的训练评测链路。工程上实现了统一搜索工具层、trajectory JSONL、offline diagnostics、reward sensitivity 和 turn-credit analysis。方法上针对多跳搜索中的 follow-up 缺失和 final-hop 早答问题，设计 evidence bridge / final-hop attribute turn-level reward shaping，并在最强 guard-fix checkpoint 上加入 gated OPSD v2。最终路线为 guard-fix 20step + OPSD v2 5step：dev70 clean EM 48.6%、correct 34/70；bridge150 clean EM 52.4%、correct 87/150、format 90.7%。

## 指标边界口径

面试中不要只说“我把 EM 提到 52.4%”，要补一句边界：

```text
这个 52.4% 是 guard-fix 20step + OPSD v2 5step 在 bridge_eval_150 上的 clean chunk 合并结果，10 个 chunk 共 150 条 trajectory，tool failures 为 0。它比此前 guard-fix 20step patched 的 51.4% 更适合作为正式项目口径。
```

如果想保守一点：

```text
dev70 上 clean EM 48.6%，bridge150 上 clean EM 52.4%；但 bridge format 90.7% 仍低于 evidence-v2 20step 的 94.0%，所以我把下一步放在 alias80 和 format/max-search case review，而不是继续堆训练步数。
```

## 高频追问速答

**为什么不用普通 RAG？**

普通 RAG 多数是固定检索再生成，检索不是模型策略的一部分；我的项目里模型在 rollout 中主动决定搜不搜、搜什么、是否 follow-up 和何时回答，训练信号作用在这些决策上。

**为什么不用 SFT？**

SFT 需要高质量搜索轨迹标注，成本高且容易学固定模式。这个项目更关注真实搜索环境下的策略行为，所以用 GRPO 和规则 reward 做小预算 RL，再通过 trajectory 和 diagnostics 解释行为。

**最大的失败经验是什么？**

简单惩罚搜索行为会带来副作用。它可能减少无效搜索，但也可能压掉必要二跳搜索，导致 EM 下降。所以后面我从“惩罚坏行为”转向“奖励必要搜索轮次”，也就是 turn-level credit。

**为什么 5step 可以作为最终选择？**

这里的 5step 不是从 base 重新学搜索能力，而是在 guard-fix 20step 强 checkpoint 上做保守 refinement。20step seed43 对照显示，更多 OPSD 步数虽然让 bridge EM macro 略高，但 correct、format 和搜索成本综合更差，所以选择 5step 是实证选择，不是省略实验。

**怎么保证不是工具 API 波动造成的？**

所有 trajectory 都记录 tool success rate、tool failures、empty result 和错误类型。success rate 低于 1.0 的 full run 不进入正式表；patched 协议单独标注，不冒充独立全量评测。

**下一步做什么？**

主路线已固定，下一步补 `alias_granularity_eval_80`，验证最终 checkpoint 是否牺牲 alias/granularity；同时做 gained/lost case review，把 5step 的提升和 format 剩余问题讲清楚。

## 最后 15 秒总结

我认为这个项目最有价值的地方是把搜索型 Agent RL 做成了可解释闭环：不只训练一个 checkpoint，而是能回答模型为什么搜索、为什么答错、工具有没有污染结果，以及某个 reward shaping 到底改善了什么、牺牲了什么。
