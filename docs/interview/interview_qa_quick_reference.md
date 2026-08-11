# 面试问答速查版

本文用于面试前快速过一遍高频追问。回答口径尽量压缩成 30 秒到 1 分钟可直接口述的版本。

## 项目一句话

我做的是 Robust Search-R1 MiniLab：基于 Qwen3.5-4B、PyTRIO GRPO 和真实 Zhihu Search backend，搭建一个可观测的搜索型 Agentic RL 实验框架。最终路线固定为 guard-fix 20step + OPSD v2 5step，核心是先通过 final-hop turn-level credit 学搜索策略，再用 gated on-policy self-distillation 做保守微调。

## 为什么做这个项目

Search-R1 类 Agent 不只是回答问题，还要决定什么时候搜索、搜什么、是否继续 follow-up、什么时候停止并输出短答案。真实搜索 API 又会有空结果、限流、解析错误和超时。如果只看最终 EM，很难判断模型到底是策略变好了，还是工具环境污染了结果。所以我重点做了统一工具层、trajectory JSONL、offline diagnostics 和 turn-level reward，让训练和评测可复盘。

## 为什么不是普通 RAG

普通 RAG 通常是先固定检索，再把文档交给模型回答；检索行为多数不是模型策略的一部分。我的项目里搜索是在 rollout 中由模型主动决策的：模型可以生成 search query，读取 observation，再决定继续搜还是回答。训练信号会作用到这些决策上，所以这是 Agentic RL，而不是只做检索增强生成。

## Search-R1 和 ReAct/RAG 的区别

ReAct 更偏 prompt 诱导，让模型按 reasoning/action 格式调用工具，但不一定训练策略。RAG 更偏检索模块加生成模型。Search-R1 把搜索调用纳入 RL rollout，用 reward 优化模型的搜索和回答策略。我的项目继承的是 Search-R1 这条线，重点是让搜索行为可训练、可诊断、可评测。

## 项目做了哪些工程

我把实现放在 `my-search-r1/`，主要做了四层：

- 工具层：统一 mock、local BM25、Zhihu Search 和 failure injection。
- Rollout 层：保存完整 trajectory，记录 query、observation、final answer、reward、format、tool failures。
- 训练层：迁移 PyTRIO GRPO train/eval，加入 advantage standardization、ratio clip 和 KL-style reference 约束。
- 分析层：实现 trajectory report、offline diagnostics、reward sensitivity、turn-credit analysis 和 gained/lost review。

## 核心算法改进是什么

核心改进分两段。第一段是从最终答案 reward 逐步走到 turn-level credit：奖励 evidence bridge search 和 final-hop attribute search，同时惩罚 early answer、missing final-hop attribute 和 final-answer/max-search no-answer。第二段是在最强 guard-fix checkpoint 上加入 OPSD v2，但不是全序列自蒸馏，而是只在 positive advantage 且被 turn-credit 命中的 assistant turn 上做小系数 distillation。

## 为什么需要 turn-level credit

多跳搜索题的关键动作经常发生在中间 turn。例如先搜电影找到导演，再搜导演生日。如果只给最终答案 reward，模型不知道是哪次搜索有用，也不知道自己是否过早回答。turn-level credit 可以把功劳或责任分给具体搜索轮次，更直接地引导“该搜什么、什么时候继续搜、什么时候回答”。

## Reward shaping 失败经验

最典型的失败是简单 penalty。比如惩罚重复搜索、空结果或搜满不答，确实能降低平均搜索次数和部分 bad loop，但也会让模型更保守，少做必要的 follow-up query。结果是 format 或搜索效率变好，EM 反而下降。这个经验让我不再只调 penalty 权重，而是引入 offline diagnostics 和 turn-level 正向 credit，先判断错误类型，再设计更定向的 reward。

## 为什么不继续堆训练步数

项目里观察到步数变多不一定更好。比如一些 50-step 或 20-step 对照在 format、搜索效率或 missing follow-up 上出现退化。小预算 Agentic RL 对 reward 和工具环境很敏感，盲目扩步数可能放大错误策略。所以我的结论是先固定 prompt 和稳定化训练配置，再通过 dev70、targeted eval、gained/lost 和 diagnostics 判断是否值得扩大训练。

## 训练稳定性怎么做

默认使用 standardized advantage、advantage clip 2.0、policy ratio clip 0.2、learning rate 1e-5 和 KL-style reference 约束。直觉是：GRPO 用组内相对 reward 构造 advantage，不额外训练 critic；ratio clip 和 KL 约束限制 policy 远离 reference model，避免小样本 reward 把模型推到格式崩坏或语言退化。

## 指标怎么报

我不会只报 EM。搜索型 Agent 至少要看：

- EM / correct count。
- format rate。
- average search calls。
- tool success rate 和 tool failures。
- missing follow-up、bad max-search loop、possible alias 和 answer granularity。
- gained/lost case。

这样才能区分答案正确性、格式收束、搜索效率和外部工具可靠性。

## 当前最好结果怎么讲

最终路线是 `turn-credit-final-hop-guardfix-20step-20260806 -> guardfix20-resume-opsd-v2-5step-20260811`。dev70 clean 结果是 EM 0.4857、correct 34/70、format 0.9857、平均搜索 1.7286，tool success rate 1.0。

bridge_eval_150 上，最终路线 clean chunk 合并结果是 EM 0.5242、correct 87/150、format 0.9067、平均搜索 3.1400，tool failures 0。相对 prompt-only base 的 EM 0.4750、correct 74/150、format 0.7200 有明显提升；相对 guard-fix 20step patched 的 0.5142/83/150 也更强。

## 多种尝试如何对比

可以按实验路线讲：

| 阶段 | 代表结果 | 经验 |
| --- | --- | --- |
| prompt-only guard | dev70 EM 0.4143 | prompt 能改善格式和搜索预算，但策略学习有限 |
| behavior penalty | 部分减少重复/空搜 | 容易压掉必要 follow-up |
| evidence-v2 20step | dev70 format 1.0000，bridge format 0.9400 | format 强，但 EM/correct 不最高 |
| final-hop guard-fix 20step | dev70 EM 0.4571，bridge patched 0.5142 | final-hop credit 提升 EM/correct |
| guardfix20 + OPSD v2 5step | dev70 0.4857，bridge clean 0.5242 | 当前最终路线 |
| OPSD v2 20step seed43 | bridge EM 0.5317，但 correct 81/150、format 0.8133 | 宏平均高但综合不如 5step |

结论不要说“20step 更差”这么简单，而是说：20step seed43 提高了部分宏平均，但损伤了 bridge correct、format 和搜索成本，综合目标下不替代 5step。

## patched protocol 是什么

patched protocol 是为处理真实搜索工具失败做的评测补救。guard-fix 20-step 的 bridge full run 里有 3 条样本出现 5 个 `url_error`，Zhihu success rate 0.9896，不能作为正式全量成功 run。我单独重跑这 3 条失败样本，确认 success rate 1.0 后，用 retry 记录替换 full run 里的失败记录，得到 patched bridge150。最终 OPSD v2 5step 的 bridge150 已经是 clean chunks 合并，不再依赖 patched 口径。

## patched 结果能不能写简历

现在简历优先写最终路线的 clean bridge150：EM 52.4%、correct 87/150、format 90.7%、tool failures 0。patched guard-fix 20step 只作为历史对照，必要时说明它是 147 条 full run 记录加 3 条失败样本 retry 合成，不等同一次独立全量 success rate 1.0 run。

## OPSD 是什么，为什么不是自己模仿自己这么简单

OPSD 可以理解为在 on-policy RL 过程中加入一个辅助 distillation objective，让 student 不只依赖稀疏 reward，也能从 teacher/reference 的 token logprob 获得更密集信号。但 naive OPSD 确实可能退化成“自己模仿自己”，甚至蒸馏错误答案。

本项目的 v2 做了三个约束：teacher 使用 guard-fix final weights；mask 只覆盖 credited turns，不蒸馏 tool observation；positive policy 只选择 positive advantage 的 turn。这样它不是全序列自复制，而是在强 checkpoint 上保留有正向证据的搜索/回答行为。

## 为什么最终选 5step，不会被质疑太少吗

我的回答是：5step 不是从 base 开始训练，而是在 guard-fix 20step 的强 checkpoint 上做 conservative refinement。OPSD gate 本身很稀疏，20step 不等于 20 次强有效信号。实际方差对照里，20step seed43 的 bridge EM macro 略高，但 correct 从 87/150 降到 81/150，format 从 0.9067 降到 0.8133，平均搜索也更高。所以选 5step 是基于综合评测，而不是为了节省训练。

## 为什么强调工具成功率

因为真实搜索 API 的失败会直接影响 observation 和最终答案。如果不记录 tool failure，模型答错可能被误判为策略问题。我的项目规定 success rate 低于 1.0 的 full run 不进入正式效果表；patched 结果也必须单独标注。这是为了把模型策略问题和外部环境问题分开。

## local BM25 有什么用

local BM25 不代表真实搜索能力，它在完整 dev 上空结果率很高，只适合 smoke、mock 和回归测试。它的价值是无费用、可复现，可以先验证工具协议、trajectory schema、报告生成和训练入口。真正效果评测还是看 Zhihu backend，并且要报告工具成功率。

## offline diagnostics 有什么价值

它在不重新训练、不重新调用模型的情况下，对已有 trajectory 标注错误类型，例如 possible alias、answer granularity、missing follow-up、bad max-search loop。这样可以解释为什么某个 checkpoint EM 上升或下降，也能在新 reward 上线前做 sensitivity，避免误伤正确样本。

## 如果老师问最大难点

最大难点不是 GRPO 公式，而是 Agentic RL 的信号归因和评测可信度。多跳搜索里中间 query 和最终答案之间有长程 credit assignment；真实工具又会失败，污染 reward 和指标。因此我先搭可观测链路，再做 reward shaping，而不是直接追单个分数。

## 如果老师问创新点

我会讲成四点：

1. 工程上，把 Search-R1 改造成可复现、可观测的 MiniLab，支持多 backend、失败注入和 trajectory 报告。
2. 方法上，从行为 penalty 迭代到 final-hop turn-level credit，针对多跳 follow-up 和 early answer 做定向 reward。
3. 训练上，在强 guard-fix checkpoint 上加入 gated OPSD v2，避免 naive self-distillation，同时用 5step/20step 对照验证步数不是越多越好。
4. 评测上，建立 dev70、bridge150、alias80 和 gained/lost/diagnostics，明确区分模型策略失败和工具失败。

## 如果老师质疑指标不够高

我会承认这是小预算 POC，不包装成论文 SOTA。项目价值在于完整闭环：复现 Search-R1 链路，定位真实工具环境下的失败类型，提出并验证 turn-level reward 的正向证据，同时如实记录 format 和 patched protocol 的边界。这比只报一个不透明 EM 更适合面试展示。

## 如果老师问下一步

主路线已经固定，不再盲目扩训练步数。下一步是补最终 checkpoint 的 alias80，确认 alias/granularity 不被牺牲；同时做 gained/lost case review，把为什么 5step 提升 dev70/bridge、为什么 20step 损伤 format 讲清楚。

## 30 秒收尾

这个项目让我学到：搜索型 Agent 的 RL 不能只看最终答案，必须同时管理工具可靠性、轨迹可观测性、行为 reward 和失败诊断。我最后得到的不是单个完美 checkpoint，而是一套能解释模型为什么变好或变坏的 Agentic RL 实验框架。
