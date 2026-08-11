# OPD / OPSD 面试知识笔记

本文面向 Robust Search-R1 MiniLab 的面试复习，解释 OPD/OPSD 的基本思想、常见风险，以及本项目为什么采用 gated OPSD v2。

## 一句话解释

OPD 可以理解为在 RL 训练中加入策略蒸馏信号，让当前 policy 在合适位置贴近一个更可靠的 teacher policy。OPSD 则是 on-policy self-distillation：用当前或近期策略产生的轨迹，再用筛选后的 token-level teacher logprob 作为辅助目标。

本项目里的 OPSD v2 不是“模型无脑模仿自己”，而是在 guard-fix 20-step 强 checkpoint 上做保守 refine：

```text
guard-fix 20-step policy
  -> rollout 得到搜索轨迹和 GRPO advantage
  -> 只在正向且被 turn-credit 命中的 assistant tokens 上加 OPSD loss
  -> GRPO 仍是主目标，OPSD 只是小系数辅助项
```

最终路线固定为：

```text
turn-credit-final-hop-guardfix-20step-20260806
  -> guardfix20-resume-opsd-v2-5step-20260811
```

## OPD 与 OPSD

OPD 的重点是 policy distillation。它通常需要回答三个问题：

| 问题 | 含义 |
| --- | --- |
| teacher 是谁 | base model、reference model、历史 checkpoint、best-of-N response 或人工/规则筛选轨迹 |
| 蒸馏哪些 token | 全序列、final answer、tool call、被 reward/credit 命中的 turn |
| 什么时候启用 | 全部样本、正 reward 样本、正 advantage 样本、特定行为 bucket |

OPSD 的特殊点是 teacher 与 student 更接近，甚至来自同一训练过程。因此它更便宜，但也更容易把当前 policy 的坏习惯固化下来。

面试中可以这样说：

> OPD/OPSD 的价值不是替代 RL reward，而是给稀疏 reward 之外提供更密集的 token-level 约束。关键不在“有没有蒸馏”，而在 teacher、mask 和 gate 是否避免把错误轨迹也蒸馏进去。

## 和 SFT、KL、RLHF 的区别

| 方法 | 训练信号 | 主要作用 | 风险 |
| --- | --- | --- | --- |
| SFT | 人工或构造好的目标文本 | 学固定格式和示范轨迹 | 需要高质量标注，可能过拟合示范搜索模式 |
| KL reference | 惩罚偏离参考模型 | 稳定 RL，防止语言/格式崩坏 | 只约束漂移，不告诉模型哪段行为有用 |
| RL / GRPO | 轨迹 reward 和 advantage | 优化最终任务收益和行为偏好 | reward 稀疏，credit assignment 难 |
| OPD/OPSD | teacher token logprob | 提供 token-level 辅助信号 | 可能蒸馏错误、早答、少搜或格式问题 |

本项目使用 KL-style reference 约束稳定整体分布，同时使用 gated OPSD v2 对少量正向搜索行为做局部强化。二者不是同一个东西：KL 是“别偏太远”，OPSD 是“这些被筛中的动作可以更像 teacher”。

## 为什么不能做 naive self-distillation

搜索型 Agent 的轨迹里既有 assistant token，也有工具 observation；既有正确搜索，也有错误 query；既有最终答案，也有中间思考和工具调用。

如果直接全序列蒸馏，会有几个问题：

- 把 wrong-valid final answer 蒸馏进去，模型更自信地答错。
- 把 tool observation tokens 纳入 loss，等于训练模型复述环境返回，而不是学习行动策略。
- 把 early answer 或短答案偏好固化，削弱必要 follow-up search。
- 把无效搜索、重复 query、max-search loop 也当成可学习行为。
- 在小预算训练中，辅助 loss 可能压过 GRPO 主目标。

因此 OPSD 是否合理，核心看 gate 和 mask。没有 gate 的 OPSD 很容易变成“自己模仿自己”；有 gate 的 OPSD 才可能成为保守的正向行为蒸馏。

## 本项目的 Gated OPSD v2

OPSD v2 的设计目标是：只保留对 final-hop 搜索策略有帮助的密集 token-level 信号，不改变 GRPO/turn-level credit 主线。

公开参数口径：

| 参数 | 值 | 解释 |
| --- | --- | --- |
| `--opsd-coef` | `0.01` | 小系数辅助 loss，避免盖过 RL |
| `--opsd-mask-policy` | `credited_turns` | 只蒸馏被 turn-credit 命中的 assistant turn |
| `--opsd-positive-policy` | `positive_advantage` | 只在正向 advantage 轨迹上启用 |
| `--opsd-min-teacher-logprob` | `-3.0` | 过滤 teacher 低置信 token |
| `--resume-state` | guard-fix 20-step final state | 从强 checkpoint 恢复训练 |
| `--reference-model-path` | guard-fix 20-step final weights | 同时作为 KL/reference 与 OPSD teacher |

训练口径：

- 第一阶段 guard-fix 20-step 学会 bridge search、final-hop search 和 final-answer guard。
- 第二阶段 OPSD v2 5-step 只做 conservative refinement。
- teacher 与 student 起点相同，但 gate 使用当批 rollout 的正向 advantage 和 turn-credit 命中，避免全序列自复制。

面试中可以强调：我不是从 Qwen base 上直接做 OPSD 最终选择，而是先得到有搜索行为基础的 guard-fix checkpoint，再用 OPSD v2 做短程、低系数、正向 gated refine。

## 为什么最终选择 5 Step

5 step 容易被质疑“训练太少”，回答重点是它不是 base model 的完整训练，而是强 checkpoint 上的 conservative refinement。

已有对照：

| 模型/路线 | dev70 | bridge150 | 结论 |
| --- | ---: | ---: | --- |
| guard-fix 20-step | EM 0.4571, correct 32/70, format 0.9571 | patched EM 0.5142, correct 83/150, format 0.8267 | 搜索策略基座 |
| guardfix20 + OPSD v2 5-step | EM 0.4857, correct 34/70, format 0.9857 | clean EM 0.5242, correct 87/150, format 0.9067 | 当前最终路线 |
| guardfix20 + OPSD v2 20-step seed43 | EM 0.4571, correct 32/70, format 1.0000 | clean EM 0.5317, correct 81/150, format 0.8133 | bridge EM macro 略高，但 correct/format/search 综合弱 |

选择 5-step 的理由：

- 它同时提升 dev70 与 bridge150 clean correct 数。
- 它比 20-step 保持更好的 bridge format 和平均搜索效率。
- 20-step 对照说明“更多步数”不是稳定收益来源。
- OPSD 是辅助 refine，不是重新训练主策略，短步数本身合理。

## 面试常见追问

**OPSD 和 KL 有什么区别？**

KL 是全局稳定约束，防止 policy 偏离 reference 太远；OPSD 是局部 token-level 学习信号，只在被 gate 选中的 token 上鼓励贴近 teacher。

**为什么 teacher 不是 base model？**

base model 没有经过项目中的 final-hop guard-fix 训练，搜索策略弱。最终路线使用 guard-fix final weights 作为 teacher，是为了蒸馏已经学到的搜索行为，而不是把 base 的早答/少搜倾向拉回来。

**OPSD v2 是否只是自我模仿？**

不是无条件自我模仿。v2 同时要求 `credited_turns` 和 `positive_advantage`，并过滤低 teacher logprob token；没有被 turn-credit 证明有用、或者 advantage 非正的轨迹不会贡献 OPSD loss。

**为什么不把 mask 设成 null 或全序列？**

全序列 mask 会把工具 observation、错误 final answer、无效 query、max-search loop 一起蒸馏，容易强化坏行为。搜索型 Agent 更适合 action-token/credited-turn mask。

**OPSD 失败时怎么判断？**

不能只看训练 loss，要看 dev EM、correct、format、avg search、missing follow-up、bad max-search loop 和 tool success rate。如果 EM 或 format 下降，或者平均搜索上升但 correct 不升，就说明辅助目标可能干扰了策略。
