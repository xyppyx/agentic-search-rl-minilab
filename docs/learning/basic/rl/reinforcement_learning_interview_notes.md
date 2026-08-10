# 强化学习面试知识笔记

本文面向 Search-R1 改进项目的面试复习，重点解释项目里真正用到的 RL 概念，而不是完整教材。

## 一句话框架

强化学习是在交互环境中学习策略。Agent 根据状态选择动作，环境返回下一个状态和 reward，训练目标是最大化期望累计回报。

在 LLM 场景里：

| RL 概念 | LLM/Agent 对应 |
| --- | --- |
| state / observation | 当前 prompt、历史对话、工具 observation |
| action | 生成下一个 token，或生成一次 tool call / final answer |
| policy | 当前语言模型 `pi_theta` |
| reward | 答案正确性、格式、工具行为、人工偏好或规则打分 |
| trajectory | 从问题到多轮搜索再到最终答案的完整轨迹 |
| return | 轨迹级 reward 或逐 token/逐 turn credit 的累计值 |

本项目可以说成：把 Search-R1 的搜索问答过程建模为一个带工具交互的序列决策问题，用 GRPO 类 policy gradient 方法优化模型在真实搜索环境下的行为。

## MDP 与 POMDP

经典 MDP 包含：

```text
(S, A, P, R, gamma)
```

- `S`：状态空间。
- `A`：动作空间。
- `P(s' | s, a)`：状态转移。
- `R(s, a)`：即时 reward。
- `gamma`：折扣因子。

LLM Agent 更接近 POMDP，因为模型通常看不到完整世界状态，只看到 prompt、历史消息和工具返回的局部 observation。搜索工具也可能为空、限流、超时或返回弱相关结果，因此策略需要对不完整和不可靠 observation 做决策。

面试回答要点：Search-R1 不是简单分类任务，模型每次搜索 query 都会改变后续上下文和答案质量，所以它是序列决策问题。

## Policy Gradient

policy gradient 直接优化策略参数，使高 reward 轨迹概率升高，低 reward 轨迹概率下降。

基本目标：

```text
J(theta) = E_{tau ~ pi_theta}[R(tau)]
```

REINFORCE 梯度形式：

```text
grad J(theta) = E[grad log pi_theta(a_t | s_t) * R(tau)]
```

实际训练常用 advantage 降低方差：

```text
grad J(theta) = E[grad log pi_theta(a_t | s_t) * A_t]
```

其中：

```text
A_t = R_t - baseline
```

直觉：

- `A_t > 0`：这个动作比同组平均更好，增加它的概率。
- `A_t < 0`：这个动作比同组平均更差，降低它的概率。

本项目里 group rollout 会对同一批问题采样多条 trajectory，再用组内 reward 计算 advantage。这样比只看单条样本更稳定，也更适合小预算训练。

## PPO

PPO 的核心是限制新旧 policy 更新幅度，避免一次更新把模型推得太远。

常见 ratio：

```text
r_t(theta) = pi_theta(a_t | s_t) / pi_old(a_t | s_t)
```

clip 目标：

```text
min(r_t * A_t, clip(r_t, 1 - eps, 1 + eps) * A_t)
```

含义：

- 如果新策略相对旧策略概率变化太大，就截断收益。
- 对 LLM 来说，这能减少训练后突然格式崩坏、重复输出或偏离原模型语言能力的风险。

本项目没有把重点放在完整 PPO 复现，而是采用 GRPO/PyTRIO 训练链路，并加入 ratio clip、KL-style reference 约束、advantage standardization 等稳定化手段。

## GRPO

GRPO 可以理解为面向 LLM RL 的 group-based policy optimization。它不依赖单独训练 critic，而是对同一 prompt 采样多个 response，用组内 reward 均值或标准差构造 advantage。

典型做法：

```text
A_i = (r_i - mean(r_group)) / std(r_group)
```

优点：

- 不需要 value model / critic，工程更轻。
- 同一问题的多条回答天然可比较，适合问答类 RL。
- 对小规模实验更容易跑通。

局限：

- group 内 reward 全相同时 advantage 为 0，可能跳过更新。
- reward 设计不稳时，模型会优化错误捷径，例如少搜、早答、只学格式。
- 多跳搜索的关键动作发生在中间 turn，只用最终答案 reward 容易产生 credit assignment 问题。

本项目面试表述：我使用 GRPO 的组内相对优势来训练搜索型 Agent，并用 trajectory 诊断发现最终 reward 不足以解释“哪一次搜索有用”，因此加入 turn-level credit。

## KL 约束

LLM RL 常加入 reference model 约束，避免 policy 远离 SFT/base model。

常见形式：

```text
reward' = reward - beta * KL(pi_theta || pi_ref)
```

或者在 loss 中加入 sampled-token logprob drift penalty。

本项目默认使用 `kl_coef=0.01` 的 KL-style reference 约束。面试时要讲清楚：这里的 KL 不是为了提升搜索能力本身，而是为了稳定训练，避免小样本 reward 把模型推到格式退化或语言退化区域。

## Reward Shaping

reward shaping 是在最终任务 reward 之外加入中间奖励或惩罚，引导更好的学习路径。

好处：

- 缓解 sparse reward。
- 让模型更快学到格式、工具调用、停止策略。
- 能把业务偏好显式写成训练信号。

风险：

- 权重过大会压过主目标，导致 reward hacking。
- 惩罚搜索次数可能减少无效搜索，也可能压掉必要 follow-up。
- 离线 sensitivity 通过不代表在线训练一定提升，因为策略分布会变。

本项目的经验：

- duplicate/empty/max-search penalty 能改善搜索效率，但过强会损伤 EM。
- 只做最终 reward 容易让模型过早回答。
- turn-level reward 更适合搜索型 Agent，因为关键行为发生在中间搜索轮次。

## Credit Assignment

credit assignment 指：最终答对或答错后，训练应该把功劳或责任分给哪些动作。

在 Search-R1 中，最终答案可能由多轮动作共同决定：

```text
query bridge entity -> read observation -> query final-hop attribute -> output Answer
```

如果只给最终答案 reward，模型不知道是哪次 search query 起作用，也不知道早答错在哪里。项目里的 `evidence_bridge_search`、`final_hop_attribute_search`、`early_answer_missing_followup` 就是在解决 turn-level credit assignment。

## 面试常见追问

**为什么不用监督微调直接学搜索轨迹？**

SFT 需要高质量标注轨迹，成本高且容易过拟合固定搜索模式。RL 可以从结果和行为信号中学习策略，但需要更强的可观测性和 reward 设计。本项目是小预算 POC，所以用规则 reward、offline diagnostics 和 targeted eval 做可控验证。

**为什么 GRPO 不需要 critic？**

GRPO 用同一 prompt 下多条采样结果的相对 reward 构造 baseline 和 advantage，避免额外训练 value model。代价是依赖 group 采样质量，group reward 全同会没有有效更新。

**为什么 reward 不能只看 EM？**

EM 是最终结果指标，但对 Agent 训练太稀疏。搜索型 Agent 还要学会何时搜索、搜什么、何时停止、如何处理工具失败和如何输出合法格式。只看 EM 会掩盖 format 和工具行为退化。

**你项目里最大的 RL 难点是什么？**

不是公式实现，而是 reward 信号和真实工具环境耦合。外部搜索 API 失败会污染 reward；多跳题中间搜索动作和最终答案之间存在长程 credit assignment；小预算训练下指标波动大，所以必须用 trajectory、gained/lost review 和 targeted eval 解释。
