# Reward And Auxiliary Objective Design

本文是公开版 reward 与辅助目标设计说明，记录路线原则和方法取舍。

## 核心问题

搜索型 Agent 的 reward 设计难点不在于给最终答案打分，而在于：

- 最终答案 reward 太稀疏，无法解释哪一次 search action 有用。
- 多跳问题需要 bridge entity 和 final-hop attribute 两类关键搜索。
- 简单惩罚搜索次数会同时压掉无效搜索和必要 follow-up。
- 真实工具失败会污染 reward，必须与模型策略失败分开。

因此本项目的 reward 路线从“最终答案 reward + 行为惩罚”逐步转向“turn-level credit + gated auxiliary objective”。

## 路线阶段

| 阶段                    | 设计重点                                                 | 结论                                         |
| ----------------------- | -------------------------------------------------------- | -------------------------------------------- |
| Base reward             | 只检查最终答案格式和 exact match                         | 可作为起点，但无法解释搜索行为               |
| Behavior penalty        | 惩罚重复搜索、空结果、搜满不答等坏行为                   | 能改善部分表面行为，但可能压掉必要 follow-up |
| Prompt guard            | 在 rollout prompt 中约束先搜索、控制搜索预算和最终短答案 | 提供强 prompt baseline                       |
| Stable GRPO             | 加入 advantage 标准化、ratio 约束和 reference 约束       | 提升小预算训练稳定性                         |
| Turn-level credit       | 奖励 bridge/final-hop 有用搜索，惩罚早答和停止失败       | 成为搜索策略主线                             |
| Gated self-distillation | 只在筛选后的正向 action tokens 上加入辅助蒸馏            | 作为强策略上的保守 refinement                |

## Turn-Level Credit

Turn-level credit 的目标是解决 credit assignment：

```text
final answer correct / wrong
  -> 哪个 search turn 应该被奖励或惩罚？
```

公开设计只保留信号类别：

| 信号类别                          | 作用                              |
| --------------------------------- | --------------------------------- |
| Evidence / bridge search credit   | 鼓励模型先找到中间实体或关键证据  |
| Final-hop attribute search credit | 鼓励模型继续查询最终答案所需属性  |
| Early-answer penalty              | 抑制只搜到中间信息就过早回答      |
| Missing-final-hop penalty         | 抑制缺少 final-hop 查询的错误答案 |
| Final-answer guard                | 抑制搜满后仍不回答或格式非法      |

这些信号不替代最终答案 reward，而是把最终成败拆到更可解释的搜索轮次上。

## Gated Self-Distillation

辅助蒸馏的目标是给 sparse reward 之外增加 token-level 信号，但它必须满足三个约束：

1. 主目标仍是 RL / turn-level credit，蒸馏只是辅助项。
2. 只蒸馏模型自己产生的 action tokens，不蒸馏工具 observation。
3. 只在正向轨迹或被 credit 命中的 turn 上启用，避免复制错误答案和早答行为。

不采用 naive full-sequence self-distillation 的原因：

- 它可能强化 wrong-valid final answer。
- 它可能把工具返回内容当成模型应学习的文本。
- 它可能固化少搜、早答或重复 query。
- 在小预算训练中，它可能掩盖 RL 主目标。

## 评测约束

Reward 或辅助目标只有在同时满足以下观察维度时，才被认为有效：

| 维度             | 目的                                          |
| ---------------- | --------------------------------------------- |
| Correctness      | 确认最终答案没有退化                          |
| Format           | 确认输出协议稳定                              |
| Search cost      | 确认没有无效搜索膨胀                          |
| Tool reliability | 确认结果不被外部工具失败污染                  |
| Diagnostics      | 确认 follow-up、早答、bad loop 等行为没有恶化 |

单一训练 reward 上升、单一平均指标变好，不能单独作为路线选择依据。

## 当前收敛

当前公开结论是：

- penalty-only 路线停放。
- prompt guard 保留为强 baseline。
- turn-level credit 是搜索策略主线。
- gated self-distillation 只作为强策略上的保守 refinement。
- 后续不优先扩大训练步数，而是优先做 trajectory case review 和更可靠 teacher / replay 设计。
