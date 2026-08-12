# Evaluation Design

本文是公开版评测设计说明，解释为什么采用 dev70 与 bridge150 两层评测，以及它们分别验证什么能力。

## 评测目标

Search-R1 MiniLab 评测不只看最终答案是否正确，还要区分：

- 输出协议是否稳定。
- 模型是否主动搜索并做必要 follow-up。
- 搜索动作是否真正服务于 final answer。
- 外部工具是否失败、为空或污染 observation，并据此排除不 clean 的评测结果。
- 训练是否诱导少搜、早答、重复搜索或搜满不答。

因此当前采用两层评测：

| Eval      | 角色         | 关注点                            |
| --------- | ------------ | --------------------------------- |
| dev70     | 快速健康评测 | checkpoint 是否整体退化           |
| bridge150 | 多跳定向评测 | bridge/final-hop 搜索策略是否改善 |

## Dev70

dev70 是小规模通用评测集，按不同数据来源均衡抽样。它的作用不是给出最终论文式结论，而是快速判断一个训练分支是否值得继续。

主要用途：

- 比较 prompt、reward、turn-credit 和辅助目标路线。
- 检查输出格式是否崩坏。
- 检查搜索次数是否失控。
- 检查外部工具失败是否污染评测。
- 作为进入更强 targeted eval 前的初筛。

Dev70 更像 checkpoint health gate：如果一个路线在 dev70 上已经明显退化，就不应直接进入更昂贵的长评测。

## Bridge150

bridge150 是面向多跳搜索的 targeted eval。它从候选数据中筛选更可能需要 bridge entity、role binding 和 final-hop attribute 查询的问题。

主要用途：

- 验证模型是否能先找到中间实体。
- 验证模型是否会继续查询 final-hop 属性。
- 检查模型是否因为训练而过早回答。
- 检查搜满后是否能输出合法短答案。
- 验证 turn-level credit 是否真的改善搜索策略，而不只是改善格式。

Bridge150 不是无偏总体评估；它是压力测试集，用来回答“这个 Agent 是否学会多跳搜索行为”。

## 指标口径

两层评测都需要同时观察以下维度：

| 维度                 | 目的                                                  |
| -------------------- | ----------------------------------------------------- |
| Answer correctness   | 判断最终任务效果                                      |
| Format               | 判断输出协议稳定性                                    |
| Search calls         | 判断搜索成本和停止策略                                |
| Tool reliability     | 排除真实工具失败污染                                  |
| Behavior diagnostics | 定位 early answer、missing follow-up、bad loop 等问题 |

正式结果必须说明工具成功率边界。真实搜索失败的 run 不能和 clean run 混写；如果使用补救口径，需要单独标注。

注意：工具失败记录是评测可信度约束，不是当前主算法目标。早期“面向不可靠搜索工具”的故障注入训练方向已暂时搁置；当前路线要求正式效果尽量来自 tool success rate 达标的 clean run。

## 与路线选择的关系

Dev70 负责发现整体退化，bridge150 负责验证最终路线真正改善的核心能力。

当前路线选择遵循这个顺序：

```text
dev70 health gate
  -> bridge150 targeted eval
  -> clean/patched boundary check
  -> case review
```

这种分层评测可以避免只追单个 EM，也能避免把工具波动误判成模型策略变化。
