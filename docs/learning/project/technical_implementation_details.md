# Robust Search-R1 MiniLab 技术实现细节

## 系统结构

项目正式实现位于 `my-search-r1/`，围绕 Search-R1 的“模型生成 tool call - 搜索工具返回 observation - 模型继续推理 - 最终 Answer”循环构建。

核心模块：

| 模块 | 作用 |
| --- | --- |
| `search_r1_minilab/tools/` | 搜索 backend、registry、Zhihu API 适配、failure injection |
| `search_r1_minilab/protocol.py` | chat template、tool call 解析、tool observation 构造 |
| `search_r1_minilab/rollout.py` | 训练级 group rollout，记录 old logprob、advantage、turns 和 tool events |
| `search_r1_minilab/training.py` | PyTRIO GRPO datum 构建、loss、KL/reference 约束和 optimizer step |
| `search_r1_minilab/rewards.py` | EM reward、format reward、行为 penalty/bonus |
| `search_r1_minilab/diagnostics.py` | trajectory 行为 bucket 和失败类型统计 |
| `search_r1_minilab/offline_diagnostics.py` | alias、答案粒度、missing follow-up、bad-loop 等离线标注 |
| `search_r1_minilab/turn_credit.py` | helpful/evidence/final-hop turn-level credit detector |
| `scripts/eval_pytrio.py` | base/checkpoint 统一评测入口 |
| `scripts/train_pytrio.py` | PyTRIO GRPO 训练入口 |
| `scripts/analyse_*.py` | trajectory、checkpoint、reward sensitivity、turn-credit 分析 |

## 工具层

搜索 backend 统一封装为 registry：

- `mock_search`：固定 fixture，适合单元测试。
- `local_bm25`：读取本地 JSONL 小语料，适合离线 smoke。
- `zhihu_search`：调用真实 Zhihu Search API，支持多 key、限流切换、有限重试和错误脱敏。
- `FailureWrapperBackend`：按固定 seed 注入 timeout、empty、noise、rate-limited，用于验证不可靠工具场景。

工具层记录的核心指标：

- request count、success rate、latency。
- error、timeout、rate limit、empty result。
- 每条 trajectory 的 `search_calls`、`tool_failures`、empty observation、duplicate query。

这使评测时可以区分：

- 模型策略失败：query 错、过早回答、证据读错。
- 工具环境失败：`url_error`、`parse_error`、timeout、rate limit。

## Rollout 协议

模型输出必须符合 Search-R1 工具调用协议：

```text
<tool_call>
<function=search>
<parameter=query>
...
</parameter>
</function>
</tool_call>
```

最终答案要求：

```text
Answer: <shortest single answer span>
```

rollout 状态机会：

1. 构造系统 prompt 和用户问题。
2. 调用 PyTRIO sampler 生成 assistant turn。
3. 解析 assistant 输出，如果是 tool call，则调用 registry 中的 search backend。
4. 将 tool observation 追加回对话，并加入 follow-up / final-answer 格式提醒。
5. 直到模型输出 `Answer:`、达到 max search calls、格式非法或超过 token budget。
6. 保存完整 trajectory，包括每轮 assistant text、tool query、observation、stop reason、reward 和 metadata。

## GRPO 训练

训练入口是 `scripts/train_pytrio.py`。默认稳定化配置：

| 参数 | 值 |
| --- | ---: |
| advantage normalization | `standardize` |
| advantage clip | 2.0 |
| KL coef | 0.01 |
| policy ratio clip | 0.2 |
| learning rate | 1e-5 |
| group size | 4 |
| questions per batch | 2 |

训练流程：

1. 从训练集采样 question batch。
2. 每个 question 采样 group size 条 trajectory。
3. 计算 base reward、format reward、行为 penalty/bonus 和 turn-level credit。
4. 对组内 reward 做 advantage normalization。
5. 构建 PyTRIO training datum，包含 old logprobs、loss mask 和 advantage。
6. 计算 reference logprobs，用 KL-style drift penalty 限制 sampled token 分布漂移。
7. 执行 backward 和 optimizer step。
8. 保存每 step 的 trajectory JSONL/Markdown，并按 `save_every` 保存 state 和 sampler weights。

## Reward 设计演进

### Base Reward

最初 reward 主要看：

- 是否输出合法 `Answer:`。
- extracted answer 是否 exact match gold answers。

问题：只看最终答案会让模型学到“少搜、早答、格式收束”，但多跳题容易漏掉 final-hop search。

### Behavior Penalty

尝试过：

- duplicate query penalty。
- empty result penalty。
- max-search no-answer penalty。
- verbose answer penalty。
- bad max-search penalty。
- date granularity penalty。
- multi-candidate answer penalty。

观察：轻量 penalty 能改善搜索效率和 format，但过强 penalty 会压掉必要 follow-up query，导致 EM 下降。

### Prompt / Rollout Guard

加入 prompt 约束：

- 多跳/关系题先找 bridge entity。
- 查完 bridge entity 后再查 final-hop 属性。
- 最终只输出一行短答案。
- 每次 tool observation 后追加 follow-up 与 final-answer 格式提醒。

效果：`prompt_search_budget_guard` 成为较强 prompt-only base，dev70 EM 0.4143、format 0.8857。

### Turn-Level Credit

核心思路：不要只给最终答案 reward，而是给“有用搜索轮次”可解释信号。

当前主策略 `final_hop_bridge` 包含：

| Signal | 作用 | 权重 |
| --- | --- | ---: |
| `evidence_bridge_search` | 奖励找到 bridge/evidence 的搜索 | +0.05 |
| `final_hop_attribute_search` | 奖励查 final-hop 属性的搜索 | +0.10 |
| `early_answer_missing_followup` | 惩罚未完成必要 follow-up 就回答 | -0.05 |
| `missing_final_hop_attribute` | 惩罚 final-hop 属性未查 | -0.08 |
| `final_answer_guard` | 惩罚搜过后 max-search 不答或 invalid final answer | -0.06 |

检测逻辑重点：

- evidence bridge 判断 query 与 observation 是否覆盖关键实体/候选。
- final-hop 判断 query 是否显式包含属性，如 director、birth date、death date、nationality、place of death 等。
- missing-final-hop detector 收紧 date 类判断，避免 observation 中任意年份误判为属性已覆盖。
- final-answer guard 主要覆盖 `max_search_calls` 且没有合法 `Answer:` 的最后一轮。

## Diagnostics

trajectory report 会输出：

- correct、wrong、invalid format、tool failure。
- direct/searched correct。
- empty search、duplicate query、max-search no-answer。
- too many search no gain。
- average reward、average search calls。

offline diagnostics 进一步标注：

- `possible_alias_match`：严格 EM 可能误伤的别名问题。
- `answer_granularity_miss`：答案粒度过细或过粗。
- `missing_followup_query`：多跳题缺少必要 follow-up。
- `bad_max_search_loop`：搜满仍没有有效答案。
- `multi_candidate_answer`：最终答案包含多个候选。

turn-credit analysis 会统计：

- candidate records/turns。
- training credit turns。
- early-answer penalty records。
- missing-final-hop penalty records。
- final-answer guard penalty records。

## 当前最优 Checkpoint

当前主 checkpoint：

```text
turn-credit-final-hop-guardfix-20step-20260806
```

训练配置：

| 参数 | 值 |
| --- | ---: |
| max steps | 20 |
| trajectories | 160 |
| group size | 4 |
| questions per batch | 2 |
| training tool failures | 0 |
| training correct | 50/160 |
| training format | 143/160 |
| mean reward | 0.3019 |
| avg search | 1.8438 |

训练信号命中：

| Label | Count |
| --- | ---: |
| `evidence_bridge_search` | 31 |
| `final_hop_attribute_search` | 18 |
| `early_answer_missing_followup` | 4 |
| `missing_final_hop_attribute` | 7 |
| `final_answer_guard` | 16 |

## 评测结果

### Dev70

有效 retry：

| 指标 | 值 |
| --- | ---: |
| Zhihu success rate | 1.0000 |
| EM macro | 0.4571 |
| correct | 32/70 |
| format | 0.9571 |
| avg search | 1.9000 |
| missing follow-up | 0 |
| answer granularity miss | 0 |
| bad max-search loop | 2 |

### Bridge Eval 150

完整 run 未过工具门槛，Zhihu success rate 0.9896。patched 版本替换 3 条工具失败样本后：

| 指标 | 值 |
| --- | ---: |
| tool failures | 0 |
| EM macro | 0.5142 |
| correct | 83/150 |
| format | 0.8267 |
| avg search | 3.2000 |
| missing follow-up | 3 |
| possible alias match | 4 |
| bad max-search loop | 8 |

对比：

| Model | EM macro | correct | format | avg search |
| --- | ---: | ---: | ---: | ---: |
| prompt-only base | 0.4750 | 74/150 | 0.7200 | 3.3067 |
| evidence-v2 20-step | 0.4583 | 81/150 | 0.9400 | 3.0933 |
| guard-fix 5-step patched | 0.4842 | 80/150 | 0.7933 | 3.2200 |
| guard-fix 20-step patched | 0.5142 | 83/150 | 0.8267 | 3.2000 |

结论：guard-fix 20-step patched 是当前 bridge EM/correct 最强，但 format 不如 evidence-v2 20-step。

### Alias / Granularity Eval 80

已完成的是 base vs evidence-v2 20-step：

| Model | EM macro | correct | format | avg search |
| --- | ---: | ---: | ---: | ---: |
| prompt-only base | 0.4500 | 36/80 | 0.9250 | 1.6500 |
| evidence-v2 20-step | 0.4375 | 35/80 | 0.9625 | 1.5125 |

结论：alias/granularity 上尚未取得 EM/correct 超过 base 的结果；guard-fix 20-step 尚未在该 eval 集上验证。

## 工程注意点

- 任何 success rate < 1.0 的 full run 不进入正式模型效果表。
- patched 协议必须写清楚组成，不能冒充独立全量 run。
- 不公开 PyTRIO sampler weights URI、SwanLab 私有链接、真实 API key 或远程凭据。
- 评测报告中必须区分 tool failure、empty observation、format error 和 exact-match wrong。
- 当前下一步不是盲目扩步数，而是围绕 format/max-search no-answer 做更细的 guard 或独立全量 rerun。
