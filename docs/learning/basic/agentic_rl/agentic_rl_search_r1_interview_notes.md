# Agentic RL 与 Search-R1 面试知识笔记

本文聚焦搜索型 LLM Agent 的 RL 训练与评测，服务 Robust Search-R1 MiniLab 项目面试。

## Agentic RL 是什么

Agentic RL 指对具备外部行动能力的 LLM Agent 做强化学习。这里的行动不只是生成文本，还包括调用搜索、代码执行、数据库、浏览器或其他工具。

普通问答 RL 主要优化：

```text
prompt -> answer
```

搜索型 Agentic RL 优化：

```text
prompt -> think/search query -> tool observation -> follow-up query -> final answer
```

关键差异：

- 动作空间包含自然语言 token 和工具调用。
- 环境返回来自外部工具，不完全可控。
- 中间动作质量会影响最终答案。
- 评测必须同时看答案、格式、工具成功率和搜索效率。

## Search-R1 的核心思想

Search-R1 类方法把搜索作为 LLM 推理的一部分，让模型学会在需要外部知识时主动查询，并基于 observation 继续推理。

典型流程：

1. 模型读入问题。
2. 模型生成 search tool call。
3. 工具返回网页或搜索摘要。
4. 模型根据 observation 决定继续搜索或回答。
5. 训练用 final answer reward 和格式约束优化策略。

面试中可以强调：Search-R1 的价值不只是“接搜索 API”，而是把检索行为纳入可训练策略，让模型学习 query 改写、证据读取、停止搜索和答案收束。

## ReAct、RAG 与 Search-R1 的区别

| 方法 | 核心 | 是否训练搜索策略 |
| --- | --- | --- |
| RAG | 先检索，再把文档塞给模型回答 | 通常不训练 LLM 的检索行动 |
| ReAct | 用 prompt 诱导模型交替 reasoning/action | 通常是 prompting，不一定 RL |
| Search-R1 | 把搜索调用纳入 rollout，用 RL 优化 | 是，重点是训练策略 |

本项目更接近 Search-R1：搜索工具是环境的一部分，模型在 rollout 中多轮调用工具，训练和评测都保存完整 trajectory。

## 搜索型 Agent 的关键能力

面试可按四类讲：

- Query planning：知道该搜哪个实体、属性或关系。
- Evidence reading：能从 observation 中抽取真正支持答案的信息。
- Follow-up search：多跳问题能先找 bridge entity，再查 final-hop attribute。
- Stop and answer：搜够后输出短答案，不陷入 max-search loop。

本项目最终围绕 final-hop 问题做 turn-level credit，是因为 gained/lost case review 显示：许多错误不是不会搜，而是少做必要二跳搜索或过早回答。

## Trajectory 为什么重要

Agentic RL 的平均 reward 不够解释行为变化。必须保存完整 trajectory：

```text
question
assistant tool call
search query
tool observation
assistant follow-up
final answer
reward / exact_match / format / search_calls / tool_failures
```

通过 trajectory 可以回答：

- 是模型没搜，还是工具返回空？
- 是 query 错，还是证据读错？
- 是 answer 错，还是 strict EM 误伤？
- 是训练真的提升，还是 format 变好导致表面指标变好？
- 是工具 API 失败污染结果，还是 checkpoint 策略退化？

这也是本项目区别于只跑脚本复现的核心工程价值。

## 工具不可靠性

真实搜索 backend 会有：

- timeout。
- rate limit。
- HTTP error。
- parse error。
- empty result。
- 弱相关或噪声结果。

Agentic RL 中不能把这些都当作模型错误，否则 reward 会被污染。本项目用工具层统一记录 `success_rate`、`error_rate`、`timeout_rate`、`rate_limit_rate`、`empty_rate`，并规定 success rate 低于 1.0 的 full run 不进入正式效果表。

面试回答口径：我把工具失败和模型策略失败分开记录，这样才能知道训练改进是否真实来自 policy，而不是外部 API 波动。

## Reward 设计

搜索型 Agent reward 通常包含：

- Correctness reward：最终答案是否正确。
- Format reward：是否输出合法 `Answer:`。
- Tool behavior reward：是否有效调用工具、是否重复搜索、是否处理空结果。
- Turn-level credit：中间搜索是否对最终答案有帮助。

本项目经历了三类方案：

1. 行为 penalty：惩罚重复 query、空结果、max-search 不答。
2. Prompt/rollout guard：提示模型先做 bridge search，再做 final-hop search，最终短答案。
3. Turn-level credit：奖励 evidence bridge search 和 final-hop attribute search，惩罚 early answer 与 final-answer guard 问题。

主要经验：单纯惩罚搜索成本容易让模型少搜和早答；对搜索型 Agent，更有效的信号通常是“奖励必要搜索”和“惩罚缺失关键 follow-up”。

## 多跳问题与 Final-Hop

多跳问题常见结构：

```text
question mentions entity A
need find bridge entity B
then query attribute of B
answer is final-hop attribute value
```

例子形式：

```text
某电影的导演出生在哪一年？
```

正确轨迹应该是：

```text
search film -> identify director -> search director birth date -> Answer
```

如果模型只搜 film 后直接回答，就可能出现 early answer。项目中的 `final_hop_attribute_search` 和 `missing_final_hop_attribute` 正是针对这类问题。

## Offline Diagnostics

offline diagnostics 是在不重新训练、不重新调用模型的情况下，读取已有 JSONL 轨迹做失败类型标注。

本项目使用的诊断：

| 诊断 | 含义 |
| --- | --- |
| `possible_alias_match` | strict EM 可能把别名判错 |
| `answer_granularity_miss` | 答案粒度过细或过粗 |
| `missing_followup_query` | 多跳题缺少必要后续搜索 |
| `bad_max_search_loop` | 搜满仍没有有效答案 |
| `multi_candidate_answer` | 最终答案含多个候选 |

价值：

- 降低盲目调参。
- 解释 gained/lost case。
- 在新 reward 上线前做 sensitivity，检查是否误伤正确样本。

## 评测指标

搜索型 Agent 不能只报 EM。建议至少报：

- EM / correct count。
- format rate。
- average search calls。
- tool success rate / tool failure count。
- empty result rate。
- repeated query rate。
- missing follow-up count。
- bad max-search loop count。
- gained/lost case review。

本项目对 `bridge_eval_150` 的结果会特别标注 patched protocol，是因为它由 full run 加失败样本 retry 合成，不等同一次独立全量 success rate 1.0 run。面试时主动说明这个边界，可信度更高。

## 项目可讲的技术贡献

可归纳为四点：

1. 工具层：统一 mock/local BM25/Zhihu/failure injection，隔离真实搜索 API 不稳定性。
2. 可观测性：保存 trajectory JSONL 和 Markdown report，支持 bucket、gained/lost、offline diagnostics。
3. 训练链路：迁移 PyTRIO GRPO train/eval，加入 std advantage、ratio clip 和 KL-style reference 约束。
4. 算法改进：从行为 penalty 迭代到 final-hop turn-level reward shaping，在 dev70 和 bridge targeted eval 上取得 EM/correct 正向证据，同时定位 format/max-search no-answer 为剩余瓶颈。

## 面试常见追问

**为什么说这是 Agentic RL，而不是普通 RAG？**

因为搜索不是固定前处理步骤，而是模型策略的一部分。模型在 rollout 中决定是否调用工具、用什么 query、是否继续 follow-up，以及何时输出最终答案。训练信号作用在这些决策上。

**为什么需要 mock 和 local BM25？**

真实搜索 API 有费用、限流和不稳定性。mock/local BM25 用于单元测试、smoke 和回归，保证工具协议、trajectory schema、report 和训练入口先可复现；真实 Zhihu backend 再用于效果评测。

**如何判断 reward shaping 没有 reward hacking？**

我不会只看训练 reward，而是看 dev EM、format、平均搜索、工具失败、offline diagnostics 和 gained/lost case。如果 reward 提升但 EM 降低、missing follow-up 增加或 format 退化，就认为出现了错误优化方向。

**为什么 patched bridge 结果不能直接当正式论文式结果？**

因为它不是一次独立全量 run，而是 full run 中工具失败样本单独 retry 后合成。它能说明当前策略在去除外部工具失败污染后有潜在正向效果，但严格结论仍需要 success rate 1.0 的独立全量重跑。

**项目下一步最值得做什么？**

不是继续盲目加训练步数，而是解决 format/max-search no-answer 和 final-hop follow-up 的权衡：在保持 EM/correct 的同时，提高 bridge eval 的 format，并做独立全量 success rate 1.0 重跑或多 seed 稳定性验证。
