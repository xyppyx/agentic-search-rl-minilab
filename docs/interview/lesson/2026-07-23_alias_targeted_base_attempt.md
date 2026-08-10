# Alias/Granularity Targeted Base 评测阻塞记录

记录时间：2026-07-23 10:00 CST

## 目标

在 `alias_granularity_eval_80.jsonl` 上先跑 prompt-only base，再跑 `turn_credit_evidence_bridge_20step`，用于比较 turn-level reward checkpoint 是否损伤别名、答案粒度和 strict EM 相关能力。

## 配置

- 数据：`my-search-r1/datasets/alias_granularity_eval_80.jsonl`
- 样本数：80
- backend：`zhihu_search`
- model：prompt-only base，未传 `--model-path`
- seed：42
- batch size：1
- 输出：`my-search-r1/eval_results/targeted_eval_20260723/alias_prompt_base_20260723.jsonl`

## 结果

本轮生成 80 条 trajectory，但未通过工具门槛：

- Zhihu requests：132
- success rate：0.9924
- error rate：0.0076
- rate-limit/timeout：0
- tool failure records：1

失败样本：

- id：`dev_8490`
- source：`2wikimultihopqa`
- question：`Which country the director of film Queen Of Blood (2014 Film) is from?`
- failed query：`Chris Alexander director Queen of Blood 2014 nationality`
- error type：`parse_error: TypeError`

终端参考指标如下，但由于 success rate 未达 1.0，不能作为有效 prompt-only baseline：

- EM macro：0.4375
- format：0.9125
- 平均搜索：1.6500
- `possible_alias_match=10`
- `multi_candidate_answer=2`
- `missing_followup_query=0`

## 决策

按项目规则，真实搜索 eval 中出现 `tool_failures > 0` 或 success rate 低于 1.0 时应停止实验并总结。因此本轮没有继续运行 `turn_credit_evidence_bridge_20step`，避免生成不可对齐的无效对比。

## 下一步

在 Zhihu health check 正常后重跑 alias prompt-only base，要求 success rate 1.0。只有有效 base 建立后，再运行 `turn_credit_evidence_bridge_20step` 并做 gained/lost 对比。
