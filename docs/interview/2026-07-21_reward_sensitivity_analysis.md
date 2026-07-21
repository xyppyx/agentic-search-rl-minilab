# Reward Sensitivity Analysis - 2026-07-21

## 目标

用已有 Zhihu dev 70 eval trajectory JSONL 做离线 reward sensitivity/rescore，比较不同 penalty 配置对 reward 分布和误伤风险的影响。第一版目标不是在 70 条 dev 上调出最优权重，而是排除明显过强的 penalty，并确认 `penalty_v2_candidate` 是否比 `penalty_v1` 更温和。

输入 JSONL：

- base：`my-search-r1/eval_results/reward_train_compare_2026-07-21/base_dev.jsonl`
- 5-step 原始 reward：`my-search-r1/eval_results/reward_train_compare_2026-07-21/baseline_reward_ckpt_dev.jsonl`
- 5-step penalty reward：`my-search-r1/eval_results/reward_train_compare_2026-07-21/penalty_reward_ckpt_dev.jsonl`
- 20-step 原始 reward：`my-search-r1/eval_results/reward_train_compare_2026-07-21/base_20step_dev.jsonl`

输出 summary JSON、per-record JSONL 和 Markdown report 都写入同一 ignored 目录，不提交 Git。

## 配置

默认运行 4 个配置：

| Config | duplicate | empty | max_search | verbose | verbose_threshold |
| --- | ---: | ---: | ---: | ---: | ---: |
| `base_reward_v0` | 0.00 | 0.00 | 0.00 | 0.00 | 0 |
| `penalty_v1` | 0.05 | 0.03 | 0.05 | 0.02 | 8 |
| `penalty_v2_candidate` | 0.03 | 0.01 | 0.00 | 0.00 | 0 |
| `penalty_v2_no_empty` | 0.03 | 0.00 | 0.00 | 0.00 | 0 |

离线 `verbose_answer_penalty` 使用 final answer 的 whitespace word count 近似 tokenizer token count；真实训练仍以 tokenizer token count 为准。

## 命令

示例命令：

```bash
PYTHONPATH=my-search-r1 uv run python my-search-r1/scripts/analyse_reward_sensitivity.py \
  --input my-search-r1/eval_results/reward_train_compare_2026-07-21/base_20step_dev.jsonl \
  --summary-output my-search-r1/eval_results/reward_train_compare_2026-07-21/base_20step_reward_sensitivity_summary.json \
  --jsonl-output my-search-r1/eval_results/reward_train_compare_2026-07-21/base_20step_reward_sensitivity.jsonl \
  --report-output my-search-r1/eval_results/reward_train_compare_2026-07-21/base_20step_reward_sensitivity.md \
  --title '20-step Reward Sensitivity'
```

四份输入均用同一 CLI 和同一默认配置运行。

## 结果摘要

| Run | Config | mean_base_reward | mean_final_reward | mean_delta | penalized | correct penalized | wrong-valid penalized | duplicate | empty | max-search | verbose | missing-followup | alias | granularity |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base | `penalty_v1` | 0.2057 | 0.1907 | -0.0150 | 23 | 0 | 8 | 1 | 3 | 8 | 23 | 0 | 4 | 0 |
| base | `penalty_v2_candidate` | 0.2057 | 0.2049 | -0.0009 | 4 | 0 | 0 | 1 | 3 | 0 | 0 | 0 | 0 | 0 |
| base | `penalty_v2_no_empty` | 0.2057 | 0.2053 | -0.0004 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| 5-step 原始 reward | `penalty_v1` | 0.2871 | 0.2766 | -0.0106 | 15 | 0 | 7 | 2 | 4 | 3 | 15 | 1 | 1 | 0 |
| 5-step 原始 reward | `penalty_v2_candidate` | 0.2871 | 0.2857 | -0.0014 | 5 | 0 | 3 | 2 | 4 | 0 | 0 | 0 | 0 | 0 |
| 5-step 原始 reward | `penalty_v2_no_empty` | 0.2871 | 0.2863 | -0.0009 | 2 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 |
| 5-step penalty reward | `penalty_v1` | 0.2014 | 0.1914 | -0.0100 | 16 | 0 | 8 | 1 | 3 | 5 | 16 | 1 | 2 | 0 |
| 5-step penalty reward | `penalty_v2_candidate` | 0.2014 | 0.2006 | -0.0009 | 4 | 0 | 1 | 1 | 3 | 0 | 0 | 0 | 0 | 0 |
| 5-step penalty reward | `penalty_v2_no_empty` | 0.2014 | 0.2010 | -0.0004 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| 20-step 原始 reward | `penalty_v1` | 0.2600 | 0.2506 | -0.0094 | 15 | 0 | 6 | 0 | 3 | 3 | 15 | 1 | 1 | 0 |
| 20-step 原始 reward | `penalty_v2_candidate` | 0.2600 | 0.2596 | -0.0004 | 3 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 | 0 |
| 20-step 原始 reward | `penalty_v2_no_empty` | 0.2600 | 0.2600 | 0.0000 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## 结论

`penalty_v1` 的问题不是平均扣分很大，而是扣分对象不够干净：它在 4 份 JSONL 上都扣到 wrong-valid 样本，并在 5-step/20-step 上扣到 missing-followup 风险样本。这解释了 5-step penalty 训练虽然减少搜索次数，却降低 EM。

`penalty_v2_candidate` 明显更保守。四组离线重评分中，它没有扣到正确样本、missing-followup 样本、possible alias 样本或答案粒度样本；主要扣分来自空结果和少量重复 query。因此它比 `penalty_v1` 更适合作为下一轮小预算训练候选。

`penalty_v2_no_empty` 是一个 useful ablation：在 20-step 原始 reward 上完全不扣分，说明当前 20-step 的候选改造收益主要来自是否惩罚空结果，而不是重复 query。下一轮训练可以优先跑 `penalty_v2_candidate`，再视结果决定是否补一个 no-empty ablation。

## 面试叙事

这一步体现的是 Agentic RL reward 设计的工程闭环：

```text
trajectory 观测 -> offline diagnostic 分类错因 -> sensitivity/rescore 排除误伤配置
-> 小预算训练验证 -> 再决定是否进入更长训练或多 seed
```

相比直接调 reward 权重，这个流程更能解释为什么某个 penalty 会伤害策略：不是只看平均 reward，而是看它扣到了哪些类型的样本。
