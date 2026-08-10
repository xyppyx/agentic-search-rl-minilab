# Alias/Granularity Base vs 20-Step 评测复盘

记录时间：2026-07-23 10:15 CST

## 目标

在 `alias_granularity_eval_80.jsonl` 上比较 prompt-only base 与 `turn_credit_evidence_bridge_20step`，验证 turn-level evidence reward 是否损伤别名、答案粒度和 strict EM 相关能力。

## 前置修复

首次 alias prompt-only base 尝试中，`dev_8490` 出现 1 次 Zhihu `parse_error: TypeError`。本轮先修复 Zhihu backend 的错误信息收集：

- parse error 时在 `SearchResult.metadata` 中记录 `api_status`
- 记录截断后的 `api_response_excerpt`
- 记录 `api_response_truncated`
- 记录 `parse_error_detail`
- 不记录 request header、Authorization 或 API key

验证：

```bash
PYTHONPATH=my-search-r1 uv run python -m unittest my-search-r1/tests/test_tools.py -v
PYTHONPATH=my-search-r1 uv run python -m compileall -q my-search-r1/search_r1_minilab my-search-r1/scripts
git diff --check
```

结果：`test_tools.py` 15 个测试全部通过，compileall 和 diff check 通过。

## 数据与配置

- 数据：`my-search-r1/datasets/alias_granularity_eval_80.jsonl`
- 样本数：80
- backend：`zhihu_search`
- seed：42
- batch size：1
- 输出目录：`my-search-r1/eval_results/targeted_eval_20260723/`

有效 base 使用 retry run：

- `my-search-r1/eval_results/targeted_eval_20260723/alias_prompt_base_retry_20260723.jsonl`

20-step 使用已有 final sampler weights，本复盘不记录远端 URI：

- `my-search-r1/eval_results/targeted_eval_20260723/alias_turn_credit_evidence_20step_20260723.jsonl`

对比报告：

- `my-search-r1/eval_results/targeted_eval_20260723/alias_base_vs_20step_comparison_20260723.md`

## 指标

两组均达到 Zhihu success rate 1.0。

| Run | EM macro | Overall correct | Format | Avg search | Alias risk | Granularity miss | Multi-candidate | Missing follow-up | Bad max-search loop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| prompt-only base retry | 0.4500 | 36/80 | 0.9250 | 1.6500 | 10 | 0 | 2 | 0 | 2 |
| evidence 20-step | 0.4375 | 35/80 | 0.9625 | 1.5125 | 10 | 0 | 2 | 1 | 1 |

Source-level EM：

| Source | Base | 20-step | Delta |
| --- | ---: | ---: | ---: |
| 2WikiMultihopQA | 0.5625 | 0.5000 | -0.0625 |
| HotpotQA | 0.0000 | 0.0000 | +0.0000 |
| NQ | 0.1875 | 0.1875 | +0.0000 |
| PopQA | 0.6875 | 0.6875 | +0.0000 |
| TriviaQA | 0.8125 | 0.8125 | +0.0000 |

## Gained / Lost

20-step 相对 prompt-only base：

- gained：1
- lost：2
- net：-1
- gained ids：`dev_12115`
- lost ids：`dev_11271`、`dev_8490`

## 结论

在 alias/granularity targeted eval 上，`turn_credit_evidence_bridge_20step` 没有出现答案粒度退化，也没有增加 alias risk 或 multi-candidate answer；format 从 0.9250 提升到 0.9625，平均搜索从 1.6500 降到 1.5125，bad max-search loop 从 2 降到 1。

但 20-step 的 EM macro/overall correct 略低于 prompt-only base：36/80 降到 35/80，主要来自 2WikiMultihopQA 从 0.5625 降到 0.5000。结论是 20-step 在 alias/granularity 能力上基本保持稳定，但不能说提升；它更像是用 1 条 correct 的代价换来更高 format 和更低搜索成本。

## 下一步

- 对 `dev_11271`、`dev_8490`、`dev_12115` 做小型 case review，确认 20-step 的 lost/gained 是否是真实能力变化还是 strict EM/搜索结果波动。
- alias/granularity 集暂不优先跑 50-step；50-step 在 bridge targeted 上已有工具失败和弱于 20-step 的信号。
