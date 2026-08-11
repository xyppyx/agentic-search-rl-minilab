# Guardfix20 Resume OPSD v2 Bridge150 Attempt

日期：2026-08-11

## 目标

按用户要求，使用 `guardfix20-resume-opsd-v2-5step-20260811` 的 final sampler weights 跑 `bridge_eval_150.jsonl`，验证该 checkpoint 在 bridge150 上是否能延续 dev70 clean 提升。

公开复盘不记录远程 sampler weights URI。

## Checkpoint

- Run name: `guardfix20-resume-opsd-v2-5step-20260811`
- Source: 从 `turn-credit-final-hop-guardfix-20step-20260806` final training state 恢复训练 5 step
- Eval weights: 该 run 的 final sampler weights
- Dataset: `my-search-r1/datasets/bridge_eval_150.jsonl`
- Backend: `zhihu_search`

## Health5

先跑 bridge150 前 5 条作为 health check，结果 clean：

| Metric | Value |
| --- | ---: |
| EM macro | 0.8333 |
| correct | 4/5 |
| format | 1.0000 |
| avg search | 2.0000 |
| too many search no gain | 0/5 |
| max-search no-answer | 0/5 |
| Zhihu requests | 10 |
| Zhihu success rate | 1.0000 |
| tool failures | 0 |

输出：

- `my-search-r1/eval_results/gated_opsd_v2_guardfix_resume_20260811/guardfix20_resume_opsd_v2_5step_bridge_health5_20260811.jsonl`
- `my-search-r1/eval_results/gated_opsd_v2_guardfix_resume_20260811/guardfix20_resume_opsd_v2_5step_bridge_health5_20260811.md`

## First Full Bridge150 Attempt

首次 full run 使用单个 PyTRIO sampling session 跑 150 条，在进度 19/150 处中断：

- failure: `sampling_run_inactive`
- completed records: 未落盘；`eval_pytrio.py` 只在完整 eval 结束后写 JSONL，因此该中断不能产生正式指标
- 判定：不是模型策略错误，也不是 Zhihu 搜索工具返回失败，而是 PyTRIO sampling run 外部状态中断

为降低长会话失活风险，随后给 `my-search-r1/scripts/eval_pytrio.py` 增加 `--offset` 参数，准备按 15 条一段分片重跑并合并正式 bridge150 JSONL/report。

分片重跑在创建新的 sampling run 时被 PyTRIO 拒绝：

- failure: `billing_insufficient_balance`
- status: 无法继续创建 sampling run
- 判定：当前 bridge150 正式全量评测被外部计费/余额状态阻塞

## Retry Result

用户要求重试后，PyTRIO 余额/计费状态已恢复。为了避免长 sampling session 再次失活，本次按 15 条一段分片评测。首次 chunk 0-15 出现 1 次 Zhihu 工具错误，虽然该样本最终答对，但不符合 clean 口径；随后重跑 chunk 0-15 并通过 tool success rate 1.0 检查。后续所有 chunk 均要求 15/15 记录且 `tool_failures=0`。

最终合并 10 个 clean chunks：

| Metric | Value |
| --- | ---: |
| EM macro | 0.5242 |
| correct | 87/150 |
| format | 136/150 |
| format rate | 0.9067 |
| avg search | 3.1400 |
| tool failures | 0 |
| duplicate query trajectories | 7/150 |
| max-search no-answer | 14/150 |
| too many search no gain | 46/150 |
| empty observations | 7/471 |

按数据源：

| Source | EM | Correct |
| --- | ---: | ---: |
| 2wikimultihopqa | 0.6300 | 63/100 |
| bamboogle | 0.6000 | 6/10 |
| hotpotqa | 0.4667 | 14/30 |
| musique | 0.4000 | 4/10 |

Offline diagnostics:

| Metric | Value |
| --- | ---: |
| wrong valid | 49 |
| possible alias match | 4 |
| missing follow-up query | 4 |
| helpful follow-up query | 126 |
| bad max-search loop | 6 |
| multi-candidate answer | 0 |
| answer granularity miss | 0 |

Turn-credit analysis:

| Metric | Value |
| --- | ---: |
| evidence candidate records | 124 |
| evidence training credit turns | 56 |
| final-hop candidate records | 119 |
| final-hop training credit turns | 53 |
| training credit records | 37 |
| early answer penalty records | 5 |
| missing final-hop penalty records | 3 |
| final answer guard penalty records | 14 |

输出：

- `my-search-r1/eval_results/gated_opsd_v2_guardfix_resume_20260811/guardfix20_resume_opsd_v2_5step_bridge150_20260811.jsonl`
- `my-search-r1/eval_results/gated_opsd_v2_guardfix_resume_20260811/guardfix20_resume_opsd_v2_5step_bridge150_20260811.md`
- `my-search-r1/eval_results/gated_opsd_v2_guardfix_resume_20260811/guardfix20_resume_opsd_v2_5step_bridge150_offline_diagnostics_20260811.*`
- `my-search-r1/eval_results/gated_opsd_v2_guardfix_resume_20260811/guardfix20_resume_opsd_v2_5step_bridge150_turn_credit_analysis_20260811.*`

## Current Conclusion

- `guardfix20-resume-opsd-v2-5step-20260811` 在 bridge150 上取得 clean EM macro 0.5242、correct 87/150、format 0.9067、平均搜索 3.1400。
- 该结果超过此前 `turn-credit-final-hop-guardfix-20step-20260806` bridge150 patched EM 0.5142/correct 83/150，且本次是 150/150 clean 分片合并结果。
- 主要风险仍是 bridge format 和 max-search/no-answer：invalid format 14/150、max-search no-answer 14/150、too many search no gain 46/150。
- 下一步应补 alias80，确认收益是否只来自 bridge/multihop 场景，还是会牺牲 alias/granularity。

## Resume Protocol

建议按 15 条分片：

```bash
PYTHONPATH=my-search-r1 uv run python my-search-r1/scripts/eval_pytrio.py \
  --data my-search-r1/datasets/bridge_eval_150.jsonl \
  --backend zhihu_search \
  --offset 0 \
  --limit 15 \
  --batch-size 1 \
  --model-path "<guardfix20-resume-opsd-v2-5step-final-weights>" \
  --jsonl-output my-search-r1/eval_results/gated_opsd_v2_guardfix_resume_20260811/bridge150_chunks/guardfix20_resume_opsd_v2_5step_bridge150_0_15_20260811.jsonl \
  --report-output my-search-r1/eval_results/gated_opsd_v2_guardfix_resume_20260811/bridge150_chunks/guardfix20_resume_opsd_v2_5step_bridge150_0_15_20260811.md
```

依次覆盖 offsets: `0, 15, 30, 45, 60, 75, 90, 105, 120, 135`。全部成功后再合并 JSONL，生成统一 Markdown report、offline diagnostics 和 turn-credit analysis。
