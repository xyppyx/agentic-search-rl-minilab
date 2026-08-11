# Guardfix20 Resume OPSD v2 20-Step Seed43

日期：2026-08-11

## 目标

用户希望重训一次 20-step OPSD v2，以排查此前 20-step 结果是否受方差或工具失败影响；同时确认 eval 分片是否只是临时机制，以及训练能否分片。

公开复盘不记录远程 state/weights URI。

## Chunking Decision

- `eval_pytrio.py --offset` 是本轮新增的数据选择能力；默认仍是 `offset=0`，不自动分片。
- 不建议把 eval 分片设为默认。dev70 等短评测保持单次 eval 更简单；bridge150 这类长评测推荐显式分片并在报告中标注 clean chunks 合并。
- 训练不能像 eval 一样拆成独立 chunks 后合并；训练包含 optimizer state 和 on-policy rollout 依赖。可用 `--save-every` 和 `--resume-state` 做 checkpoint resume，但这不是独立分片合并。

## Training

Run name: `guardfix20-resume-opsd-v2-20step-seed43-20260811`

配置与此前 guardfix20 resume OPSD v2 相同，只将 seed 改为 43：

- source checkpoint: `turn-credit-final-hop-guardfix-20step-20260806` final training state
- reference/OPSD teacher: same source checkpoint final sampler weights
- `--max-steps 20`
- `--seed 43`
- `--turn-credit-policy final_hop_bridge`
- `--opsd-coef 0.01`
- `--opsd-mask-policy credited_turns`
- `--opsd-positive-policy positive_advantage`
- `--opsd-min-teacher-logprob -3.0`
- `--save-every 5`

训练结果：

| Metric | Value |
| --- | ---: |
| steps | 20/20 |
| effective updates | 16/20 |
| skipped update steps | 1, 3, 8, 20 |
| trajectories | 160 |
| correct | 47/160 |
| valid format | 151/160 |
| avg search | 2.0312 |
| mean reward / step | 0.2881 |
| Zhihu requests | 325 |
| Zhihu success rate | 1.0000 |
| tool failures | 0 |
| OPSD masked tokens | 4402 |
| avg OPSD masked tokens / step | 220.10 |
| avg OPSD mask rate | 0.0156 |
| avg student-teacher logprob gap | 0.0070 |
| credited trajectories | 57 |
| credited tokens | 7430 |

训练是 clean，但有效 update 稀疏：20 step 中 4 step 没有可训练 micro-batch。这个现象说明 20-step 并不是简单“更多有效优化”，而是稀疏门控信号上的长程微调。

## Dev5

| Metric | Value |
| --- | ---: |
| EM | 0.4000 |
| correct | 2/5 |
| format | 1.0000 |
| avg search | 2.8000 |
| Zhihu requests | 14 |
| Zhihu success rate | 1.0000 |
| tool failures | 0 |

输出：

- `my-search-r1/eval_results/gated_opsd_v2_guardfix_resume_20260811/guardfix20_resume_opsd_v2_20step_seed43_dev5_20260811.*`

## Dev70

Dev70 采用 7 个 clean chunks 合并。首次 chunk 20-30 出现 1 次 Zhihu 工具错误，已重跑并用 clean chunk 覆盖。

| Metric | Value |
| --- | ---: |
| EM macro | 0.4571 |
| correct | 32/70 |
| format | 1.0000 |
| avg search | 1.8857 |
| tool failures | 0 |
| too many search no gain | 13/70 |
| max-search no-answer | 0/70 |
| duplicate query trajectories | 0 |

按数据源：

| Source | EM | Correct |
| --- | ---: | ---: |
| 2wikimultihopqa | 0.5000 | 5/10 |
| bamboogle | 0.7000 | 7/10 |
| hotpotqa | 0.5000 | 5/10 |
| musique | 0.4000 | 4/10 |
| nq | 0.3000 | 3/10 |
| popqa | 0.3000 | 3/10 |
| triviaqa | 0.5000 | 5/10 |

Dev70 结论：seed43 20-step 恢复了 format 到 1.0，但 EM/correct 没有超过 5-step；与原 20-step reference 的 EM 0.4571 / correct 32/70 同档，并且这次是 clean。

输出：

- `my-search-r1/eval_results/gated_opsd_v2_guardfix_resume_20260811/guardfix20_resume_opsd_v2_20step_seed43_dev70_20260811.*`
- `my-search-r1/eval_results/gated_opsd_v2_guardfix_resume_20260811/guardfix20_resume_opsd_v2_20step_seed43_dev70_offline_diagnostics_20260811.*`
- `my-search-r1/eval_results/gated_opsd_v2_guardfix_resume_20260811/guardfix20_resume_opsd_v2_20step_seed43_dev70_turn_credit_analysis_20260811.*`

## Bridge150

Bridge150 采用 10 个 clean chunks 合并。

| Metric | Value |
| --- | ---: |
| EM macro | 0.5317 |
| correct | 81/150 |
| format | 122/150 |
| format rate | 0.8133 |
| avg search | 3.2533 |
| tool failures | 0 |
| duplicate query trajectories | 8/150 |
| max-search no-answer | 27/150 |
| too many search no gain | 55/150 |

按数据源：

| Source | EM | Correct |
| --- | ---: | ---: |
| 2wikimultihopqa | 0.5600 | 56/100 |
| bamboogle | 0.7000 | 7/10 |
| hotpotqa | 0.4667 | 14/30 |
| musique | 0.4000 | 4/10 |

Offline diagnostics:

| Metric | Value |
| --- | ---: |
| wrong valid | 41 |
| possible alias match | 3 |
| missing follow-up query | 2 |
| helpful follow-up query | 130 |
| bad max-search loop | 8 |
| multi-candidate answer | 0 |
| answer granularity miss | 0 |

Turn-credit analysis:

| Metric | Value |
| --- | ---: |
| evidence candidate records | 129 |
| evidence training credit turns | 56 |
| final-hop candidate records | 122 |
| final-hop training credit turns | 51 |
| training credit records | 35 |
| early answer penalty records | 3 |
| missing final-hop penalty records | 2 |
| final answer guard penalty records | 27 |

输出：

- `my-search-r1/eval_results/gated_opsd_v2_guardfix_resume_20260811/guardfix20_resume_opsd_v2_20step_seed43_bridge150_20260811.*`
- `my-search-r1/eval_results/gated_opsd_v2_guardfix_resume_20260811/guardfix20_resume_opsd_v2_20step_seed43_bridge150_offline_diagnostics_20260811.*`
- `my-search-r1/eval_results/gated_opsd_v2_guardfix_resume_20260811/guardfix20_resume_opsd_v2_20step_seed43_bridge150_turn_credit_analysis_20260811.*`

## Comparison To 5-Step

| Model | dev70 EM | dev70 Correct | dev70 Format | bridge150 EM | bridge150 Correct | bridge150 Format | bridge150 Avg Search |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| OPSD v2 5-step seed42 | 0.4857 | 34/70 | 0.9857 | 0.5242 | 87/150 | 0.9067 | 3.1400 |
| OPSD v2 20-step seed43 | 0.4571 | 32/70 | 1.0000 | 0.5317 | 81/150 | 0.8133 | 3.2533 |

Bridge150 的 EM macro 对 20-step seed43 更高，但 correct 数更低、format 明显更差、平均搜索更高。这说明宏平均被数据源分布影响，不能只看 EM macro。若选择最终 checkpoint，5-step seed42 仍更稳：dev70 更高，bridge150 correct 更多，format 更好。

## Decision

- 不建议用 20-step seed43 替代 5-step seed42。
- 20-step seed43 可作为方差对照：它证明 20-step 能保持 clean dev70 format，但不能稳定提升 correct，并且 bridge150 format/max-search 风险更高。
- 5-step “步数少”的质疑可以用本轮结果回应：更多 step 并没有带来更可靠的综合收益，反而放大 bridge format 和 max-search no-answer。
- 下一步若继续补证据，应优先跑 5-step alias80，而不是继续加训 20-step。
