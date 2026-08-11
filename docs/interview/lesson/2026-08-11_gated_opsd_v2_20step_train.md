# Gated OPSD v2 20-Step Training

日期：2026-08-11

## 目标

用户要求在 OPSD v2 已停放建议之后继续训练 20 step。本次实验作为压力测试和反证 run：验证 `credited_turns + positive_advantage` 的 same-context OPSD 在更长步数下是否能追回 5-step 的退化。

本次仍不把 OPSD 作为主 loss。GRPO/turn-credit 继续主导，OPSD 仅以小系数辅助。

## 配置

- run name: `gated-opsd-v2-guardfix-20step-20260811`
- data: `my-search-r1/datasets/train.jsonl`
- eval data: `my-search-r1/datasets/dev.jsonl`
- backend: `zhihu_search`
- seed: `42`
- max steps: `20`
- questions per batch: `2`
- group size: `4`
- train trajectories: `160`
- turn credit policy: `final_hop_bridge`
- OPSD: `--opsd-coef 0.01 --opsd-mask-policy credited_turns --opsd-positive-policy positive_advantage --opsd-min-teacher-logprob -3.0`
- checkpoints: step 5/10/15/20 and final checkpoint saved remotely by PyTRIO；公开文档不记录远程 sampler weights URI。

训练命令：

```bash
timeout 14400s env PYTHONPATH=my-search-r1 uv run python my-search-r1/scripts/train_pytrio.py \
  --data my-search-r1/datasets/train.jsonl \
  --max-steps 20 \
  --questions-per-batch 2 \
  --group-size 4 \
  --backend zhihu_search \
  --env-file my-search-r1/.env \
  --seed 42 \
  --temperature 1.0 \
  --top-p 1.0 \
  --swanlab-mode disabled \
  --save-every 5 \
  --run-name gated-opsd-v2-guardfix-20step-20260811 \
  --trajectory-output-dir my-search-r1/outputs/train_pytrio \
  --turn-credit-policy final_hop_bridge \
  --evidence-search-turn-bonus 0.05 \
  --final-hop-search-turn-bonus 0.10 \
  --early-answer-turn-penalty 0.05 \
  --missing-final-hop-turn-penalty 0.08 \
  --final-answer-guard-turn-penalty 0.06 \
  --opsd-coef 0.01 \
  --opsd-mask-policy credited_turns \
  --opsd-positive-policy positive_advantage \
  --opsd-min-teacher-logprob -3.0
```

## Pretrain Health

输出：

- `my-search-r1/eval_results/gated_opsd_v2_20260811/gated_opsd_v2_20step_pretrain_health_dev5_20260811.jsonl`
- `my-search-r1/eval_results/gated_opsd_v2_20260811/gated_opsd_v2_20step_pretrain_health_dev5_20260811.md`

指标：

| Metric | Value |
| --- | ---: |
| trajectories | 5 |
| EM | 0.4000 |
| correct | 2/5 |
| format | 0.8000 |
| avg search | 3.0000 |
| Zhihu requests | 15 |
| Zhihu success rate | 1.0000 |
| tool failures | 0 |

## Training Result

训练 20/20 step 完成，耗时约 11.5 分钟。训练阶段 Zhihu 工具成功率为 1.0000，未出现工具失败。

| Metric | Value |
| --- | ---: |
| trajectories | 160 |
| correct | 52/160 |
| valid format | 144/160 |
| avg search | 2.0188 |
| mean reward | 0.3150 |
| mean correct rate | 0.3250 |
| mean format rate | 0.9000 |
| Zhihu requests | 323 |
| Zhihu success rate | 1.0000 |
| empty result rate | 0.0310 |
| tool failures | 0 |
| avg OPSD masked tokens / step | 121.45 |
| avg OPSD mask rate | 0.0088 |
| avg student-teacher logprob gap | 0.0101 |

Turn-credit 训练信号汇总：

| Signal | Count |
| --- | ---: |
| evidence bridge search turns | 45 |
| final-hop attribute search turns | 16 |
| early answer penalty turns | 4 |
| missing final-hop penalty turns | 5 |
| final answer guard penalty turns | 15 |
| credited trajectories | 67 |
| credited tokens | 3566 |

## Final Dev-5 Eval

输出：

- `my-search-r1/eval_results/gated_opsd_v2_20260811/gated_opsd_v2_guardfix_20step_dev5_20260811.jsonl`
- `my-search-r1/eval_results/gated_opsd_v2_20260811/gated_opsd_v2_guardfix_20step_dev5_20260811.md`

| Metric | Value |
| --- | ---: |
| trajectories | 5 |
| EM | 0.4000 |
| correct | 2/5 |
| format | 1.0000 |
| avg search | 2.8000 |
| Zhihu requests | 14 |
| Zhihu success rate | 1.0000 |
| tool failures | 0 |

## Final Dev70 Eval

输出：

- `my-search-r1/eval_results/gated_opsd_v2_20260811/gated_opsd_v2_guardfix_20step_dev70_20260811.jsonl`
- `my-search-r1/eval_results/gated_opsd_v2_20260811/gated_opsd_v2_guardfix_20step_dev70_20260811.md`

这次 dev70 有 1 次 Zhihu API parse error，success rate 为 0.9930，未满足项目 `success rate == 1.0` 的正式评测门槛。因此下表只能作为 reference，不进入正式 baseline 表。

| Metric | Value |
| --- | ---: |
| trajectories | 70 |
| EM macro / searched correct rate | 0.4429 |
| correct | 31/70 |
| format | 1.0000 |
| avg search | 2.0286 |
| repeated search trajectories | 1 |
| duplicate query rate | 0.0143 |
| empty observation rate | 0.0141 |
| too many search no gain | 17/70 |
| max-search no-answer | 0/70 |
| Zhihu requests | 142 |
| Zhihu success rate | 0.9930 |
| tool failures | 1 |

失败边界：

- failed record: `test_12420`
- question: `Who is the author of West?`
- failure type: Zhihu API parse error during one search call
- model output: format valid but exact match false

## Diagnostics

离线诊断输出：

- `my-search-r1/eval_results/gated_opsd_v2_20260811/gated_opsd_v2_guardfix_20step_dev70_offline_diagnostics_20260811.jsonl`
- `my-search-r1/eval_results/gated_opsd_v2_20260811/gated_opsd_v2_guardfix_20step_dev70_offline_diagnostics_20260811.md`

| Metric | Value |
| --- | ---: |
| total | 70 |
| wrong valid | 39 |
| possible alias match | 6 |
| answer granularity miss | 0 |
| missing follow-up query | 0 |
| helpful follow-up query | 40 |
| bad max-search loop | 3 |
| multi-candidate answer | 1 |

Turn-credit analysis 输出：

- `my-search-r1/eval_results/gated_opsd_v2_20260811/gated_opsd_v2_guardfix_20step_dev70_turn_credit_analysis_20260811.jsonl`
- `my-search-r1/eval_results/gated_opsd_v2_20260811/gated_opsd_v2_guardfix_20step_dev70_turn_credit_analysis_20260811.md`

| Metric | Value |
| --- | ---: |
| evidence candidate records | 45 |
| evidence candidate turns | 62 |
| evidence training credit turns | 38 |
| final-hop candidate records | 24 |
| final-hop candidate turns | 35 |
| final-hop training credit turns | 19 |
| training credit records | 26 |
| v1-shape candidate turns | 48 |
| v1 training credit turns | 27 |
| early answer penalty records | 0 |
| missing final-hop penalty records | 0 |
| final answer guard penalty records | 0 |

## 对比结论

与 OPSD v2 5-step 相比，20-step reference dev70 从 EM 0.4143 / format 0.9143 / avg search 1.9714 变成 EM 0.4429 / format 1.0000 / avg search 2.0286。更长训练追回了 format，并把 EM 拉到 evidence-v2 20-step 同档，但搜索成本更高，而且 dev70 没有通过工具成功率门槛。

与 turn-credit 主线相比，20-step OPSD v2 仍没有形成明确优势：

- `turn_credit_evidence_bridge_20step`: dev70 EM 0.4429、format 1.0000、avg search 1.7714，且是当前高 format 正式 checkpoint。
- `turn-credit-final-hop-guardfix-20step-20260806`: dev70 retry EM 0.4571、format 0.9571、avg search 1.9000，是当前最高 dev70 EM/correct 探索证据。
- 本次 OPSD v2 20-step reference: EM 0.4429、format 1.0000、avg search 2.0286，但工具 success rate 0.9930，不是正式有效 dev70。

当前判断：20-step 证明 v2 不是完全无效，能改善 5-step 的 format/EM 退化；但 same-context on-policy OPSD 仍像是在给已有策略做轻量自模仿，收益没有超过 turn-credit 主线，且平均搜索更高。除非先补一个 clean dev70 或 patched dev70 口径，否则不应继续扩到 bridge150/alias80。

## 下一步建议

1. 若需要正式数字：只重试 `test_12420` 或重跑 dev70，并把 patched/full 口径分开标注。
2. 若只按方法路线推进：停放 same-context OPSD，转向离线 correct trajectory teacher、preference-filtered replay 或 answer-span 级别蒸馏。
3. 不建议直接扩 bridge150/alias80；当前 dev70 还没有 clean success rate 1.0 证据。
