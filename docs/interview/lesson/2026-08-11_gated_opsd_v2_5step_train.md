# Gated OPSD v2 5-Step 训练复盘

记录时间：2026-08-11 CST

## 目标

在 OPSD v2 正向 gate 实现后，运行真实 Zhihu 5-step 训练，验证 `credited_turns + positive_advantage + opsd_coef=0.01` 是否能修复 v1 的 format/search 退化。

公开复盘不记录 PyTRIO 远端 sampler weights URI、真实 API key、SwanLab 私有链接或本地敏感配置。

## 参数

训练数据：`my-search-r1/datasets/train.jsonl`

训练命令：

```bash
PYTHONPATH=my-search-r1 uv run python my-search-r1/scripts/train_pytrio.py \
  --data my-search-r1/datasets/train.jsonl \
  --max-steps 5 \
  --questions-per-batch 2 \
  --group-size 4 \
  --backend zhihu_search \
  --env-file my-search-r1/.env \
  --seed 42 \
  --temperature 1.0 \
  --top-p 1.0 \
  --swanlab-mode disabled \
  --save-every 5 \
  --run-name gated-opsd-v2-guardfix-5step-20260811 \
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

评测使用训练后的 final sampler weights，公开文档中记为 `<final-sampler-weights-uri>`。

## Health Check

训练前运行 Zhihu dev-5 health：

| 指标 | 值 |
| --- | ---: |
| trajectories | 5 |
| EM | 0.2000 |
| format | 0.6000 |
| avg search | 3.0000 |
| Zhihu requests | 15 |
| Zhihu success rate | 1.0000 |
| error/timeout/rate-limit | 0 |

Health check 通过工具门槛，因此进入真实训练。

## 训练结果

5/5 step 完成，保存 step-5 与 final。训练阶段未观察到 PyTRIO sampling 阻塞，也未观察到 Zhihu 429、timeout、credential/http error 或 tool failure。

逐 step 训练指标：

| Step | mean reward | correct rate | format | avg search | loss mean | OPSD masked tokens | OPSD mask rate | student-teacher gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.3625 | 0.3750 | 0.8750 | 2.2500 | 0.0083 | 182 | 0.0093 | 0.0044 |
| 2 | 0.0000 | 0.0000 | 1.0000 | 1.5000 | -0.0004 | 138 | 0.0163 | 0.0048 |
| 3 | 0.2375 | 0.2500 | 0.8750 | 2.3750 | 0.0033 | 191 | 0.0095 | 0.0020 |
| 4 | 0.0750 | 0.1250 | 0.5000 | 3.8750 | 0.0020 | 163 | 0.0070 | 0.0086 |
| 5 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0004 | 0 | 0.0000 | - |

训练 JSONL 聚合：

| 指标 | 值 |
| --- | ---: |
| trajectories | 40 |
| correct | 6/40 |
| valid format | 34/40 |
| avg search | 2.2000 |
| Zhihu requests | 88 |
| Zhihu success rate | 1.0000 |
| tool failures | 0 |
| avg OPSD masked tokens / step | 134.8 |
| avg OPSD mask rate | 0.0084 |
| avg student-teacher gap | 0.0049 |

Turn-credit 聚合：

| Label/Metric | Count |
| --- | ---: |
| `evidence_bridge_search` turns | 15 |
| `final_hop_attribute_search` turns | 6 |
| `early_answer_missing_followup` turns | 4 |
| `missing_final_hop_attribute` turns | 1 |
| `final_answer_guard` turns | 6 |
| credited trajectories | 24 |
| credited tokens | 963 |

相比 v1，v2 的 OPSD mask 更窄：平均 masked tokens 从 314.2 降到 134.8，平均 mask rate 从 0.0203 降到 0.0084。训练 correct 从 5/40 到 6/40，但 format 仍为 34/40，step 4 仍出现 format 0.5 与 avg search 3.875 的风险信号。

## Dev70 Eval

产物：

- `my-search-r1/eval_results/gated_opsd_v2_20260811/gated_opsd_v2_pretrain_health_dev5_20260811.jsonl`
- `my-search-r1/eval_results/gated_opsd_v2_20260811/gated_opsd_v2_guardfix_5step_dev5_20260811.jsonl`
- `my-search-r1/eval_results/gated_opsd_v2_20260811/gated_opsd_v2_guardfix_5step_dev70_20260811.jsonl`
- `my-search-r1/eval_results/gated_opsd_v2_20260811/gated_opsd_v2_guardfix_5step_dev70_20260811.md`
- `my-search-r1/eval_results/gated_opsd_v2_20260811/gated_opsd_v2_guardfix_5step_dev70_offline_diagnostics_20260811.md`
- `my-search-r1/eval_results/gated_opsd_v2_20260811/gated_opsd_v2_guardfix_5step_dev70_turn_credit_analysis_20260811.md`

训练后 dev-5：

| 指标 | 值 |
| --- | ---: |
| EM | 0.4000 |
| format | 0.6000 |
| avg search | 2.8000 |
| Zhihu requests | 14 |
| Zhihu success rate | 1.0000 |
| error/timeout/rate-limit | 0 |

Dev70 有效评测：

| 指标 | 值 |
| --- | ---: |
| trajectories | 70 |
| EM macro | 0.4143 |
| correct | 29/70 |
| format | 0.9143 |
| avg search | 1.9714 |
| no-search rate | 0.0000 |
| duplicate query rate | 0.0286 |
| max-search no-answer rate | 0.0714 |
| bad max-search loop rate | 0.0286 |
| Zhihu requests | 138 |
| Zhihu success rate | 1.0000 |
| error/timeout/rate-limit | 0 |

Offline diagnostics：

| 指标 | 值 |
| --- | ---: |
| missing follow-up query | 0 |
| answer granularity miss | 0 |
| bad max-search loop | 2 |
| possible alias match | 9 |
| multi-candidate answer | 1 |
| wrong valid | 35 |

Turn-credit analysis：

| 指标 | 值 |
| --- | ---: |
| evidence candidate records | 38 |
| evidence training credit turns | 27 |
| final-hop candidate records | 22 |
| final-hop training credit turns | 15 |
| training credit records | 19 |
| early-answer penalty records | 0 |
| missing-final-hop penalty records | 0 |
| final-answer guard penalty records | 6 |

## 对比

| Run | EM | Correct | Format | Avg search | bad max-search loop | 备注 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| guard-fix 5-step | 0.4286 | 30/70 | 0.9857 | 1.8000 | 1 | 非 OPSD 5-step 对照 |
| OPSD v1 5-step | 0.4286 | 30/70 | 0.9286 | 1.9143 | 1 | `final_and_credited`, coef 0.05 |
| OPSD v2 5-step | 0.4143 | 29/70 | 0.9143 | 1.9714 | 2 | `credited_turns + positive_advantage`, coef 0.01 |
| evidence-v2 20-step | 0.4429 | - | 1.0000 | 1.7714 | 3 | 当前最高 format checkpoint |
| guard-fix 20-step retry | 0.4571 | 32/70 | 0.9571 | 1.9000 | 2 | 当前最高 dev70 EM |

## 结论

OPSD v2 成功收窄了蒸馏范围，mask rate 从 v1 的 0.0203 降到 0.0084，且训练阶段工具链稳定。但 dev70 结果更弱：EM 从 v1 的 0.4286 降到 0.4143，format 从 0.9286 降到 0.9143，平均搜索从 1.9143 升到 1.9714。

这说明当前 OPSD 的主要问题不只是 v1 gate 太宽；即使只蒸馏正向 credited turn，辅助 token-likelihood 信号也没有带来更好的策略更新。合理推断是：same-context self-teacher 没有提供高质量新信息，OPSD 与 GRPO/turn-credit 的行为目标仍存在冲突，尤其在小样本 on-policy 训练中会干扰 format 和停止策略。

本轮不继续扩大 OPSD 到 20-step、bridge150 或 alias80。OPSD 可作为“工程可行但当前收益不足”的失败分支保留。后续除非改成更本质的 teacher 设计，例如离线 correct trajectory teacher、answer-span 级别蒸馏或 preference-filtered replay，否则不建议继续在 same-context on-policy OPSD 上消耗真实搜索预算。
