# Gated OPSD 5-Step 训练复盘

记录时间：2026-08-11 CST

## 目标

在 `exp/gated-opsd` 分支验证 gated on-policy self-distillation 是否能作为 Search-R1 MiniLab 的辅助 loss 跑通真实 PyTRIO 训练，并用 dev70 小预算对照判断是否值得扩到 bridge150 或 alias80。

公开复盘不记录 PyTRIO 远端 sampler weights URI、真实 API key、SwanLab 私有链接或本地敏感配置。

## 参数

OPSD v1 约束：

- context policy：`same_context`
- mask policy：`final_and_credited`
- `opsd_coef=0.05`
- GRPO/turn-credit 仍为主 loss，沿用默认 KL/std 稳定化配置
- 不构造 gold-answer teacher，不蒸馏 tool observation tokens

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
  --run-name gated-opsd-guardfix-5step-20260811 \
  --trajectory-output-dir my-search-r1/outputs/train_pytrio \
  --turn-credit-policy final_hop_bridge \
  --evidence-search-turn-bonus 0.05 \
  --final-hop-search-turn-bonus 0.10 \
  --early-answer-turn-penalty 0.05 \
  --missing-final-hop-turn-penalty 0.08 \
  --final-answer-guard-turn-penalty 0.06 \
  --opsd-coef 0.05 \
  --opsd-mask-policy final_and_credited
```

评测使用训练后的 final sampler weights，公开文档中记为 `<final-sampler-weights-uri>`。

## 验证链路

运行训练前先完成代码与 smoke 验证：

- `PYTHONPATH=my-search-r1 uv run python -m unittest my-search-r1/tests/test_rollout_training.py -v`：41 tests OK。
- local BM25 1-step OPSD train smoke 完成，进入 reference logprobs、OPSD teacher logprobs、custom backward、optimizer 和 checkpoint 路径。

Local smoke 关键指标：

| 指标 | 值 |
| --- | ---: |
| trajectories | 2 |
| correct | 1/2 |
| format | 1.0000 |
| avg search | 1.0000 |
| local BM25 success rate | 1.0000 |
| OPSD masked tokens | 10 |
| OPSD mask rate | 0.0078 |
| student-teacher gap | 0.0020 |

训练前用 local-smoke checkpoint 运行 Zhihu dev-5 health，确认真实搜索链路可用：

| 指标 | 值 |
| --- | ---: |
| trajectories | 5 |
| EM | 0.4000 |
| format | 0.8000 |
| avg search | 3.0000 |
| Zhihu requests | 15 |
| Zhihu success rate | 1.0000 |
| error/timeout/rate-limit | 0 |

## 训练结果

5/5 step 完成，保存 step-5 与 final。训练阶段未观察到 PyTRIO sampling 阻塞，也未观察到 Zhihu 429、timeout、credential/http error 或 tool failure。

逐 step 训练指标：

| Step | mean reward | correct rate | format | avg search | loss mean | OPSD masked tokens | OPSD mask rate | student-teacher gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.3625 | 0.3750 | 0.8750 | 2.3750 | 0.0073 | 357 | 0.0177 | 0.0097 |
| 2 | 0.0000 | 0.0000 | 1.0000 | 1.6250 | -0.0012 | 431 | 0.0398 | 0.0123 |
| 3 | 0.2375 | 0.2500 | 0.8750 | 2.6250 | -0.0001 | 272 | 0.0125 | 0.0039 |
| 4 | -0.0500 | 0.0000 | 0.5000 | 3.8750 | 0.0051 | 443 | 0.0190 | 0.0067 |
| 5 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0003 | 68 | 0.0124 | 0.0069 |

训练 JSONL 聚合：

| 指标 | 值 |
| --- | ---: |
| trajectories | 40 |
| correct | 5/40 |
| valid format | 34/40 |
| avg search | 2.3000 |
| Zhihu requests | 92 |
| Zhihu success rate | 1.0000 |
| tool failures | 0 |
| avg OPSD masked tokens / step | 314.2 |
| avg OPSD mask rate | 0.0203 |
| avg student-teacher gap | 0.0079 |

Turn-credit 聚合：

| Label/Metric | Count |
| --- | ---: |
| `evidence_bridge_search` turns | 17 |
| `final_hop_attribute_search` turns | 9 |
| `early_answer_missing_followup` turns | 4 |
| `missing_final_hop_attribute` turns | 1 |
| `final_answer_guard` turns | 6 |
| credited trajectories | 26 |
| credited tokens | 1327 |

## Dev70 Eval

产物：

- `my-search-r1/eval_results/gated_opsd_20260811/gated_opsd_guardfix_5step_dev5_20260811.jsonl`
- `my-search-r1/eval_results/gated_opsd_20260811/gated_opsd_guardfix_5step_dev5_20260811.md`
- `my-search-r1/eval_results/gated_opsd_20260811/gated_opsd_guardfix_5step_dev70_20260811.jsonl`
- `my-search-r1/eval_results/gated_opsd_20260811/gated_opsd_guardfix_5step_dev70_20260811.md`
- `my-search-r1/eval_results/gated_opsd_20260811/gated_opsd_guardfix_5step_dev70_offline_diagnostics_20260811.md`
- `my-search-r1/eval_results/gated_opsd_20260811/gated_opsd_guardfix_5step_dev70_turn_credit_analysis_20260811.md`

训练后 dev-5：

| 指标 | 值 |
| --- | ---: |
| EM | 0.4000 |
| format | 0.8000 |
| avg search | 3.0000 |
| Zhihu requests | 15 |
| Zhihu success rate | 1.0000 |
| error/timeout/rate-limit | 0 |

Dev70 有效评测：

| 指标 | 值 |
| --- | ---: |
| trajectories | 70 |
| EM macro | 0.4286 |
| correct | 30/70 |
| format | 0.9286 |
| avg search | 1.9143 |
| no-search rate | 0.0000 |
| duplicate query rate | 0.0143 |
| max-search no-answer rate | 0.0571 |
| bad max-search loop rate | 0.0143 |
| Zhihu requests | 134 |
| Zhihu success rate | 1.0000 |
| error/timeout/rate-limit | 0 |

Offline diagnostics：

| 指标 | 值 |
| --- | ---: |
| missing follow-up query | 0 |
| answer granularity miss | 0 |
| bad max-search loop | 1 |
| possible alias match | 8 |
| multi-candidate answer | 1 |
| wrong valid | 35 |

Turn-credit analysis：

| 指标 | 值 |
| --- | ---: |
| evidence candidate records | 36 |
| evidence training credit turns | 24 |
| final-hop candidate records | 21 |
| final-hop training credit turns | 14 |
| training credit records | 16 |
| early-answer penalty records | 1 |
| missing-final-hop penalty records | 0 |
| final-answer guard penalty records | 5 |

## 结论

Gated OPSD v1 的工程链路已通过真实 PyTRIO 训练验证：teacher logprob、OPSD mask、custom backward、checkpoint、Zhihu train/eval 与诊断报告都能跑通。`opsd_coef=0.05` 的 token mask 占比没有失控，训练阶段工具 success rate 为 1.0。

但 5-step OPSD 没有超过当前 turn-credit 主线。相对 `turn-credit-final-hop-guardfix-5step-20260805`，dev70 EM 同为 0.4286，但 format 从 0.9857 降到 0.9286，平均搜索从 1.8000 升到 1.9143。相对 `turn_credit_evidence_bridge_20step` 和 `turn-credit-final-hop-guardfix-20step-20260806`，EM、format 和搜索效率也都不是更强结果。

因此本轮不扩到 bridge150 或 alias80。若继续 OPSD，应先做更保守的 v2，例如降低 `opsd_coef` 到 0.01、加入 `--opsd-min-teacher-logprob`，或调整 mask 只覆盖 final answer 中更短的 answer span；否则应把 OPSD 作为已验证但当前不优的分支停放。
