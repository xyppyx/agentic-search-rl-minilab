# Final-Hop Guard-Fix 20-Step 训练复盘

记录时间：2026-08-06 CST

## 目标

按用户要求，将 `turn-credit-final-hop-guardfix-5step-20260805` 的 guard-fix 配置扩展到 20-step 训练，观察 longer run 是否继续改善 dev70 与 bridge targeted 表现。

公开复盘不记录 PyTRIO 远端 sampler weights URI。首次 dev70 评测未通过工具门槛；随后重试 dev70 已通过工具门槛。`bridge_eval_150` full run 未通过工具门槛，随后对 3 条工具失败样本单独重跑并生成 patched 版本。

## 参数

训练数据：`my-search-r1/datasets/train.jsonl`

训练命令：

```bash
PYTHONPATH=my-search-r1 uv run python my-search-r1/scripts/train_pytrio.py \
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
  --run-name turn-credit-final-hop-guardfix-20step-20260806 \
  --trajectory-output-dir my-search-r1/outputs/train_pytrio \
  --turn-credit-policy final_hop_bridge \
  --evidence-search-turn-bonus 0.05 \
  --final-hop-search-turn-bonus 0.10 \
  --early-answer-turn-penalty 0.05 \
  --missing-final-hop-turn-penalty 0.08 \
  --final-answer-guard-turn-penalty 0.06
```

## Health Check

训练前运行 Zhihu dev-5 health：

| 指标 | 值 |
| --- | ---: |
| trajectories | 5 |
| EM | 0.4000 |
| format | 0.8000 |
| avg search | 3.0000 |
| Zhihu requests | 15 |
| Zhihu success rate | 1.0000 |
| error/timeout/rate-limit | 0 |

Health check 达标，因此进入正式训练。

## 训练结果

20/20 step 完成，保存 step 5/10/15/20/final。step 14 和 step 20 因可训练 datum/advantage 条件跳过参数更新；其余 step 进入 optimizer。训练阶段未观察到 Zhihu 429、timeout、credential/http error 或 tool failure。

训练 JSONL 聚合：

| 指标 | 值 |
| --- | ---: |
| trajectories | 160 |
| correct | 50/160 |
| correct rate | 0.3125 |
| valid format | 143/160 |
| format rate | 0.8938 |
| avg search | 1.8438 |
| mean reward | 0.3019 |
| tool failures | 0 |

Turn-credit label 聚合：

| Label | Count |
| --- | ---: |
| `evidence_bridge_search` | 31 |
| `final_hop_attribute_search` | 18 |
| `early_answer_missing_followup` | 4 |
| `missing_final_hop_attribute` | 7 |
| `final_answer_guard` | 16 |

相比 guard-fix 5-step 训练，20-step 中 `missing_final_hop_attribute` 和 `final_answer_guard` 继续有真实训练命中，说明新增信号没有失活。

## Dev70 Eval - First Attempt

产物：

- `my-search-r1/eval_results/final_hop_bridge_20260806/guardfix_20step_dev_20260806.jsonl`
- `my-search-r1/eval_results/final_hop_bridge_20260806/guardfix_20step_dev_20260806.md`
- `my-search-r1/eval_results/final_hop_bridge_20260806/guardfix_20step_dev_20260806_offline_diagnostics.md`
- `my-search-r1/eval_results/final_hop_bridge_20260806/guardfix_20step_dev_turn_credit_analysis_20260806.md`

本轮 dev70 eval 未通过工具门槛：

| 工具指标 | 值 |
| --- | ---: |
| Zhihu requests | 135 |
| Zhihu success rate | 0.9407 |
| error rate | 0.0593 |
| timeout/rate-limit | 0 |
| failed trajectories | 3 |
| failed tool events | 8 |
| observed error type | `url_error` |

失败样本为 `dev_6965`、`dev_3822` 和 `dev_692`。例如 `dev_6965` 中 3 个查询返回 `URLError` / `url_error`。因此以下指标只能作为参考，不能作为正式模型效果结论：

| 指标 | 参考值 |
| --- | ---: |
| EM macro | 0.4571 |
| correct | 32/70 |
| format | 0.9857 |
| avg search | 1.9286 |
| max-search no-answer rate | 0.0143 |
| bad max-search loop rate | 0.0429 |
| too many search no gain rate | 0.2143 |

参考 offline diagnostics：

| 指标 | 值 |
| --- | ---: |
| wrong-valid | 37 |
| helpful follow-up | 31 |
| missing follow-up | 0 |
| possible alias match | 6 |
| answer granularity miss | 0 |
| multi-candidate answer | 0 |
| bad max-search loop | 3 |

参考 turn-credit analysis：

| 指标 | 值 |
| --- | ---: |
| evidence candidate records | 32 |
| evidence training credit turns | 22 |
| final-hop candidate records | 22 |
| final-hop training credit turns | 15 |
| early-answer penalty records | 0 |
| missing final-hop penalty records | 1 |
| final-answer guard penalty records | 1 |

## Dev70 Eval - Retry

产物：

- `my-search-r1/eval_results/final_hop_bridge_20260806/guardfix_20step_dev_retry_20260806.jsonl`
- `my-search-r1/eval_results/final_hop_bridge_20260806/guardfix_20step_dev_retry_20260806.md`
- `my-search-r1/eval_results/final_hop_bridge_20260806/guardfix_20step_dev_retry_20260806_offline_diagnostics.md`
- `my-search-r1/eval_results/final_hop_bridge_20260806/guardfix_20step_dev_retry_turn_credit_analysis_20260806.md`

重试 dev70 eval 通过工具门槛：

| 工具指标 | 值 |
| --- | ---: |
| Zhihu requests | 133 |
| Zhihu success rate | 1.0000 |
| error/timeout/rate-limit | 0 |
| empty rate | 0.0301 |

正式 dev70 retry 指标：

| 指标 | 值 |
| --- | ---: |
| EM macro | 0.4571 |
| correct | 32/70 |
| format | 0.9571 |
| avg search | 1.9000 |
| max-search no-answer rate | 0.0429 |
| bad max-search loop rate | 0.0286 |
| too many search no gain rate | 0.2000 |

Source EM：

| Source | EM |
| --- | ---: |
| 2WikiMultihopQA | 0.5000 |
| Bamboogle | 0.6000 |
| HotpotQA | 0.5000 |
| MuSiQue | 0.4000 |
| NQ | 0.3000 |
| PopQA | 0.3000 |
| TriviaQA | 0.6000 |

Retry offline diagnostics：

| 指标 | 值 |
| --- | ---: |
| wrong-valid | 35 |
| helpful follow-up | 31 |
| missing follow-up | 0 |
| possible alias match | 7 |
| answer granularity miss | 0 |
| multi-candidate answer | 1 |
| bad max-search loop | 2 |

Retry turn-credit analysis：

| 指标 | 值 |
| --- | ---: |
| evidence candidate records | 36 |
| evidence training credit turns | 26 |
| final-hop candidate records | 22 |
| final-hop training credit turns | 15 |
| early-answer penalty records | 0 |
| missing final-hop penalty records | 1 |
| final-answer guard penalty records | 3 |

## Bridge150 Eval

Bridge-5 health 先通过工具门槛：

| 指标 | 值 |
| --- | ---: |
| trajectories | 5 |
| EM macro | 0.8333 |
| format | 1.0000 |
| avg search | 2.0000 |
| Zhihu requests | 10 |
| Zhihu success rate | 1.0000 |
| error/timeout/rate-limit | 0 |

随后运行完整 `bridge_eval_150`。

产物：

- `my-search-r1/eval_results/final_hop_bridge_20260806/guardfix_20step_bridge_health5_20260806.jsonl`
- `my-search-r1/eval_results/final_hop_bridge_20260806/guardfix_20step_bridge_20260806.jsonl`
- `my-search-r1/eval_results/final_hop_bridge_20260806/guardfix_20step_bridge_20260806.md`
- `my-search-r1/eval_results/final_hop_bridge_20260806/guardfix_20step_bridge_20260806_offline_diagnostics.md`
- `my-search-r1/eval_results/final_hop_bridge_20260806/guardfix_20step_bridge_turn_credit_analysis_20260806.md`

Full bridge150 未通过工具门槛：

| 工具指标 | 值 |
| --- | ---: |
| Zhihu requests | 480 |
| Zhihu success rate | 0.9896 |
| error rate | 0.0104 |
| timeout/rate-limit | 0 |
| failed trajectories | 3 |
| failed tool events | 5 |
| observed error type | `url_error` |

失败样本为 `dev_12320`、`dev_12391` 和 `dev_12483`。这些失败仍是外部 Zhihu `URLError` / `url_error`，因此 full run 不能进入正式 targeted 指标表。参考指标为 EM macro 0.5092、correct 81/150、format 0.8200、平均搜索 3.2000。

## Bridge150 Patched

按前一轮 `dev_7742` 的 patched 协议，单独抽出 3 条工具失败样本重跑：

| 指标 | 值 |
| --- | ---: |
| trajectories | 3 |
| correct | 2/3 |
| format | 1.0000 |
| avg search | 4.0000 |
| Zhihu requests | 12 |
| Zhihu success rate | 1.0000 |
| error/timeout/rate-limit | 0 |

随后用 3 条 retry 记录替换 full bridge150 中对应失败记录，生成 patched 版本。

patched 产物：

- `my-search-r1/eval_results/final_hop_bridge_20260806/guardfix_20step_bridge_failed3_20260806_input.jsonl`
- `my-search-r1/eval_results/final_hop_bridge_20260806/guardfix_20step_bridge_failed3_retry_20260806.jsonl`
- `my-search-r1/eval_results/final_hop_bridge_20260806/guardfix_20step_bridge_patched_20260806.jsonl`
- `my-search-r1/eval_results/final_hop_bridge_20260806/guardfix_20step_bridge_patched_20260806.md`
- `my-search-r1/eval_results/final_hop_bridge_20260806/guardfix_20step_bridge_patched_20260806_offline_diagnostics.md`
- `my-search-r1/eval_results/final_hop_bridge_20260806/guardfix_20step_bridge_patched_turn_credit_analysis_20260806.md`

patched 指标：

| 指标 | Prompt-only base | Evidence-v2 20-step | Guard-fix 5-step patched | Guard-fix 20-step patched |
| --- | ---: | ---: | ---: | ---: |
| EM macro | 0.4750 | 0.4583 | 0.4842 | 0.5142 |
| correct | 74/150 | 81/150 | 80/150 | 83/150 |
| format | 0.7200 | 0.9400 | 0.7933 | 0.8267 |
| avg search | 3.3067 | 3.0933 | 3.2200 | 3.2000 |
| tool failures | 0 | 0 | 0 | 0 |

patched source EM：

| Source | Guard-fix 20-step patched |
| --- | ---: |
| 2WikiMultihopQA | 0.5900 |
| Bamboogle | 0.6000 |
| HotpotQA | 0.4667 |
| MuSiQue | 0.4000 |

Gained/lost：

| Comparison | Gained | Lost | Net |
| --- | ---: | ---: | ---: |
| vs prompt-only base | 12 | 3 | +9 |
| vs evidence-v2 20-step | 11 | 9 | +2 |
| vs guard-fix 5-step patched | 5 | 2 | +3 |

patched offline diagnostics：

| 指标 | 值 |
| --- | ---: |
| wrong-valid | 41 |
| helpful follow-up | 127 |
| missing follow-up | 3 |
| possible alias match | 4 |
| answer granularity miss | 0 |
| multi-candidate answer | 0 |
| bad max-search loop | 8 |

patched turn-credit analysis：

| 指标 | 值 |
| --- | ---: |
| evidence candidate records | 129 |
| evidence training credit turns | 52 |
| final-hop candidate records | 123 |
| final-hop training credit turns | 49 |
| early-answer penalty records | 3 |
| missing final-hop penalty records | 1 |
| final-answer guard penalty records | 25 |

## 结论

训练本身完成且可复盘：20-step guard-fix 生成 160 条训练 trajectory，训练阶段 tool failures 为 0，并保存 final weights。

首次 final dev70 eval 因 Zhihu `url_error` 导致 success rate 只有 0.9407，未通过工具门槛；重试 dev70 eval 的 Zhihu success rate 为 1.0，正式指标为 EM macro 0.4571、format 0.9571、平均搜索 1.9000。

相对 guard-fix 5-step dev70 的 EM 0.4286、format 0.9857、平均搜索 1.8000，20-step retry 提升了 EM，但 format 和搜索效率略退；相对 evidence-v2 20-step dev70 的 EM 0.4429、format 1.0000、平均搜索 1.7714，20-step guard-fix EM 略高，但 format 与搜索效率仍弱。

Bridge150 full run 未通过工具门槛；patched 版本由 147 条 full run 记录和 3 条失败样本 retry 记录合成，tool failures 为 0，EM macro 0.5142、correct 83/150、format 0.8267、平均搜索 3.2000。该结果显示 guard-fix 20-step 在 bridge EM/correct 上超过 prompt-only base、evidence-v2 20-step 和 guard-fix 5-step patched，但 format 仍明显低于 evidence-v2 20-step 的 0.9400。

更稳妥的表述是：guard-fix 20-step patched 是当前 bridge150 最高 EM/correct 证据，但不是一次独立全量 success rate 1.0 run，且 format/max-search no-answer 仍是进入简历核心指标前的主要短板。
