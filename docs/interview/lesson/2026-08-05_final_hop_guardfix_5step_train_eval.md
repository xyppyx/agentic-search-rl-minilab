# Final-Hop Guard-Fix 5-Step 训练评测复盘

记录时间：2026-08-05 CST

## 目标

在 `turn_credit_final_hop_bridge_v3_guard_fix` 代码上运行一次真实 Zhihu 5-step 小预算训练，验证新增 final-answer guard 与修正后的 missing-final-hop detector 是否改善上轮 bridge format/max-search 问题。

公开复盘不记录 PyTRIO 远端 sampler weights URI。`bridge_eval_150.jsonl` 只作为评测集，不进入训练。

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
  --seed 42 \
  --temperature 1.0 \
  --top-p 1.0 \
  --swanlab-mode disabled \
  --save-every 0 \
  --run-name turn-credit-final-hop-guardfix-5step-20260805 \
  --trajectory-output-dir my-search-r1/outputs/train_pytrio \
  --turn-credit-policy final_hop_bridge \
  --evidence-search-turn-bonus 0.05 \
  --final-hop-search-turn-bonus 0.10 \
  --early-answer-turn-penalty 0.05 \
  --missing-final-hop-turn-penalty 0.08 \
  --final-answer-guard-turn-penalty 0.06
```

评测使用训练后的 final sampler weights，公开文档中记为 `<final-sampler-weights-uri>`。

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

5/5 step 完成，均进入 optimizer；训练阶段未观察到 Zhihu 429、timeout、credential/http error 或 tool failure。

| Step | mean reward | correct rate | avg search | input tokens | loss tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.362 | 0.375 | 2.12 | 19398 | 662 |
| 2 | 0.000 | 0.000 | 1.62 | 10698 | 462 |
| 3 | 0.237 | 0.250 | 2.25 | 19003 | 526 |
| 4 | -0.050 | 0.000 | 3.75 | 22889 | 1148 |
| 5 | 0.000 | 0.000 | 1.00 | 5376 | 54 |

训练 JSONL 聚合：

| 指标 | 值 |
| --- | ---: |
| trajectories | 40 |
| correct | 5/40 |
| valid format | 34/40 |
| avg search | 2.1500 |
| tool events | 86 |
| tool success | 86 |
| tool failures | 0 |
| `evidence_bridge_search` labels | 16 |
| `final_hop_attribute_search` labels | 9 |
| `early_answer_missing_followup` labels | 4 |
| `missing_final_hop_attribute` labels | 2 |
| `final_answer_guard` labels | 6 |

相比上一轮 v3 训练，`missing_final_hop_attribute` 从 0 变为 2，`final_answer_guard` 命中 6，说明新增/修正的 turn-level 训练信号在真实训练轨迹中生效。

## Dev70 Eval

产物：

- `my-search-r1/eval_results/final_hop_bridge_20260805/guardfix_5step_dev_20260805.jsonl`
- `my-search-r1/eval_results/final_hop_bridge_20260805/guardfix_5step_dev_20260805.md`
- `my-search-r1/eval_results/final_hop_bridge_20260805/guardfix_5step_dev_20260805_offline_diagnostics.md`

| 指标 | 值 |
| --- | ---: |
| EM macro | 0.4286 |
| correct | 30/70 |
| format | 0.9857 |
| avg search | 1.8000 |
| max-search no-answer rate | 0.0143 |
| bad max-search loop rate | 0.0143 |
| duplicate query rate | 0.0000 |
| Zhihu requests | 126 |
| Zhihu success rate | 1.0000 |
| missing follow-up | 0 |
| possible alias match | 8 |
| answer granularity miss | 0 |
| multi-candidate answer | 1 |

Dev70 是有效 eval，且通过不明显退化门槛：EM 高于 0.4143，format 高于 0.9000。

## Bridge150 Eval

产物：

- `my-search-r1/eval_results/final_hop_bridge_20260805/guardfix_5step_bridge_20260805.jsonl`
- `my-search-r1/eval_results/final_hop_bridge_20260805/guardfix_5step_bridge_20260805.md`
- `my-search-r1/eval_results/final_hop_bridge_20260805/guardfix_5step_bridge_20260805_offline_diagnostics.md`
- `my-search-r1/eval_results/final_hop_bridge_20260805/guardfix_5step_bridge_turn_credit_analysis_20260805.md`

本轮 bridge150 eval 未通过工具门槛：

| 工具指标 | 值 |
| --- | ---: |
| Zhihu requests | 483 |
| Zhihu success rate | 0.9855 |
| error rate | 0.0145 |
| timeout/rate-limit | 0 |
| failed tool events | 7 |
| error type | `url_error` |

因此 bridge150 不进入正式模型效果比较。以下指标只能作为参考：

| 指标 | 值 |
| --- | ---: |
| EM macro | 0.5042 |
| correct | 79/150 |
| format | 0.7933 |
| avg search | 3.2200 |
| max-search no-answer rate | 0.1933 |
| bad max-search loop rate | 0.0467 |
| duplicate query rate | 0.0400 |
| missing follow-up | 2 |
| possible alias match | 5 |
| answer granularity miss | 0 |

参考 source EM：

| Source | EM |
| --- | ---: |
| 2WikiMultihopQA | 0.5500 |
| Bamboogle | 0.7000 |
| HotpotQA | 0.4667 |
| MuSiQue | 0.3000 |

Turn-credit 离线分析参考：

| 指标 | 值 |
| --- | ---: |
| wrong-valid records | 40 |
| evidence training credit turns | 51 |
| final-hop training credit turns | 48 |
| early-answer penalty records | 2 |
| missing final-hop penalty records | 1 |
| final-answer guard penalty records | 29 |

## 结论

本轮训练和 dev70 eval 有效。Guard fix 的训练信号确实进入真实训练轨迹：`final_answer_guard=6`，`missing_final_hop_attribute=2`。Dev70 保持 EM 0.4286，并把 format 提到 0.9857。

Bridge150 eval 未达 Zhihu success rate 1.0，不能作为正式 bridge 结论。参考指标显示 EM macro 0.5042、correct 79/150、format 0.7933；这相对上轮 final-hop v3 的 valid bridge 结果有 format 参考提升，但仍低于 0.9000 门槛，并且工具失败使该比较无效。

下一步如果要给出正式 bridge 结论，应在 Zhihu health 正常后重跑 bridge150 eval；如果仍出现 `url_error` 或 success rate < 1.0，应暂停 targeted bridge 结论，只保留 dev70 和训练信号结果。

## Key2 Rerun Attempt

用户要求使用另一个 Zhihu API key 重跑 `bridge_eval_150`。本轮先强制只使用第二个 key 运行 bridge-5 health：

```bash
ZHIHU_SEARCH_KEYS="$ZHIHU_API_KEY2" ZHIHU_API_KEY= ZHIHU_API_KEY2= \
PYTHONPATH=my-search-r1 uv run python my-search-r1/scripts/eval_pytrio.py \
  --model-path <final-sampler-weights-uri> \
  --data my-search-r1/datasets/bridge_eval_150.jsonl \
  --backend zhihu_search \
  --limit 5 \
  --batch-size 1 \
  --seed 42 \
  --temperature 0.0 \
  --top-p 1.0 \
  --jsonl-output my-search-r1/eval_results/final_hop_bridge_20260805/guardfix_5step_bridge_key2_health5_20260805.jsonl \
  --report-output my-search-r1/eval_results/final_hop_bridge_20260805/guardfix_5step_bridge_key2_health5_20260805.md
```

结果：

| 指标 | 值 |
| --- | ---: |
| trajectories | 5 |
| Zhihu requests | 14 |
| Zhihu success rate | 0.0000 |
| rate-limit rate | 1.0000 |
| error/timeout | 0 |

第二个 key 已被限流，未通过 health 门槛，因此没有继续运行完整 bridge150。当前仍没有 guard-fix bridge150 的正式有效结果。

## Key1 Rerun Attempt

用户随后要求使用原本的 Zhihu API key 重跑 `bridge_eval_150`。本轮先强制只使用原 key 运行 bridge-5 health，health 达标后运行完整 bridge150。

Health 结果：

| 指标 | 值 |
| --- | ---: |
| trajectories | 5 |
| Zhihu requests | 10 |
| Zhihu success rate | 1.0000 |
| error/timeout/rate-limit | 0 |

完整 bridge150 rerun 结果未通过工具门槛：

| 工具指标 | 值 |
| --- | ---: |
| Zhihu requests | 483 |
| Zhihu success rate | 0.9979 |
| error rate | 0.0021 |
| failed tool events | 1 |
| failed sample | `dev_7742` |
| error type | `parse_error` |

因此本轮仍不进入正式模型效果比较。以下指标只能作为参考：

| 指标 | 值 |
| --- | ---: |
| EM macro | 0.4817 |
| correct | 79/150 |
| format | 0.7867 |
| avg search | 3.2200 |
| max-search no-answer rate | 0.2000 |
| bad max-search loop rate | 0.0400 |
| duplicate query rate | 0.0333 |
| missing follow-up | 2 |
| possible alias match | 5 |
| answer granularity miss | 0 |

参考 source EM：

| Source | EM |
| --- | ---: |
| 2WikiMultihopQA | 0.5600 |
| Bamboogle | 0.6000 |
| HotpotQA | 0.4667 |
| MuSiQue | 0.3000 |

Turn-credit 离线分析参考：

| 指标 | 值 |
| --- | ---: |
| wrong-valid records | 39 |
| evidence training credit turns | 49 |
| final-hop training credit turns | 45 |
| early-answer penalty records | 2 |
| missing final-hop penalty records | 1 |
| final-answer guard penalty records | 30 |

到目前为止，guard-fix checkpoint 仍没有 success rate 1.0 的正式 bridge150 结果；不能把参考 EM macro 0.4817 或 0.5042 写作正式超过 base。

## Single-Case Patch For `dev_7742`

用户要求先忽略 key1 rerun 中唯一一个 Zhihu `parse_error`，分析 eval 结论，并确认是否可以单独重跑错误 case 以避免全量重跑。

当前 `eval_pytrio.py` 没有 `--example-id` 或 `--offset` 参数，但可以从 `bridge_eval_150.jsonl` 抽出对应一行，构造临时 1 条 JSONL，用同一个 final sampler weights 单独评测该样本。公开复盘不记录真实 sampler URI 或 API key。

单样本输入：

| 字段 | 值 |
| --- | --- |
| id | `dev_7742` |
| data_source | `2wikimultihopqa` |
| question | `Which film has the director who is older, Lukket Avdeling or The Heart Of St. Pauli?` |
| gold | `Heart of St. Pauli` / `The Heart Of St. Pauli` / `The Heart of St. Pauli` |

单样本补跑结果：

| 指标 | 值 |
| --- | ---: |
| trajectories | 1 |
| Zhihu requests | 4 |
| Zhihu success rate | 1.0000 |
| exact match | 1 |
| format | 1.0000 |
| search calls | 4 |

补跑 query 序列：

1. `Lukket Avdeling film director`
2. `The Heart Of St. Pauli film director`
3. `Eugen York director birth date`
4. `Arnljot Berg director birth date`

最终答案为 `Answer: The Heart Of St. Pauli`。这说明原 full bridge key1 rerun 中的 `dev_7742` 失败是外部工具 parse error 导致的无效轨迹，不是模型策略在该样本上的必然失败。

随后生成 patched 版本：只用单样本补跑记录替换 `guardfix_5step_bridge_key1_rerun_20260805.jsonl` 中的第 20 条，其余 149 条保持不变。

相关产物：

- `my-search-r1/eval_results/final_hop_bridge_20260805/guardfix_5step_bridge_key1_rerun_dev_7742_single_20260805.jsonl`
- `my-search-r1/eval_results/final_hop_bridge_20260805/guardfix_5step_bridge_key1_rerun_patched_20260805.jsonl`
- `my-search-r1/eval_results/final_hop_bridge_20260805/guardfix_5step_bridge_key1_rerun_patched_20260805.md`
- `my-search-r1/eval_results/final_hop_bridge_20260805/guardfix_5step_bridge_key1_rerun_patched_20260805_offline_diagnostics.md`
- `my-search-r1/eval_results/final_hop_bridge_20260805/guardfix_5step_bridge_key1_rerun_patched_turn_credit_analysis_20260805.md`

patched 指标：

| 指标 | Prompt-only base | Guard-fix patched |
| --- | ---: | ---: |
| EM macro | 0.4750 | 0.4842 |
| correct | 74/150 | 80/150 |
| format | 0.7200 | 0.7933 |
| avg search | 3.3067 | 3.2200 |
| tool failures | 0 | 0 |
| gained/lost vs base | - | 9/3 |

patched source EM：

| Source | Base | Guard-fix patched |
| --- | ---: | ---: |
| 2WikiMultihopQA | 0.5000 | 0.5700 |
| Bamboogle | 0.6000 | 0.6000 |
| HotpotQA | 0.5000 | 0.4667 |
| MuSiQue | 0.3000 | 0.3000 |

patched offline diagnostics：

| 指标 | 值 |
| --- | ---: |
| wrong-valid | 39 |
| helpful follow-up | 125 |
| missing follow-up | 2 |
| possible alias match | 5 |
| answer granularity miss | 0 |
| bad max-search loop | 6 |

patched turn-credit analysis：

| 指标 | 值 |
| --- | ---: |
| evidence training credit turns | 49 |
| final-hop training credit turns | 45 |
| early-answer penalty records | 2 |
| missing final-hop penalty records | 1 |
| final-answer guard penalty records | 29 |

结论：如果接受“仅补跑唯一工具失败样本”的 patched 评测协议，guard-fix checkpoint 在 bridge150 上超过 prompt-only base：EM macro 0.4842 > 0.4750，correct 80/150 > 74/150，平均搜索也略低。但它仍没有通过原计划的完整门槛，因为 format 只有 0.7933，低于 0.9000；并且 patched run 是由 149 条原 full rerun + 1 条单样本 rerun 合成，不等同于一次全量 success rate 1.0 的独立 run。更稳妥的表述是：guard-fix 在 bridge targeted 上呈现小幅 EM/correct 正向，但最终可作为简历核心指标前，仍需要解决 format/max-search no-answer。
