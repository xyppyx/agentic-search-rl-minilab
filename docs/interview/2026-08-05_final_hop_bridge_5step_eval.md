# Final-Hop Bridge Guard 5-Step 训练评测复盘

记录时间：2026-08-05 CST

## 目标

在 `exp/final-hop-bridge-guard` 分支上实现 `final_hop_bridge` turn-credit 策略，并用一次真实 Zhihu 5-step 训练验证它是否能在 `bridge_eval_150.jsonl` 上超过 prompt-only base。

本轮只使用 `my-search-r1/datasets/train.jsonl` 训练；`bridge_eval_150.jsonl` 只作为评测集，不进入训练或调参数据。公开复盘不记录 PyTRIO 远端 sampler weights URI。

## 实现范围

- `my-search-r1/search_r1_minilab/turn_credit.py`：新增 final-hop 属性 cue、`find_final_hop_attribute_turns(...)`、`detect_missing_final_hop_risk(...)`。
- `my-search-r1/search_r1_minilab/training.py`：新增 `final_hop_bridge` 策略、final-hop search bonus 与 missing-final-hop final-answer penalty。
- `my-search-r1/scripts/train_pytrio.py`：新增 `--final-hop-search-turn-bonus` 与 `--missing-final-hop-turn-penalty`。
- `my-search-r1/scripts/analyse_turn_credit.py`：新增 final-hop candidate、training credit 和 missing-final-hop penalty 统计。
- `my-search-r1/tests/test_rollout_training.py`：覆盖 final-hop 奖励、普通 evidence 不误奖、缺少属性查询早答惩罚，以及 correct/invalid/empty/covered 不误触发。

## 参数

训练配置：

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
  --run-name turn-credit-final-hop-bridge-5step-20260805 \
  --trajectory-output-dir my-search-r1/outputs/train_pytrio \
  --turn-credit-policy final_hop_bridge \
  --evidence-search-turn-bonus 0.05 \
  --final-hop-search-turn-bonus 0.10 \
  --early-answer-turn-penalty 0.05 \
  --missing-final-hop-turn-penalty 0.08
```

评测使用训练后的 final sampler weights，命令中的模型路径在公开文档中记为 `<final-sampler-weights-uri>`。

## 训练前 Health Check

命令：

```bash
PYTHONPATH=my-search-r1 uv run python my-search-r1/scripts/eval_pytrio.py \
  --data my-search-r1/datasets/dev.jsonl \
  --backend zhihu_search \
  --limit 5 \
  --batch-size 1 \
  --seed 42 \
  --temperature 0.0 \
  --top-p 1.0 \
  --jsonl-output my-search-r1/eval_results/final_hop_bridge_20260805/health_dev5.jsonl \
  --report-output my-search-r1/eval_results/final_hop_bridge_20260805/health_dev5.md
```

结果：5 条 trajectory，EM 0.4000，format 0.8000，平均搜索 3.0000；Zhihu requests 15，success rate 1.0000，error/timeout/rate-limit 均为 0。因此进入正式训练。

## 训练结果

5/5 step 完成，均进入 optimizer 路径；训练阶段未观察到 Zhihu 429、timeout、credential/http error 或 tool failure。

按 step JSONL 聚合：

| 指标 | 值 |
| --- | ---: |
| trajectories | 40 |
| correct | 5/40 |
| valid format | 35/40 |
| avg search | 2.1750 |
| tool events | 87 |
| tool success | 87 |
| tool failures | 0 |
| `evidence_bridge_search` labels | 18 |
| `final_hop_attribute_search` labels | 10 |
| `early_answer_missing_followup` labels | 4 |
| `missing_final_hop_attribute` labels | 0 |

## Dev70 评测

产物：

- `my-search-r1/eval_results/final_hop_bridge_20260805/final_hop_5step_dev_20260805.jsonl`
- `my-search-r1/eval_results/final_hop_bridge_20260805/final_hop_5step_dev_20260805.md`
- `my-search-r1/eval_results/final_hop_bridge_20260805/final_hop_5step_dev_20260805_offline_diagnostics.md`

| 指标 | 值 |
| --- | ---: |
| EM macro | 0.4286 |
| correct | 30/70 |
| format | 0.9714 |
| avg search | 1.8143 |
| no-search rate | 0.0000 |
| helpful follow-up query rate | 0.4143 |
| bad max-search loop rate | 0.0286 |
| duplicate query rate | 0.0143 |
| Zhihu requests | 127 |
| Zhihu success rate | 1.0000 |
| missing follow-up | 0 |
| possible alias match | 7 |
| answer granularity miss | 0 |

Dev70 通过不明显退化门槛：EM 高于 0.4143，format 高于 0.9000。

## Bridge150 评测

产物：

- `my-search-r1/eval_results/final_hop_bridge_20260805/final_hop_5step_bridge_20260805.jsonl`
- `my-search-r1/eval_results/final_hop_bridge_20260805/final_hop_5step_bridge_20260805.md`
- `my-search-r1/eval_results/final_hop_bridge_20260805/final_hop_5step_bridge_20260805_offline_diagnostics.md`
- `my-search-r1/eval_results/final_hop_bridge_20260805/final_hop_5step_bridge_turn_credit_analysis_20260805.md`

| 指标 | prompt-only base | evidence 20-step | final-hop 5-step |
| --- | ---: | ---: | ---: |
| EM macro | 0.4750 | 0.4583 | 0.4767 |
| correct | 74/150 | 81/150 | 77/150 |
| format | 0.7200 | 0.9400 | 0.7600 |
| avg search | 3.3067 | 3.0933 | 3.2267 |
| Zhihu success rate | 1.0000 | 1.0000 | 1.0000 |
| missing follow-up | 3 | 5 | 3 |
| possible alias match | 3 | 7 | 4 |
| bad max-search loop | 9 | 6 | 9 |

Bridge source-level EM：

| Source | EM |
| --- | ---: |
| 2WikiMultihopQA | 0.5400 |
| Bamboogle | 0.6000 |
| HotpotQA | 0.4667 |
| MuSiQue | 0.3000 |

Turn-credit analysis：

| 指标 | 值 |
| --- | ---: |
| wrong-valid records | 37 |
| evidence candidate records | 127 |
| evidence candidate turns | 220 |
| evidence training credit turns | 43 |
| final-hop candidate records | 122 |
| final-hop candidate turns | 217 |
| final-hop training credit turns | 41 |
| early-answer penalty records | 3 |
| missing final-hop penalty records | 0 |

## Gained / Lost

Final-hop 5-step 相对 prompt-only base：

- gained：7
- lost：4
- net：+3
- gained ids：`dev_10489`、`dev_11183`、`dev_8262`、`dev_11081`、`dev_11099`、`dev_11457`、`dev_7725`
- lost ids：`dev_11279`、`dev_2331`、`dev_11817`、`dev_12504`

Final-hop 5-step 相对 evidence 20-step：

- gained：10
- lost：14
- net：-4
- gained ids：`train_80307`、`dev_10489`、`dev_11183`、`dev_7742`、`test_99`、`dev_11081`、`dev_11099`、`dev_11457`、`dev_11691`、`dev_7725`
- lost ids：`dev_10628`、`dev_11279`、`dev_7767`、`dev_8597`、`dev_10322`、`dev_11245`、`dev_11513`、`dev_11777`、`dev_11817`、`dev_11860`、`dev_12504`、`dev_12533`、`dev_7495`、`dev_7595`

## 结论

本轮是有效 Zhihu run：训练前 health、训练、dev70 eval 和 bridge150 eval 的 Zhihu success rate 均为 1.0000。

Final-hop 5-step 在 bridge 上相对 prompt-only base 有轻微正向信号：EM macro 从 0.4750 到 0.4767，correct 从 74/150 到 77/150。但它没有通过预设总门槛，因为 format 只有 0.7600，低于 0.9000；同时相对 evidence 20-step 少 4 个 correct，format 也明显更差。

因此本轮结论是 partial positive，而不是完整超过 base。它说明 final-hop 属性 search credit 的方向有价值，但当前 penalty 过于保守：训练与 bridge 分析中 `missing_final_hop_attribute` 均为 0，未真正约束“缺少属性查询就回答”的失败模式。下一步不自动扩到 20-step，应先修复 format/max-search/final-answer guard，并让 missing-final-hop detector 在真实 wrong-valid 样本上产生可解释命中。

## 验证

- `PYTHONPATH=my-search-r1 uv run python -m unittest my-search-r1/tests/test_rollout_training.py -v`：31 tests OK。
- `PYTHONPATH=my-search-r1 uv run python -m unittest discover -s my-search-r1/tests -v`：84 tests OK。
- `PYTHONPATH=my-search-r1 uv run python -m compileall -q my-search-r1/search_r1_minilab my-search-r1/scripts`：通过。
- `git diff --check`：通过。
- `analyse_turn_credit.py` 离线 dry run：通过，能输出 final-hop candidate 与 penalty 统计。
- `local_bm25` 1-step smoke：进入 reference logprobs、custom loss 和 optimizer 路径。
- Zhihu dev-5 health、5-step train、dev70 eval、bridge150 eval 均完成且 Zhihu success rate 为 1.0000。
