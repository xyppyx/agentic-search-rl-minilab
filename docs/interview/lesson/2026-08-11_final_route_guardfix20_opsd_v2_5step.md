# Final Route: Guard-Fix 20-Step + OPSD v2 5-Step

日期：2026-08-11

## 决策

固定当前最优路线为：

```text
turn-credit-final-hop-guardfix-20step-20260806
  -> guardfix20-resume-opsd-v2-5step-20260811
```

含义：

- 先用 final-hop guard-fix 20-step 学到更好的 bridge/final-hop 搜索策略。
- 再从该 checkpoint 的 final training state 恢复，做 5-step OPSD v2 保守微调。
- OPSD v2 的 reference/KL teacher 与 OPSD teacher 使用同一个 guard-fix 20-step final sampler weights。

公开复盘不记录远程 training state 或 sampler weights URI。

## Stage 1: Guard-Fix 20-Step

Run name: `turn-credit-final-hop-guardfix-20step-20260806`

公开训练参数：

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

训练结果摘要：

| Metric | Value |
| --- | ---: |
| steps | 20/20 |
| trajectories | 160 |
| correct | 50/160 |
| format | 143/160 |
| avg search | 1.8438 |
| tool failures | 0 |

评测摘要：

| Eval | EM macro | Correct | Format | Avg search | 备注 |
| --- | ---: | ---: | ---: | ---: | --- |
| dev70 retry | 0.4571 | 32/70 | 0.9571 | 1.9000 | clean retry |
| bridge150 patched | 0.5142 | 83/150 | 0.8267 | 3.2000 | patched protocol |

## Stage 2: OPSD v2 5-Step Resume

Run name: `guardfix20-resume-opsd-v2-5step-20260811`

公开训练参数：

```bash
PYTHONPATH=my-search-r1 uv run python my-search-r1/scripts/train_pytrio.py \
  --max-steps 5 \
  --data my-search-r1/datasets/train.jsonl \
  --questions-per-batch 2 \
  --group-size 4 \
  --backend zhihu_search \
  --env-file my-search-r1/.env \
  --seed 42 \
  --temperature 1.0 \
  --top-p 1.0 \
  --swanlab-mode disabled \
  --save-every 5 \
  --run-name guardfix20-resume-opsd-v2-5step-20260811 \
  --trajectory-output-dir my-search-r1/outputs/train_pytrio \
  --resume-state "<guardfix20-final-state-uri>" \
  --reference-model-path "<guardfix20-final-sampler-weights-uri>" \
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

注意：

- 真正让 OPSD v2 从 guard-fix checkpoint 起步的是 `--resume-state`。
- `--reference-model-path` 只用于 KL/reference 与 OPSD teacher logprobs。
- OPSD v2 不做全序列自蒸馏；默认只在 `credited_turns + positive_advantage` 上启用 mask。

训练结果摘要：

| Metric | Value |
| --- | ---: |
| steps | 5/5 |
| trajectories | 40 |
| correct | 7/40 |
| format | 38/40 |
| avg search | 2.1250 |
| Zhihu success rate | 1.0000 |
| tool failures | 0 |
| avg OPSD masked tokens / step | 230.20 |
| avg OPSD mask rate | 0.0158 |
| avg student-teacher logprob gap | 0.0098 |
| credited trajectories | 23 |
| credited tokens | 1349 |

评测摘要：

| Eval | EM macro | Correct | Format | Avg search | 备注 |
| --- | ---: | ---: | ---: | ---: | --- |
| dev70 | 0.4857 | 34/70 | 0.9857 | 1.7286 | 当前最高 clean dev70 |
| bridge150 | 0.5242 | 87/150 | 0.9067 | 3.1400 | 10 个 clean chunks 合并 |

## 20-Step OPSD v2 对照

已额外做 20-step seed43 方差对照：

| Model | dev70 EM | dev70 Correct | dev70 Format | bridge150 EM | bridge150 Correct | bridge150 Format | bridge150 Avg Search |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| OPSD v2 5-step seed42 | 0.4857 | 34/70 | 0.9857 | 0.5242 | 87/150 | 0.9067 | 3.1400 |
| OPSD v2 20-step seed43 | 0.4571 | 32/70 | 1.0000 | 0.5317 | 81/150 | 0.8133 | 3.2533 |

20-step 的 bridge150 EM macro 略高，但 correct 更低、format 明显更差、平均搜索更高。因此最终路线不选择 20-step OPSD v2。

## Final Selection

固定 `guardfix20-resume-opsd-v2-5step-20260811` 为当前最终候选 checkpoint。

选择依据：

- dev70 clean EM/correct 当前最高。
- bridge150 clean correct 当前最高。
- 相比 20-step OPSD v2，5-step 的 format 与搜索成本更稳。
- 5-step 不是“训练不足”，而是在强 guard-fix checkpoint 上做 gated conservative refinement；20-step 方差对照已经表明继续加步数不带来更可靠的综合收益。

后续只补验证，不再改最终主路线：

- 优先补 `alias_granularity_eval_80`，确认 alias/granularity 不被牺牲。
- 如需面试材料稳健性，可补 5-step second seed，但不改变当前路线命名。
