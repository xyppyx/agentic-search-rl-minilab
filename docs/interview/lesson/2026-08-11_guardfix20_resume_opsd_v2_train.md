# Guardfix20 Resume OPSD v2 Training

日期：2026-08-11

## 目标

用户要求找到表现最好的 guardfix-20step checkpoint，并在该 checkpoint 上继续做 OPSD v2 5-step 和 20-step 训练与评测。

本次纠正此前 OPSD v2 从 base Qwen 初始化的问题：训练策略从 `turn-credit-final-hop-guardfix-20step-20260806` 的 final training state 恢复，KL/reference 和 OPSD teacher 使用同一 checkpoint 的 final sampler weights。公开复盘不记录远程 state/weights URI。

## Source Checkpoint

当前表现最好的 guardfix-20step checkpoint 是 `turn-credit-final-hop-guardfix-20step-20260806`：

- dev70 retry: EM 0.4571、correct 32/70、format 0.9571、avg search 1.9000、Zhihu success rate 1.0
- bridge150 patched: EM 0.5142、correct 83/150、format 0.8267、avg search 3.2000

通过 PyTRIO REST checkpoint list 定位到该 run 的 final training state 和 final sampler weights。训练续跑使用：

- `--resume-state <guardfix20-final-state-uri>`
- `--reference-model-path <guardfix20-final-sampler-weights-uri>`

注意：`--reference-model-path` 只影响 KL/reference 和 OPSD teacher；真正让训练从 guardfix checkpoint 起步的是 `--resume-state`。

## Health Check

先用 guardfix final sampler weights 跑 dev5 health：

| Metric | Value |
| --- | ---: |
| EM | 0.4000 |
| correct | 2/5 |
| format | 0.8000 |
| avg search | 3.0000 |
| Zhihu requests | 15 |
| Zhihu success rate | 1.0000 |
| tool failures | 0 |

输出：

- `my-search-r1/eval_results/gated_opsd_v2_guardfix_resume_20260811/guardfix20_final_health_dev5_20260811.jsonl`
- `my-search-r1/eval_results/gated_opsd_v2_guardfix_resume_20260811/guardfix20_final_health_dev5_20260811.md`

## Common Training Config

两个续训 run 都从同一个 guardfix final state 启动，互不串联：

```bash
PYTHONPATH=my-search-r1 uv run python my-search-r1/scripts/train_pytrio.py \
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
  --trajectory-output-dir my-search-r1/outputs/train_pytrio \
  --resume-state <guardfix20-final-state-uri> \
  --reference-model-path <guardfix20-final-sampler-weights-uri> \
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

## 5-Step Result

Run name: `guardfix20-resume-opsd-v2-5step-20260811`

训练阶段：

| Metric | Value |
| --- | ---: |
| steps | 5/5 |
| trajectories | 40 |
| correct | 7/40 |
| valid format | 38/40 |
| avg search | 2.1250 |
| mean reward / step | 0.1700 |
| mean correct rate / step | 0.1750 |
| mean format rate / step | 0.9500 |
| Zhihu requests | 85 |
| Zhihu success rate | 1.0000 |
| tool failures | 0 |
| avg OPSD masked tokens / step | 230.20 |
| avg OPSD mask rate | 0.0158 |
| avg student-teacher logprob gap | 0.0098 |
| credited trajectories | 23 |
| credited tokens | 1349 |

Final dev5:

| Metric | Value |
| --- | ---: |
| EM | 0.4000 |
| correct | 2/5 |
| format | 1.0000 |
| avg search | 2.4000 |
| Zhihu requests | 12 |
| Zhihu success rate | 1.0000 |

Final dev70:

| Metric | Value |
| --- | ---: |
| EM macro | 0.4857 |
| correct | 34/70 |
| format | 0.9857 |
| avg search | 1.7286 |
| too many search no gain | 9/70 |
| max-search no-answer | 1/70 |
| duplicate query trajectories | 0 |
| Zhihu requests | 121 |
| Zhihu success rate | 1.0000 |
| tool failures | 0 |

诊断：

| Metric | Value |
| --- | ---: |
| wrong valid | 35 |
| possible alias match | 6 |
| missing follow-up query | 1 |
| helpful follow-up query | 28 |
| bad max-search loop | 1 |
| multi-candidate answer | 1 |
| evidence candidate records | 32 |
| evidence training credit turns | 20 |
| final-hop candidate records | 20 |
| final-hop training credit turns | 11 |
| early answer penalty records | 2 |
| missing final-hop penalty records | 1 |
| final answer guard penalty records | 1 |

输出：

- `my-search-r1/outputs/train_pytrio/guardfix20-resume-opsd-v2-5step-20260811/`
- `my-search-r1/eval_results/gated_opsd_v2_guardfix_resume_20260811/guardfix20_resume_opsd_v2_5step_dev5_20260811.*`
- `my-search-r1/eval_results/gated_opsd_v2_guardfix_resume_20260811/guardfix20_resume_opsd_v2_5step_dev70_20260811.*`
- `my-search-r1/eval_results/gated_opsd_v2_guardfix_resume_20260811/guardfix20_resume_opsd_v2_5step_dev70_offline_diagnostics_20260811.*`
- `my-search-r1/eval_results/gated_opsd_v2_guardfix_resume_20260811/guardfix20_resume_opsd_v2_5step_dev70_turn_credit_analysis_20260811.*`

## 20-Step Result

Run name: `guardfix20-resume-opsd-v2-20step-20260811`

训练阶段有 1 次 Zhihu 工具错误，success rate 为 0.9968；因此 20-step 结果只能作为 reference，不作为正式 clean training/eval 结论。

训练阶段：

| Metric | Value |
| --- | ---: |
| steps | 20/20 |
| trajectories | 160 |
| correct | 61/160 |
| valid format | 151/160 |
| avg search | 1.9563 |
| mean reward / step | 0.3756 |
| mean correct rate / step | 0.3813 |
| mean format rate / step | 0.9438 |
| Zhihu requests | 313 |
| Zhihu success rate | 0.9968 |
| tool failures | 1 |
| avg OPSD masked tokens / step | 126.00 |
| avg OPSD mask rate | 0.0098 |
| avg student-teacher logprob gap | 0.0137 |
| credited trajectories | 55 |
| credited tokens | 6420 |

Final dev5:

| Metric | Value |
| --- | ---: |
| EM | 0.4000 |
| correct | 2/5 |
| format | 1.0000 |
| avg search | 2.4000 |
| Zhihu requests | 12 |
| Zhihu success rate | 1.0000 |

Final dev70 reference:

| Metric | Value |
| --- | ---: |
| EM macro | 0.4571 |
| correct | 32/70 |
| format | 0.9714 |
| avg search | 2.0571 |
| too many search no gain | 16/70 |
| max-search no-answer | 2/70 |
| duplicate query trajectories | 0 |
| Zhihu requests | 144 |
| Zhihu success rate | 0.9931 |
| tool failures | 1 |

Dev70 工具失败样本：

- record: `dev_1666`
- question: `What part of the country that includes the birth city of Beatrice Heuser was the film The Beach filmed?`
- failure type: Zhihu parse error during one search call
- model output: format valid but exact match false

诊断：

| Metric | Value |
| --- | ---: |
| wrong valid | 36 |
| possible alias match | 6 |
| missing follow-up query | 1 |
| helpful follow-up query | 38 |
| bad max-search loop | 1 |
| multi-candidate answer | 0 |
| evidence candidate records | 40 |
| evidence training credit turns | 33 |
| final-hop candidate records | 21 |
| final-hop training credit turns | 13 |
| early answer penalty records | 1 |
| missing final-hop penalty records | 1 |
| final answer guard penalty records | 2 |

输出：

- `my-search-r1/outputs/train_pytrio/guardfix20-resume-opsd-v2-20step-20260811/`
- `my-search-r1/eval_results/gated_opsd_v2_guardfix_resume_20260811/guardfix20_resume_opsd_v2_20step_dev5_20260811.*`
- `my-search-r1/eval_results/gated_opsd_v2_guardfix_resume_20260811/guardfix20_resume_opsd_v2_20step_dev70_20260811.*`
- `my-search-r1/eval_results/gated_opsd_v2_guardfix_resume_20260811/guardfix20_resume_opsd_v2_20step_dev70_offline_diagnostics_20260811.*`
- `my-search-r1/eval_results/gated_opsd_v2_guardfix_resume_20260811/guardfix20_resume_opsd_v2_20step_dev70_turn_credit_analysis_20260811.*`

## Conclusion

从 guardfix-20step checkpoint 续训 OPSD v2 与从 base Qwen 训练完全不同。5-step 续训是目前最强 dev70 clean 结果：

- guardfix-20step retry: EM 0.4571、correct 32/70、format 0.9571、avg search 1.9000
- guardfix20 resume OPSD v2 5-step: EM 0.4857、correct 34/70、format 0.9857、avg search 1.7286
- guardfix20 resume OPSD v2 20-step reference: EM 0.4571、correct 32/70、format 0.9714、avg search 2.0571，但训练和 dev70 都有工具失败，不能作为正式结论

当前判断：

1. OPSD v2 对强 checkpoint 有短程增益，5-step 是有效方向，且同时提升 EM、format 和搜索效率。
2. 继续到 20-step 没有看到额外收益，反而搜索成本上升，并且遇到工具失败；不应把 20-step 作为主结果。
3. 后续若扩展，优先围绕 `guardfix20 resume OPSD v2 5-step` 做 clean bridge150/alias80，而不是继续加训练步数。
