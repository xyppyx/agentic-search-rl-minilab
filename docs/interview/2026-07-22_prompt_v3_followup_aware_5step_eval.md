# Prompt + V3 Follow-Up Aware 5-Step 复盘

复盘时间：2026-07-22 11:35 CST

## 实验目标

在当前 prompt/rollout 多跳 follow-up 与短答案格式约束基础上，做 5-step 小预算 GRPO 训练，组合 v3 follow-up-aware/bad-loop reward 约束。

目标是保住或提升 prompt base 的 EM/format，同时把平均搜索次数和 `bad_max_search_loop` 拉回 v3 group size 4 附近。

## 实验配置

- 模型：`Qwen/Qwen3.5-4B`
- 训练数据：`my-search-r1/datasets/train.jsonl`
- 评测数据：`my-search-r1/datasets/dev.jsonl`
- 搜索 backend：`zhihu_search`
- seed：`42`
- 训练预算：`max_steps=5`、`questions_per_batch=2`、`group_size=4`
- train decoding：`temperature=1.0`、`top_p=1.0`
- eval decoding：`temperature=0.0`、`top_p=1.0`
- SwanLab：`online`，公开文档不记录私有 run 链接
- checkpoint：使用 final sampler weights 跑评测，公开文档不记录远端 URI
- run name：`prompt-v3-followup-aware-5step-20260722`

Reward 参数：

- `duplicate_query_penalty=0.03`
- `empty_result_penalty=0.01`
- `bad_max_search_penalty=0.01`
- `date_granularity_penalty=0.05`
- `multi_candidate_answer_penalty=0.02`
- `max_search_no_answer_penalty=0.0`
- `verbose_answer_penalty=0.0`
- `verbose_answer_token_threshold=0`

## 命令

训练：

```bash
timeout 3600s env PYTHONPATH=my-search-r1 uv run python my-search-r1/scripts/train_pytrio.py \
  --data my-search-r1/datasets/train.jsonl \
  --max-steps 5 \
  --questions-per-batch 2 \
  --group-size 4 \
  --backend zhihu_search \
  --env-file my-search-r1/.env \
  --seed 42 \
  --temperature 1.0 \
  --top-p 1.0 \
  --duplicate-query-penalty 0.03 \
  --empty-result-penalty 0.01 \
  --bad-max-search-penalty 0.01 \
  --date-granularity-penalty 0.05 \
  --multi-candidate-answer-penalty 0.02 \
  --verbose-answer-penalty 0.0 \
  --verbose-answer-token-threshold 0 \
  --swanlab-mode online \
  --save-every 0 \
  --run-name prompt-v3-followup-aware-5step-20260722
```

dev-5 checkpoint smoke：

```bash
PYTHONPATH=my-search-r1 uv run python my-search-r1/scripts/eval_pytrio.py \
  --data my-search-r1/datasets/dev.jsonl \
  --backend zhihu_search \
  --env-file my-search-r1/.env \
  --limit 5 \
  --batch-size 1 \
  --seed 42 \
  --temperature 0.0 \
  --top-p 1.0 \
  --model-path '<final-sampler-weights-uri>' \
  --jsonl-output my-search-r1/eval_results/reward_train_compare_2026-07-21/prompt_v3_followup_aware_5step_dev5_20260722.jsonl \
  --report-output my-search-r1/eval_results/reward_train_compare_2026-07-21/prompt_v3_followup_aware_5step_dev5_20260722.md
```

dev 70：

```bash
PYTHONPATH=my-search-r1 uv run python my-search-r1/scripts/eval_pytrio.py \
  --data my-search-r1/datasets/dev.jsonl \
  --backend zhihu_search \
  --env-file my-search-r1/.env \
  --limit 0 \
  --batch-size 1 \
  --seed 42 \
  --temperature 0.0 \
  --top-p 1.0 \
  --model-path '<final-sampler-weights-uri>' \
  --jsonl-output my-search-r1/eval_results/reward_train_compare_2026-07-21/prompt_v3_followup_aware_5step_dev_20260722.jsonl \
  --report-output my-search-r1/eval_results/reward_train_compare_2026-07-21/prompt_v3_followup_aware_5step_dev_20260722.md
```

offline diagnostics：

```bash
PYTHONPATH=my-search-r1 uv run python my-search-r1/scripts/analyse_offline_diagnostics.py \
  --input my-search-r1/eval_results/reward_train_compare_2026-07-21/prompt_v3_followup_aware_5step_dev_20260722.jsonl \
  --jsonl-output my-search-r1/eval_results/reward_train_compare_2026-07-21/prompt_v3_followup_aware_5step_dev_20260722_offline_diagnostics.jsonl \
  --report-output my-search-r1/eval_results/reward_train_compare_2026-07-21/prompt_v3_followup_aware_5step_dev_20260722_offline_diagnostics.md \
  --title 'Prompt V3 Followup Aware 5-Step Dev Offline Diagnostics'
```

## 输出

- `my-search-r1/outputs/train_pytrio/prompt-v3-followup-aware-5step-20260722/`
- `my-search-r1/eval_results/reward_train_compare_2026-07-21/prompt_v3_followup_aware_5step_dev5_20260722.jsonl`
- `my-search-r1/eval_results/reward_train_compare_2026-07-21/prompt_v3_followup_aware_5step_dev5_20260722.md`
- `my-search-r1/eval_results/reward_train_compare_2026-07-21/prompt_v3_followup_aware_5step_dev5_20260722_offline_diagnostics.jsonl`
- `my-search-r1/eval_results/reward_train_compare_2026-07-21/prompt_v3_followup_aware_5step_dev5_20260722_offline_diagnostics.md`
- `my-search-r1/eval_results/reward_train_compare_2026-07-21/prompt_v3_followup_aware_5step_dev_20260722.jsonl`
- `my-search-r1/eval_results/reward_train_compare_2026-07-21/prompt_v3_followup_aware_5step_dev_20260722.md`
- `my-search-r1/eval_results/reward_train_compare_2026-07-21/prompt_v3_followup_aware_5step_dev_20260722_offline_diagnostics.jsonl`
- `my-search-r1/eval_results/reward_train_compare_2026-07-21/prompt_v3_followup_aware_5step_dev_20260722_offline_diagnostics.md`

## 训练过程

5 个 rollout step 全部完成，每步 2 个问题、8 条 trajectory，共 40 条训练 trajectory。3/5 个 step 执行 optimizer update，step 2 和 step 5 因 group advantage 全 0 跳过。

训练总体：

- 平均 reward：0.1435
- correct rate：0.1500
- format rate：0.9500
- 平均搜索次数：1.4000
- no-search rate：0.1000
- tool failures：0
- Zhihu requests：56
- Zhihu success rate：1.0000
- SwanLab 上传完成：369 条 records
- 训练耗时：约 1 分 36 秒

逐步观察：

| Step | reward | correct | format | avg search | bad max loop | update skipped | loss mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.3488 | 0.3750 | 0.7500 | 1.7500 | 0.1250 | 0 | 0.000675 |
| 2 | 0.0000 | 0.0000 | 1.0000 | 1.2500 | 0.0000 | 1 | skipped |
| 3 | -0.0013 | 0.0000 | 1.0000 | 2.0000 | 0.0000 | 0 | 0.000018 |
| 4 | 0.3700 | 0.3750 | 1.0000 | 1.5000 | 0.1250 | 0 | 0.003033 |
| 5 | 0.0000 | 0.0000 | 1.0000 | 0.5000 | 0.0000 | 1 | skipped |

## Dev Eval 结果

dev-5 smoke：

- EM：0.4000
- format：1.0000
- 平均搜索：1.0000
- `missing_followup_query=3`
- `multi_candidate_answer=0`
- `answer_granularity_miss=0`
- Zhihu requests：5
- Zhihu success rate：1.0000
- tool failures：0

完整 dev 70：

| Run | EM | Format | Avg search | no search | missing_followup_query | helpful_followup_query | multi_candidate_answer | answer_granularity_miss | bad_max_search_loop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| prompt base | 0.3429 | 0.8714 | 1.7143 | 0.0000 | 3 | 21 | 0 | 0 | 5 |
| v3 group size 4 | 0.3000 | 0.8571 | 1.5143 | 0.0429 | 6 | 15 | 1 | 0 | 2 |
| v2 20-step | 0.3571 | 0.7286 | 2.6143 | 0.0000 | 1 | 42 | 3 | 0 | 9 |
| prompt + v3 5-step | 0.2429 | 1.0000 | 0.5714 | 0.4286 | 8 | 0 | 0 | 0 | 0 |

Zhihu backend：

- dev 70 requests：40
- success rate：1.0000
- empty rate：0.0250
- error rate：0.0000
- timeout rate：0.0000
- rate-limit rate：0.0000
- latency：1.0880
- tool failures：0

分类 EM：

| Source | EM |
| --- | ---: |
| 2wikimultihopqa | 0.2000 |
| bamboogle | 0.4000 |
| hotpotqa | 0.2000 |
| musique | 0.0000 |
| nq | 0.1000 |
| popqa | 0.3000 |
| triviaqa | 0.5000 |

## Gained/Lost

相对 prompt base：

- gained 2：`test_22`、`test_8542`
- lost 9：`dev_7773`、`test_99`、`test_97`、`dev_2407`、`dev_2429`、`dev_654`、`dev_3`、`test_494`、`test_6087`

典型 lost：

- `dev_2407`：prompt base 搜索 2 次答 `Dexter`，训练后不搜索直接答 `Martin Luther King III`。
- `test_494`：prompt base 搜索 1 次答 `David Villa`，训练后不搜索直接答 `Lionel Messi`。
- `test_6087`：prompt base 搜索 1 次答 `John Masefield`，训练后不搜索直接答 `W.H. Auden`。
- `test_99`：prompt base 搜索 4 次答 `June 5, 2004`，训练后搜索 1 次答 `1981`。
- `test_97`：prompt base 搜索 2 次答 `October 2, 1869`，训练后搜索 1 次答 `October 2`，出现日期信息不完整。

相对 v3 group size 4：

- gained 4：`dev_7774`、`test_7310`、`test_22`、`test_8542`
- lost 8：`test_97`、`test_108`、`dev_2407`、`dev_3741`、`dev_1101`、`dev_2223`、`test_494`、`test_6087`

## 结论

本轮训练没有通过扩大训练门控。

达成的部分：

- format 从 prompt base 的 0.8714 提升到 1.0000。
- 平均搜索从 1.7143 降到 0.5714。
- `bad_max_search_loop` 从 5 降到 0。
- `multi_candidate_answer` 和 `answer_granularity_miss` 保持 0。
- Zhihu backend 稳定，success rate 1.0，无 error/timeout/rate-limit。

失败的部分：

- EM 从 prompt base 的 0.3429 降到 0.2429，净丢 7 条正确。
- no-search rate 从 0 升到 0.4286，模型明显过早直接答。
- `helpful_followup_query` 从 21 降到 0。
- `missing_followup_query` 从 3 升到 8。
- 虽然搜索循环被压住，但必要搜索和 follow-up query 也被压掉了。

判断：

prompt 约束本身有效；在其基础上直接套 v3 penalty 小训，会把模型推向“格式很好、搜索很少、但证据不足”的策略。下一步不应扩到 20/50 step。更合理的方向是先做 no-empty 或 lower-penalty ablation，或加入显式必要搜索/follow-up 正向门控，再跑 5-step。
