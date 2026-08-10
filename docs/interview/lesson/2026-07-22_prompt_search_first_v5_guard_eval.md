# Prompt Search-First + V5 No-Search Guard 复盘

复盘时间：2026-07-22 12:15 CST

## 实验目标

在 prompt 约束已显示有效、但 prompt+v3 5-step 训练出现 no-search 退化后，继续自主探索两条方向：

1. 继续优化 prompt，明确要求最终回答前先看到至少一次 search observation。
2. 增加 `no_search_penalty`，只惩罚未搜索且没答对的 trajectory，避免训练学成“格式正确但过早不搜”。

核心目标仍是保住 EM/format，同时降低无收益搜索循环和 no-search 退化。

## 实现变更

代码提交：`9f9f74c 奖励：增加未搜索错误保护`

变更内容：

- `protocol.py`：system prompt 从“需要证据时搜索”加强为“最终回答前先搜索，不要在看到 search result 前凭记忆回答”。
- `rewards.py`：新增默认关闭的 `no_search_penalty`。
- `train_pytrio.py` / `eval_pytrio.py`：接入 `--no-search-penalty`。
- `reward_sensitivity.py`：新增 `reward_v5_no_search_guard` 默认配置和 `no_search` 自定义字段。
- `rollout.py`：将 chat-template prompt reconstruction failure 降级成单条 trajectory 的 `prompt_reconstruction_failed` stop reason，避免长训练整进程崩溃。
- 测试覆盖 prompt、no-search reward、sensitivity config 和 prompt reconstruction failure。

验证：

```bash
PYTHONPATH=my-search-r1 uv run python -m unittest discover -s my-search-r1/tests -v
PYTHONPATH=my-search-r1 uv run python -m compileall -q my-search-r1/search_r1_minilab my-search-r1/scripts
```

结果：65 个 unittest 全部通过；compileall 通过。

## Search-First Prompt Base Eval

命令：

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
  --jsonl-output my-search-r1/eval_results/reward_train_compare_2026-07-21/prompt_search_first_dev_20260722.jsonl \
  --report-output my-search-r1/eval_results/reward_train_compare_2026-07-21/prompt_search_first_dev_20260722.md
```

结果：

- EM：0.3714
- format：0.8000
- 平均搜索：2.0571
- no-search rate：0.0000
- `missing_followup_query=0`
- `helpful_followup_query=33`
- `multi_candidate_answer=0`
- `answer_granularity_miss=0`
- `bad_max_search_loop=3`
- Zhihu requests：144
- success rate：1.0000
- tool failures：0

判断：search-first prompt 本身是当前最高 EM base，比 prompt constraints base 的 EM 0.3429 更高，也超过 v2 20-step 的 EM 0.3571；代价是 format 从 0.8714 降到 0.8000，平均搜索从 1.7143 升到 2.0571。

## V5 Sensitivity

候选配置：

```text
duplicate_query_penalty=0.02
empty_result_penalty=0.00
bad_max_search_penalty=0.005
date_granularity_penalty=0.05
multi_candidate_answer_penalty=0.02
no_search_penalty=0.03
```

离线检查：

- 在失败的 prompt+v3 5-step checkpoint 上，`reward_v5_no_search_guard` 扣到 20 个 no-search wrong-valid 样本，未扣正确样本。
- 在 prompt search-first base 上，`no_search_penalty` 命中 0；因为该 prompt base 没有 no-search 样本。

## 50-Step 尝试

先启动过一次 `prompt-v5-no-search-guard-50step-20260722`，参数为 v5 guard，`save-every=0`。该 run 在 step 41 崩溃，原因不是 Zhihu API，而是 chat template 无法恢复 assistant message boundary：

```text
ValueError: chat template cannot recover assistant message boundary
```

由于没有中间 checkpoint，无法评估 step 40 权重。保留下来的训练 step artifact 显示，到 step 40 前模型没有完全 no-search collapse，但中段已经出现多次平均搜索低于 1 且 correct 为 0 的风险信号。

处理：

- 将 prompt reconstruction failure 降级为单条 trajectory stop reason。
- 后续长训必须开启 `--save-every`。

## V5 20-Step 训练

命令：

```bash
timeout 3600s env PYTHONPATH=my-search-r1 uv run python my-search-r1/scripts/train_pytrio.py \
  --data my-search-r1/datasets/train.jsonl \
  --max-steps 20 \
  --questions-per-batch 2 \
  --group-size 4 \
  --backend zhihu_search \
  --env-file my-search-r1/.env \
  --seed 42 \
  --temperature 1.0 \
  --top-p 1.0 \
  --duplicate-query-penalty 0.02 \
  --empty-result-penalty 0.0 \
  --bad-max-search-penalty 0.005 \
  --date-granularity-penalty 0.05 \
  --multi-candidate-answer-penalty 0.02 \
  --no-search-penalty 0.03 \
  --verbose-answer-penalty 0.0 \
  --verbose-answer-token-threshold 0 \
  --swanlab-mode online \
  --save-every 10 \
  --run-name prompt-v5-no-search-guard-20step-20260722
```

训练结果：

- 20 个 rollout step 完成。
- 14/20 step 执行 optimizer update，6/20 step 因 group advantage 全 0 跳过。
- 生成 160 条训练 trajectory。
- step 10、step 20 和 final sampler weights 均保存；公开文档不记录远端 URI。
- SwanLab 上传完成 1537 条 records；公开文档不记录私有链接。
- 训练阶段未观察到 Zhihu error/timeout/rate-limit。

逐步摘要：

| Step | reward | correct | format | avg search | bad max loop | update skipped |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.3244 | 0.3750 | 0.5000 | 2.5000 | 0.1250 | 0 |
| 6 | 0.4975 | 0.5000 | 1.0000 | 2.0000 | 0.0000 | 0 |
| 10 | 0.4500 | 0.5000 | 0.5000 | 2.5000 | 0.0000 | 1 |
| 13 | 0.1750 | 0.2500 | 0.2500 | 3.5000 | 0.0000 | 0 |
| 15 | -0.0756 | 0.0000 | 0.2500 | 3.8750 | 0.1250 | 0 |
| 19 | 0.7175 | 0.7500 | 0.7500 | 1.6250 | 0.0000 | 0 |
| 20 | 0.0000 | 0.0000 | 1.0000 | 2.0000 | 0.0000 | 1 |

## V5 20-Step Eval

Final dev-5：

- EM：0.4000
- format：0.6000
- 平均搜索：3.2000
- no-search rate：0.0000
- Zhihu success rate：1.0000

Final dev 70：

| Run | EM | Format | Avg search | no-search | missing_followup | helpful_followup | multi_candidate | granularity_miss | bad_loop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| prompt constraints base | 0.3429 | 0.8714 | 1.7143 | 0.0000 | 3 | 21 | 0 | 0 | 5 |
| prompt search-first base | 0.3714 | 0.8000 | 2.0571 | 0.0000 | 0 | 33 | 0 | 0 | 3 |
| prompt+v3 5-step | 0.2429 | 1.0000 | 0.5714 | 0.4286 | 8 | 0 | 0 | 0 | 0 |
| v2 20-step | 0.3571 | 0.7286 | 2.6143 | 0.0000 | 1 | 42 | 3 | 0 | 9 |
| v3 group size 4 | 0.3000 | 0.8571 | 1.5143 | 0.0429 | 6 | 15 | 1 | 0 | 2 |
| prompt+v5 20-step final | 0.3000 | 0.7286 | 2.5857 | 0.0000 | 0 | 51 | 0 | 5 | 4 |

Zhihu backend：

- requests：181
- success rate：1.0000
- empty rate：0.0331
- error rate：0.0000
- timeout rate：0.0000
- rate-limit rate：0.0000
- tool failures：0

Step-10 dev-5 smoke：

- EM：0.2000
- format：0.6000
- 平均搜索：2.4000
- no-search rate：0.2000
- Zhihu success rate：1.0000

由于 step-10 dev-5 已弱于 final 且出现 no-search，不继续跑 step-10 full dev。

## 结论

search-first prompt 值得保留为当前最佳 base：它把 EM 提到 0.3714，并将 `missing_followup_query` 降到 0。

v5 no-search guard 的训练没有通过扩大门控：

- 20-step final EM 只有 0.3000，低于 search-first base 0.3714。
- format 0.7286，低于 search-first base 0.8000 和 prompt constraints base 0.8714。
- 平均搜索 2.5857，几乎回到 v2 20-step 的高搜索区间。
- `answer_granularity_miss=5`，这是明显退化。
- 虽然没有 no-search collapse，且 `missing_followup_query=0`，但训练把模型推向过度 follow-up 与答案粒度不稳。

下一步决策：

- 不继续扩 prompt+v5 到 50-step。
- 保留 search-first prompt，但需要补一个 final answer formatting / date completeness 方向的 prompt 或 reward。
- 长训必须使用 `--save-every 10` 或更频繁。
- 如果继续训练，优先小预算验证“format/date completeness 正向约束”，不要再单独加强 follow-up 或 no-search。
