# GRPO KL + Std 5-Step 复盘

复盘时间：2026-07-22 15:05 CST

## 实验目标

在 `prompt_search_budget_guard` 已成为最强 prompt-only base 后，验证训练链路是否可以通过更稳的 GRPO 改造进一步提升，而不是重现此前 no-search、过度搜索、format 和答案粒度退化。

本轮实现和验证两个训练稳定性改动：

1. Group advantage std normalization + clip。
2. Reference logprob drift penalty，也就是 sampled-token KL-style 约束。

目标门槛沿用 best prompt snapshot：

- EM 不低于 0.4143。
- Format 不低于 0.8857。
- `missing_followup_query=0`。
- `answer_granularity_miss=0`。
- no-search 不反弹。
- 平均搜索不明显高于 1.9429。

## 实现变更

代码变更：

- `my-search-r1/search_r1_minilab/rollout.py`
  - `RolloutConfig` 新增 `advantage_normalization`、`advantage_epsilon`、`advantage_clip`。
  - `assign_group_advantages` 支持 `center` 和 `standardize`，可选 advantage clip。
- `my-search-r1/search_r1_minilab/training.py`
  - `TrainingDatum` 支持携带本地 `reference_logprobs`。
  - 新增 `compute_reference_logprobs` / `add_reference_logprobs`，使用 PyTRIO `compute_logprobs` 对完整 trajectory token 序列重新打 reference logprob。
  - 新增 `make_grpo_kl_loss_fn`，通过 PyTRIO `forward_backward_custom` 实现 GRPO objective + sampled-token logprob drift penalty。
  - 新增 custom forward datum 和 loss input helper。
- `my-search-r1/scripts/train_pytrio.py`
  - 新增 CLI：`--advantage-normalization`、`--advantage-epsilon`、`--advantage-clip`、`--kl-coef`、`--policy-ratio-clip`、`--reference-model-path`。
  - `--kl-coef > 0` 时创建 frozen reference sampling client，训练走 custom loss；默认仍走旧 `importance_sampling`。
- `my-search-r1/tests/test_rollout_training.py`
  - 覆盖 std normalization/clip、reference logprob 对齐、custom forward datum 和 reference metadata 保留。

说明：当前 KL 是 sampled-token logprob drift penalty，约束模型在已采样 token 上相对 reference 的 logprob 偏移；它不是完整枚举词表分布的精确 KL。

## 本地验证

命令：

```bash
PYTHONPATH=my-search-r1 uv run python -m unittest discover -s my-search-r1/tests -v
PYTHONPATH=my-search-r1 uv run python -m compileall -q my-search-r1/search_r1_minilab my-search-r1/scripts
PYTHONPATH=my-search-r1 uv run python my-search-r1/scripts/train_pytrio.py \
  --data my-search-r1/tests/fixtures/smoke_eval.jsonl \
  --max-steps 1 \
  --questions-per-batch 1 \
  --group-size 2 \
  --backend local_bm25 \
  --advantage-normalization standardize \
  --advantage-clip 2.0 \
  --kl-coef 0.01 \
  --policy-ratio-clip 0.2 \
  --learning-rate 1e-5 \
  --swanlab-mode disabled \
  --save-every 0 \
  --run-name kl-std-local-smoke-20260722
```

结果：

- 69 个 unittest 全部通过。
- compileall 通过。
- local smoke 实际进入 `reference logprobs -> custom loss -> optimizer` 路径。
- local smoke 1-step：mean reward 0.500、correct 0.500、mean search 1.00、loss_mean 0.0001。

## 5-Step Zhihu 训练

命令：

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
  --advantage-normalization standardize \
  --advantage-clip 2.0 \
  --kl-coef 0.01 \
  --policy-ratio-clip 0.2 \
  --learning-rate 1e-5 \
  --swanlab-mode disabled \
  --save-every 5 \
  --run-name prompt-budget-kl-std-5step-20260722
```

训练结果：

- 5 个 rollout step 完成。
- 3/5 step 执行 optimizer update。
- step 2、step 5 因 group advantage 全 0 跳过。
- 生成 40 条训练 trajectory。
- 保存 step 5 和 final 权重；公开文档不记录远端 URI。

逐步摘要：

| Step | reward | correct | format | avg search | skipped | loss_mean | KL drift mse |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.3500 | 0.3750 | 0.7500 | 2.0000 | 0 | 0.0078 | 0.000326 |
| 2 | 0.0000 | 0.0000 | 1.0000 | 1.6250 | 1 | skipped | - |
| 3 | 0.3625 | 0.3750 | 0.8750 | 2.3750 | 0 | 0.0069 | 0.000320 |
| 4 | -0.0250 | 0.0000 | 0.7500 | 3.3750 | 0 | 0.0024 | 0.000263 |
| 5 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1 | skipped | - |

训练阶段没有观察到 Zhihu error/timeout/rate-limit。

## Eval

Final dev-5：

- EM：0.4000
- format：0.8000
- 平均搜索：3.0000
- no-search：0.0000
- Zhihu requests：15
- Zhihu success rate：1.0000

Final dev 70：

| Run | EM | Format | Avg search | no-search | missing_followup | helpful_followup | granularity_miss | multi_candidate | bad_loop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| prompt search-budget guard | 0.4143 | 0.8857 | 1.9429 | 0.0000 | 0 | 32 | 0 | 0 | 4 |
| prompt budget + KL/std 5-step | 0.4286 | 0.9429 | 1.9429 | 0.0000 | 0 | 30 | 0 | 0 | 4 |

Zhihu backend：

- requests：136
- success rate：1.0000
- empty rate：0.0294
- error rate：0.0000
- timeout rate：0.0000
- rate-limit rate：0.0000
- tool failures：0

Offline diagnostics：

- `missing_followup_query=0`
- `answer_granularity_miss=0`
- `multi_candidate_answer=0`
- `possible_alias_match=7`
- wrong-valid：36

Gained/lost 相对 prompt-only best：

- gained：2
- lost：1
- invalid format：8 降到 4

gained：

- `dev_2179`：Which film has the director who died earlier, Max And Helen or Held Einer Nacht?
- `dev_5358`：What nationality is Isabelle Coutant-Peyre's husband?

lost：

- `dev_6133`：Who is Sophie Of France (1786-1787)'s maternal grandfather?

## 结论

KL/std 方向通过了小预算门控：

- EM 从 0.4143 升到 0.4286。
- format 从 0.8857 升到 0.9429。
- 平均搜索保持 1.9429。
- no-search 没反弹。
- `missing_followup_query`、`answer_granularity_miss`、`multi_candidate_answer` 均保持 0。

这说明此前“训练不如 prompt-only”的结论需要更新：在最强 prompt base 上加入更弱学习率、std normalization、advantage clip 和 KL-style reference 约束后，5-step 训练可以小幅超过 prompt-only，并明显改善 format。

下一步决策：

- 保留 `prompt_budget_kl_std_5step` 作为当前最强 checkpoint 证据，但公开文档不记录远端权重 URI。
- 不直接 50-step；先做 2 个 ablation：
  - std+clip only，确认收益是否主要来自 advantage 标准化。
  - KL only，确认收益是否主要来自 reference 约束。
- 若要扩到 20-step，必须 `--save-every 5` 或更频繁，并用 EM 0.4286、format 0.9429、`missing_followup_query=0` 作为新门槛。
