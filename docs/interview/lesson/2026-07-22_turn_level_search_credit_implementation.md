# Turn-Level Search Credit 实现与阻塞记录

记录时间：2026-07-22 16:07 CST

## 实验目标

本轮目标是在当前 `prompt_search_budget_guard` + KL/std 稳定化 GRPO 训练链路上，探索从纯 final reward 扩展到 turn-level search credit：最终答案仍由 strict EM 决定，但对 wrong-valid 轨迹中保守判定为合理 bridge follow-up 的搜索 turn 给予小的正向训练信号，避免有价值的多跳搜索动作被最终答案错误一起压低。

新分支：

```text
exp/turn-reward-helpful-search
```

## 实现内容

代码变更：

- `my-search-r1/search_r1_minilab/rollout.py`
  - `AssistantTurn` 新增可选 `effective_advantage`、`credit_label`、`credit_bonus`、`credit_query`。
  - `trajectory_to_record` 在 metadata 中记录 `turn_credits`，便于训练 step JSONL 和 Markdown 审计。
- `my-search-r1/search_r1_minilab/training.py`
  - 新增 `TurnCreditConfig(policy, helpful_search_turn_bonus)`，支持 `none` 和 `helpful_bridge`。
  - `build_datum` 使用 turn-level effective advantage；`none` 策略下保持旧 trajectory advantage 广播。
  - `build_training_datums` 支持 `trajectory.advantage == 0` 但存在 positive turn credit 时仍生成 datum。
  - 新增 `turn_credit/*` 训练指标：policy、helpful search turns、credited trajectories、credited tokens。
- `my-search-r1/scripts/train_pytrio.py`
  - 新增 CLI：`--turn-credit-policy {none,helpful_bridge}`、`--helpful-search-turn-bonus`。

首版 `helpful_bridge` 规则：

- 只处理 wrong-valid 轨迹；correct 和 invalid-format 轨迹不加额外 turn bonus。
- 只奖励 assistant tool-call turn。
- 只奖励第 2 次及之后的非重复 query。
- query 需要有新增 terms，并满足现有 follow-up cue 形态。
- 前一条 tool observation 必须 `ok=True` 且 items 非空。
- 被奖励 turn 的 effective advantage 为 `max(trajectory.advantage, 0.0) + helpful_search_turn_bonus`。
- final answer turn 仍使用原 trajectory advantage。

## 本地验证

已运行：

```bash
PYTHONPATH=my-search-r1 uv run python -m unittest my-search-r1/tests/test_rollout_training.py -v
PYTHONPATH=my-search-r1 uv run python -m unittest discover -s my-search-r1/tests -v
PYTHONPATH=my-search-r1 uv run python -m compileall -q my-search-r1/search_r1_minilab my-search-r1/scripts
```

结果：

- `test_rollout_training.py` 23 个测试全部通过。
- 完整 unittest 73 个测试全部通过。
- compileall 通过。

新增测试覆盖：

- `none` 策略保持旧 advantage 序列。
- wrong-valid helpful bridge search turn 获得正向 turn credit。
- repeated query、首个 query、empty previous observation、invalid-format trajectory、correct trajectory 不获得 turn credit。
- `trajectory.advantage == 0` 但有 positive turn credit 时仍生成 training datum。

## Local Smoke 阻塞

尝试运行 local BM25 1-step smoke：

```bash
timeout 180 env PYTHONPATH=my-search-r1 uv run python my-search-r1/scripts/train_pytrio.py \
  --data my-search-r1/tests/fixtures/smoke_eval.jsonl \
  --max-steps 1 \
  --questions-per-batch 1 \
  --group-size 2 \
  --backend local_bm25 \
  --turn-credit-policy helpful_bridge \
  --helpful-search-turn-bonus 0.10 \
  --advantage-normalization standardize \
  --advantage-clip 2.0 \
  --kl-coef 0.01 \
  --policy-ratio-clip 0.2 \
  --learning-rate 1e-5 \
  --swanlab-mode disabled \
  --save-every 0 \
  --run-name turn-credit-local-smoke-20260722
```

结果：

- 进入 `Step 1/1 rollout: 0/2` 后超过 2 分钟没有 trajectory 进展。
- 手动中断。
- 中断栈停在 `sample_requests_async -> sampling_client.sample_async -> await response`。
- 未生成训练 trajectory、optimizer update 或 checkpoint。

判断：

- 该阻塞发生在 PyTRIO sampling 返回前，不经过 local BM25 搜索、turn credit datum 构建、reference logprobs 或 custom loss。
- 这与同日已记录的 PyTRIO sampling 服务/队列/worker 侧阻塞一致。
- 因 local smoke 无法完成，未继续启动 Zhihu 5-step 训练；否则同样会卡在首轮 sampling，无法产生可用实验结果。

## 下一步

PyTRIO sampling 恢复后，按同一分支继续：

1. 重跑 local BM25 1-step smoke，确认 `turn_credit/*` 指标和 `turn_credits` metadata 出现在 step artifact 中。
2. 运行 Zhihu 5-step：

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
  --turn-credit-policy helpful_bridge \
  --helpful-search-turn-bonus 0.10 \
  --save-every 5 \
  --swanlab-mode disabled \
  --run-name turn-credit-helpful-bridge-5step-20260722
```

3. 使用 final sampler weights 跑 dev-5、dev 70 和 offline diagnostics。
4. 与当前最强 `prompt_budget_kl_std_5step` 对比：EM 0.4286、format 0.9429、平均搜索 1.9429、`missing_followup_query=0`、`answer_granularity_miss=0`。

若 5-step EM 低于 0.4286，只记录为负向实验，不扩大训练。
