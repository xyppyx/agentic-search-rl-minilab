# KL/std 20-Step Sampling 阻塞记录

记录时间：2026-07-22 15:29 CST

## 实验目标

用户决策：由于 `std+clip only` 与 `KL only` 消融实验被 PyTRIO sampling 阻塞，当前先将 KL/std 组合视为项目合理必备技术手段，并直接尝试在 `prompt_search_budget_guard` 基础上扩大到 20-step。

本次目标是运行 `prompt_budget_kl_std_20step`，延续 5-step 最强 checkpoint 的训练配置：

- `advantage_normalization=standardize`
- `advantage_clip=2.0`
- `kl_coef=0.01`
- `policy_ratio_clip=0.2`
- `learning_rate=1e-5`
- `group_size=4`
- `questions_per_batch=2`
- `save_every=5`

## 命令

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
  --advantage-normalization standardize \
  --advantage-clip 2.0 \
  --kl-coef 0.01 \
  --policy-ratio-clip 0.2 \
  --learning-rate 1e-5 \
  --swanlab-mode disabled \
  --save-every 5 \
  --run-name prompt-budget-kl-std-20step-20260722
```

## 结果

训练未完成，也未产生可用 checkpoint 或 trajectory。

实际进度：

- 进入 `Training: 0/20` 的 step 1 rollout 阶段。
- `Step 1/20 rollout: 0/8` 后超过 2 分钟没有任何 trajectory 进展。
- 按项目 TODO 中的外部 sampling 阻塞规则手动中断。
- 检查 `my-search-r1/outputs/train_pytrio/prompt-budget-kl-std-20step-20260722/`，未发现输出目录或产物。

中断栈关键信息：

```text
sample_requests_async -> sampling_client.sample_async -> await response
asyncio.exceptions.CancelledError
KeyboardInterrupt
```

## 判断

这次失败属于 PyTRIO sampling 外部阻塞，不是知乎搜索 API、reward、KL/std loss 或本地训练逻辑的验证失败。

本次没有观察到：

- Zhihu API 429、timeout、credential/http error。
- tool failure。
- optimizer update。
- checkpoint 保存。
- dev eval 结果。

因此不能记录为 20-step 训练结果，也不能更新当前最强 checkpoint 结论。

## 本地/服务侧诊断

补充诊断时间：2026-07-22 15:34 CST

为了区分本地问题和 PyTRIO 服务问题，做了最小化检查：

- `.env` 中 `PYTRIO_API_KEY` 存在。
- `ZHIHU_SEARCH_KEY`/`ZHIHU_API_KEY` 存在。
- `pytrio` 包可导入，`ServiceClient` 可创建。
- 普通外网 TCP 检查正常。
- PyTRIO 控制面正常：`ServiceClient.get_supported_models()` 在 60 秒内返回 list。
- PyTRIO sampler 创建正常：`create_sampling_client("Qwen/Qwen3.5-4B")` 与 `get_tokenizer()` 成功。
- PyTRIO base URL 为 `https://pytrio.cn`，TCP 443 连通。
- 使用 numpy token ids 发起最小 1-token direct sample，60 秒内无返回，被 `timeout` 终止。

direct sample 只依赖 PyTRIO sampling，不经过 Search-R1 rollout、知乎搜索、reward、KL/std loss 或训练 optimizer。因此当前更可能是 PyTRIO sampling 服务/队列/worker 侧不可用或长时间排队，而不是本地代码问题。

## 下一步

PyTRIO sampling 恢复后，优先重跑同一 20-step 命令。重跑时继续保留：

- `--save-every 5`
- `--swanlab-mode disabled`，除非明确需要在线记录。
- 原 run name 可改为带 retry 后缀，避免和阻塞尝试混淆。

完成训练后再运行：

- final dev-5 smoke。
- final dev 70 eval。
- offline diagnostics。
- 与 `prompt_search_budget_guard` 和 `prompt_budget_kl_std_5step` 做 gained/lost 对比。
