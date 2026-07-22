# GRPO KL/std Ablation Partial 复盘

复盘时间：2026-07-22 15:25 CST

## 实验目标

在 `prompt_budget_kl_std_5step` 已超过 prompt-only best 后，做两个单因素 ablation：

1. `std+clip only`：只启用 group advantage standardization 和 clip，不启用 KL。
2. `KL only`：只启用 KL-style reference logprob drift penalty，不启用 std normalization。

目标是确认上一轮收益主要来自 std、KL，还是二者组合。

## 已完成部分

### Std+clip only 5-step train

命令配置：

```text
max_steps=5
questions_per_batch=2
group_size=4
backend=zhihu_search
seed=42
temperature=1.0
top_p=1.0
advantage_normalization=standardize
advantage_clip=2.0
kl_coef=0.0
learning_rate=1e-5
swanlab_mode=disabled
save_every=5
run_name=prompt-budget-std-clip-5step-20260722
```

训练结果：

- 5 个 rollout step 完成。
- 3/5 step 执行 optimizer update。
- step 2、step 5 因 group advantage 全 0 跳过。
- 生成 40 条训练 trajectory。
- 保存 step 5 和 final 权重；公开文档不记录远端 URI。

逐步摘要：

| Step | reward | correct | format | avg search | skipped | loss_mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.3625 | 0.3750 | 0.8750 | 1.8750 | 0 | 0.0049 |
| 2 | 0.0000 | 0.0000 | 1.0000 | 1.5000 | 1 | skipped |
| 3 | 0.2375 | 0.2500 | 0.8750 | 2.3750 | 0 | 0.0070 |
| 4 | -0.0500 | 0.0000 | 0.5000 | 3.3750 | 0 | 0.0093 |
| 5 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1 | skipped |

### Std+clip only dev-5

结果：

- EM：0.4000
- format：0.8000
- 平均搜索：3.0000
- no-search：0.0000
- bad max-search loop：0
- Zhihu requests：15
- Zhihu success rate：1.0000
- tool failures：0

这个 dev-5 结果与 `KL/std` 组合版 dev-5 相同，但弱于组合版 full dev 结论所需证据；还不能判断 `std+clip only` 是否足够解释 full dev 收益。

## 阻塞部分

### Std+clip only full dev

第一次 full dev 运行到 23/70 后长时间无进度，手动中断。中断栈显示卡在 PyTRIO sampling await：

```text
sample_requests_async -> sampling_client.sample_async -> await response
```

第二次重跑从 0/70 开始超过 2 分钟无进度，手动中断，栈同样在 PyTRIO sampling await。

### KL only train

`KL only` 5-step 训练在 step 1 rollout 阶段超过 2 分钟没有完成第一条 trajectory，手动中断。中断栈同样在 PyTRIO sampling await。

配置原计划：

```text
advantage_normalization=center
advantage_clip=0.0
kl_coef=0.01
policy_ratio_clip=0.2
learning_rate=1e-5
```

### Base sampling smoke

为了确认不是 ablation checkpoint 特有问题，额外运行 base dev-1 smoke。该 smoke 在 0/1 超过 90 秒无进度，手动中断，栈同样在 PyTRIO sampling await。

## 当前判断

本轮 ablation 尚未完成，不能把收益归因给 std 或 KL 的单项作用。

已知信息：

- `std+clip only` 训练可跑通，dev-5 不差，说明 advantage standardization 没有立刻造成 no-search/format 崩坏。
- `std+clip only` 训练 step 3/4 的 reward/correct 弱于 `KL/std` 组合版同 step，弱信号上看 KL 可能有稳定化作用。
- full dev 和 KL-only 被 PyTRIO sampling 阻塞，无法形成可比较结论。

下一步：

1. 等 PyTRIO sampling 恢复后，优先补 `std+clip only` full dev。
2. 再跑 `KL only` 5-step train、dev-5、full dev。
3. 三组完整后再比较：
   - prompt-only best
   - std+clip only
   - KL only
   - KL/std combo

完成前不要扩大到 20-step。
