# KL/std 20-Step Retry 训练与评测复盘

复盘时间：2026-07-22 18:35 CST

## 实验目标

PyTRIO sampling 恢复后，重跑此前被 sampling 阻塞的 `prompt_budget_kl_std_20step`，验证当前最强 5-step 配置扩大到 20-step 后是否继续提升。

对照门槛：

- 当前最强 checkpoint：`prompt_budget_kl_std_5step`
- EM：0.4286
- format：0.9429
- 平均搜索：1.9429
- `missing_followup_query=0`
- `answer_granularity_miss=0`

本轮在 `exp/turn-reward-helpful-search` 分支上运行，但未传 `--turn-credit-policy`，因此 turn credit 保持默认关闭，训练行为为 KL/std 主线配置。

## 训练配置

关键参数：

```text
max_steps=20
questions_per_batch=2
group_size=4
backend=zhihu_search
seed=42
temperature=1.0
top_p=1.0
advantage_normalization=standardize
advantage_clip=2.0
kl_coef=0.01
policy_ratio_clip=0.2
learning_rate=1e-5
save_every=5
swanlab_mode=disabled
```

run name：

```text
prompt-budget-kl-std-20step-20260722-retry
```

公开文档不记录远端 state 或 sampler weights URI。

## 训练结果

- 20/20 step 完成。
- 保存 step 5、10、15、20 和 final。
- 160 条训练 trajectory 生成完成。
- 10/20 step 执行 optimizer update。
- skipped step：2、5、7、8、14、15、17、20，以及其余 group advantage 全 0 的 step。
- 训练过程未观察到 Zhihu timeout、429、credential/http error 或 PyTRIO sampling 阻塞。

逐步摘要：

| Step | reward | correct | avg search | update | loss_mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.362 | 0.375 | 2.12 | yes | 0.0027 |
| 2 | 0.000 | 0.000 | 1.62 | no | skipped |
| 3 | 0.237 | 0.250 | 2.38 | yes | 0.0074 |
| 4 | -0.038 | 0.000 | 3.38 | yes | 0.0097 |
| 5 | 0.000 | 0.000 | 1.25 | no | skipped |
| 6 | 0.625 | 0.625 | 1.50 | yes | 0.0008 |
| 7 | 0.500 | 0.500 | 1.25 | no | skipped |
| 8 | 0.000 | 0.000 | 1.00 | no | skipped |
| 9 | 0.250 | 0.250 | 1.25 | yes | 0.0007 |
| 10 | 0.475 | 0.500 | 2.25 | yes | 0.0052 |
| 11 | 0.125 | 0.125 | 1.38 | yes | -0.0001 |
| 12 | 0.375 | 0.375 | 1.88 | yes | 0.0012 |
| 13 | 0.487 | 0.500 | 2.75 | yes | 0.0389 |
| 14 | 0.500 | 0.500 | 1.62 | no | skipped |
| 15 | 0.500 | 0.500 | 1.38 | no | skipped |
| 16 | 0.113 | 0.125 | 2.00 | yes | 0.0196 |
| 17 | 0.500 | 0.500 | 1.25 | no | skipped |
| 18 | 0.750 | 0.750 | 1.75 | yes | -0.0038 |
| 19 | 0.863 | 0.875 | 1.62 | yes | 0.0005 |
| 20 | 0.000 | 0.000 | 1.12 | no | skipped |

## Eval

Dev-5：

- EM：0.4000
- format：1.0000
- 平均搜索：2.2000
- Zhihu requests：11
- Zhihu success rate：1.0000

完整 dev 70：

| Metric | 5-step KL/std | 20-step retry |
| --- | ---: | ---: |
| EM macro | 0.4286 | 0.3714 |
| Correct count | 30 | 26 |
| Format rate | 0.9429 | 0.9857 |
| Avg search | 1.9429 | 1.4571 |
| no-search rate | 0.0000 | 0.0000 |
| helpful follow-up rate | 0.4286 | 0.2714 |
| bad max-search loop rate | 0.0571 | 0.0143 |
| too many search no gain rate | 0.2000 | 0.0714 |
| missing follow-up query | 0 | 3 |
| answer granularity miss | 0 | 0 |
| multi-candidate answer | 0 | 0 |
| Zhihu requests | 136 | 102 |
| Zhihu success rate | 1.0000 | 1.0000 |

分来源 EM：

| Source | 20-step retry EM |
| --- | ---: |
| 2wikimultihopqa | 0.5000 |
| bamboogle | 0.6000 |
| hotpotqa | 0.3000 |
| musique | 0.2000 |
| nq | 0.2000 |
| popqa | 0.2000 |
| triviaqa | 0.6000 |

Offline diagnostics：

- wrong-valid：43
- possible alias match：7
- missing follow-up query：3
- answer granularity miss：0
- bad max-search loop：1
- multi-candidate answer：0

## Gained/Lost

相对 `prompt_budget_kl_std_5step`：

- gained：0
- lost：4

lost 样本：

| ID | Source | Gold | 20-step answer | Search | Diagnostic |
| --- | --- | --- | --- | ---: | --- |
| `dev_4869` | 2wikimultihopqa | `Sextus Aelius Catus` | `Tiberius` | 1 | `missing_followup_query` |
| `dev_3741` | hotpotqa | `Dziga Vertov` | `Ken Jacobs` | 1 | evidence/entity binding error |
| `dev_2429` | hotpotqa | `42.5` | `10.5` | 1 | numeric evidence reading error |
| `dev_174` | musique | `Clifton College` | `Juilliard School` | 1 | role binding / premature answer |

## 结论

20-step retry 未通过扩大训练门槛：

- EM 从 0.4286 降到 0.3714。
- 没有新增答对样本，丢失 4 条。
- format 从 0.9429 提升到 0.9857，平均搜索从 1.9429 降到 1.4571，但这是以必要 follow-up 减少和 EM 退化为代价。
- `missing_followup_query` 从 0 增到 3，说明 20-step 继续强化了“更少搜索、更快收束”的行为，但压掉了部分多跳/role-binding 所需搜索。

决策：

- 不继续跑 KL/std 50-step。
- 保留 `prompt_budget_kl_std_5step` 作为当前最强 checkpoint 证据。
- 后续若继续训练，应优先补 `std-only` full dev 和 `KL-only` 5-step/dev，用于归因；或转向 turn-level search credit / query disambiguation，避免继续把模型推向过早单跳回答。
