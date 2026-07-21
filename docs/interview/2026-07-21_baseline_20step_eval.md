# 原始 Reward 20-Step Checkpoint 复盘

复盘时间：2026-07-21 20:15 CST

## 实验目标

在同一套 Zhihu dev 70 题 benchmark 上，固化一个原始 reward 的 20-step GRPO checkpoint 作为后续对照锚点，并比较它相对 base 与 5-step checkpoint 的收益。

本次不公开记录 PyTRIO 远端 checkpoint URI；复跑 checkpoint eval 时应从本地终端日志或 PyTRIO 控制台查找对应 final sampler weights。

## 实验配置

- 模型：`Qwen/Qwen3.5-4B`
- 训练数据：`my-search-r1/datasets/train.jsonl`
- 评测数据：`my-search-r1/datasets/dev.jsonl`，70 条
- 搜索 backend：`zhihu_search`
- seed：`42`
- eval decoding：`temperature=0.0`，`top_p=1.0`
- train decoding：`temperature=1.0`，`top_p=1.0`
- 训练预算：`max_steps=20`，`questions_per_batch=2`，`group_size=4`
- reward：原始 reward，无 penalty
- SwanLab：`disabled`
- run name：`reward-baseline-20step-20260721`

## 运行命令

训练：

```bash
timeout 10800s env PYTHONPATH=my-search-r1 uv run python my-search-r1/scripts/train_pytrio.py --data my-search-r1/datasets/train.jsonl --max-steps 20 --questions-per-batch 2 --group-size 4 --backend zhihu_search --seed 42 --temperature 1.0 --top-p 1.0 --swanlab-mode disabled --save-every 0 --run-name reward-baseline-20step-20260721
```

Checkpoint eval 使用训练输出的 final sampler weights，通过 `--model-path` 传入；真实远端路径不写入本文档。

```bash
timeout 3600s env PYTHONPATH=my-search-r1 uv run python my-search-r1/scripts/eval_pytrio.py --data my-search-r1/datasets/dev.jsonl --backend zhihu_search --limit 0 --batch-size 1 --seed 42 --temperature 0.0 --top-p 1.0 --model-path '<final-sampler-weights-uri>' --jsonl-output my-search-r1/eval_results/reward_train_compare_2026-07-21/base_20step_dev.jsonl --report-output my-search-r1/eval_results/reward_train_compare_2026-07-21/base_20step_dev.md
```

## 训练过程

- 20 个 rollout step 全部完成，每步 2 个问题、8 条 trajectory，共 160 条训练 trajectory。
- 有效 optimizer update：17/20。
- skipped step：3 个，分别为 step 4、10、19；原因是对应 batch 内 group-relative advantage 全为 0。
- 平均 step 耗时：36.26 秒。
- 训练 rollout 平均 reward：0.2888。
- 训练 rollout 平均 correct rate：0.3125。
- 训练 rollout 平均 format rate：0.7625。
- 训练 rollout 平均搜索次数：1.825。
- final step：`reward/mean=-0.0125`、`reward/correct=0.0`、`reward/format=0.8750`、`rollout/search_calls=1.5`、`trainer/loss_mean=0.001943`。
- final 累计 Zhihu backend 指标：请求 292 次，`success_rate=0.9966`，`error_rate=0.0034`，`timeout_rate=0.0`，`rate_limit_rate=0.0`。

训练输出：

- `my-search-r1/outputs/train_pytrio/reward-baseline-20step-20260721/`

## Dev Eval 结果

| 指标 | base | 5-step 原始 reward | 20-step 原始 reward |
| --- | ---: | ---: | ---: |
| trajectories | 70 | 70 | 70 |
| correct count | 17 | 21 | 19 |
| `em/macro` | 0.2429 | 0.3000 | 0.2714 |
| format count | 43 | 58 | 60 |
| `format/rate` | 0.6143 | 0.8286 | 0.8571 |
| `rollout/search_calls` | 1.9571 | 1.7714 | 1.7000 |
| `behavior/direct_correct_rate` | 0.0143 | 0.0000 | 0.0000 |
| `behavior/searched_correct_rate` | 0.2286 | 0.3000 | 0.2714 |
| `behavior/searched_wrong_rate` | 0.3714 | 0.5286 | 0.5571 |
| `behavior/duplicate_query_rate` | 0.0143 | 0.0286 | 0.0000 |
| `behavior/empty_observation_rate` | 0.0511 | 0.0484 | 0.0504 |
| `behavior/max_search_no_answer_rate` | 0.2143 | 0.1143 | 0.1286 |
| `behavior/too_many_search_no_gain_rate` | 0.3000 | 0.2143 | 0.2143 |
| tool failures | 0 | 0 | 0 |
| Zhihu requests | 137 | 124 | 119 |
| Zhihu success rate | 1.0000 | 1.0000 | 1.0000 |

20-step 分来源 EM：

| Source | EM |
| --- | ---: |
| `2wikimultihopqa` | 0.2000 |
| `bamboogle` | 0.5000 |
| `hotpotqa` | 0.3000 |
| `musique` | 0.1000 |
| `nq` | 0.2000 |
| `popqa` | 0.2000 |
| `triviaqa` | 0.4000 |

Eval 输出：

- `my-search-r1/eval_results/reward_train_compare_2026-07-21/base_20step_dev.jsonl`
- `my-search-r1/eval_results/reward_train_compare_2026-07-21/base_20step_dev.md`

## 样本级变化

相对 base：

- 20-step checkpoint 新增答对 8 条，丢失原答对 6 条，净增 2 条。
- format 新增合规 19 条，丢失 2 条，净增 17 条。

相对 5-step 原始 reward checkpoint：

- 20-step checkpoint 新增答对 2 条，丢失 5-step 答对的 4 条，净减 2 条。
- format 新增合规 5 条，丢失 3 条，净增 2 条。

## 结论

- 20-step 原始 reward checkpoint 优于 base：EM 从 0.2429 提升到 0.2714，format rate 从 0.6143 提升到 0.8571，平均搜索次数从 1.9571 降到 1.7000。
- 20-step 没有超过本轮 5-step 原始 reward checkpoint：5-step EM 为 0.3000，20-step EM 为 0.2714。
- 20-step 的主要收益是格式合规和搜索行为收敛；答对率在 5-step 之后没有继续提高，符合“step=20 后性价比下降”的经验判断，但本次仍只是单 seed 单 run 结果。
- 后续 reward shaping 不能只看搜索次数下降；需要把 EM、format、证据阅读错误和样本级 gained/lost 一起作为停止条件。

## 风险与注意

- 这是单 seed、单 checkpoint 的 20-step 对照，不代表稳定均值。
- 训练阶段累计 Zhihu backend 出现少量 error rate，但 eval 阶段 70 条无工具失败，Zhihu success rate 为 1.0。
- 严格 EM 仍会误伤部分长答案、别名、日期格式和解释性回答；后续需要补充错误案例审阅，而不是只依赖单一 EM 数字。
