# Penalty v3 Follow-Up Aware 5-Step 复盘

复盘时间：2026-07-21 22:44 CST

## 实验目标

基于 `penalty_v2_candidate`、`max_search_no_answer_penalty=0.01/0.005` 和三 checkpoint gained/lost review 的结论，验证 `penalty_v3_followup_aware`：

1. 不继续惩罚所有 `max_search_no_answer`，只惩罚明显空转的 bad max-search loop。
2. 对日期题最终答案粒度不足加 penalty。
3. 对单答案题输出多个候选或过宽答案加 penalty。
4. 保留 duplicate/empty 轻 penalty。

本次开启 SwanLab 在线记录，但不在公开文档中记录 SwanLab 私有 run 链接或 PyTRIO 远端 checkpoint URI。

## 实现变更

新增 reward 参数均默认关闭，旧实验参数语义不变：

- `bad_max_search_penalty`
- `date_granularity_penalty`
- `multi_candidate_answer_penalty`

新增 diagnostics/sensitivity 字段：

- `helpful_followup_query`
- `bad_max_search_loop`
- `multi_candidate_answer`

相关路径：

- `my-search-r1/search_r1_minilab/diagnostics.py`
- `my-search-r1/search_r1_minilab/rewards.py`
- `my-search-r1/search_r1_minilab/offline_diagnostics.py`
- `my-search-r1/search_r1_minilab/reward_sensitivity.py`
- `my-search-r1/scripts/train_pytrio.py`
- `my-search-r1/scripts/eval_pytrio.py`
- `my-search-r1/tests/test_offline_diagnostics.py`
- `my-search-r1/tests/test_reward_sensitivity.py`
- `my-search-r1/tests/test_rollout_training.py`

## 实验配置

- 模型：`Qwen/Qwen3.5-4B`
- 训练数据：`my-search-r1/datasets/train.jsonl`
- 评测数据：`my-search-r1/datasets/dev.jsonl`，70 条
- 搜索 backend：`zhihu_search`
- seed：`42`
- eval decoding：`temperature=0.0`，`top_p=1.0`
- train decoding：`temperature=1.0`，`top_p=1.0`
- 训练预算：`max_steps=5`，`questions_per_batch=2`，`group_size=4`
- SwanLab：`online`
- run name：`reward-penalty-v3-followup-aware-5step-20260721`

Reward 参数：

- `duplicate_query_penalty=0.03`
- `empty_result_penalty=0.01`
- `bad_max_search_penalty=0.01`
- `date_granularity_penalty=0.05`
- `multi_candidate_answer_penalty=0.02`
- `max_search_no_answer_penalty=0.0`
- `verbose_answer_penalty=0.0`
- `verbose_answer_token_threshold=0`

## 运行命令

训练：

```bash
timeout 3600s env PYTHONPATH=my-search-r1 uv run python my-search-r1/scripts/train_pytrio.py \
  --data my-search-r1/datasets/train.jsonl \
  --max-steps 5 \
  --questions-per-batch 2 \
  --group-size 4 \
  --backend zhihu_search \
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
  --run-name reward-penalty-v3-followup-aware-5step-20260721
```

Checkpoint eval 使用训练输出的 final sampler weights，通过 `--model-path` 传入；真实远端路径不写入本文档。

```bash
timeout 3600s env PYTHONPATH=my-search-r1 uv run python my-search-r1/scripts/eval_pytrio.py \
  --data my-search-r1/datasets/dev.jsonl \
  --backend zhihu_search \
  --limit 0 \
  --batch-size 1 \
  --seed 42 \
  --temperature 0.0 \
  --top-p 1.0 \
  --model-path '<final-sampler-weights-uri>' \
  --jsonl-output my-search-r1/eval_results/reward_train_compare_2026-07-21/penalty_v3_followup_aware_5step_dev.jsonl \
  --report-output my-search-r1/eval_results/reward_train_compare_2026-07-21/penalty_v3_followup_aware_5step_dev.md
```

## 离线 Sensitivity 预检查

在训练前对三个关键 checkpoint 重新跑新版 diagnostics 和 reward sensitivity。

| Run | v3 mean delta | penalized | correct penalized | bad max-search penalized | date penalized | multi-candidate penalized |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v2_20 | -0.0039 | 14 | 0 | 9 | 0 | 3 |
| max001 | -0.0023 | 6 | 0 | 0 | 2 | 2 |
| max0005 | -0.0024 | 9 | 0 | 3 | 0 | 3 |

判断：

- v3 离线预检查没有扣到正确样本。
- v2_20 上主要扣 bad max-search loop，符合“只扣明显空转”的目标。
- max001 上主要扣日期粒度和多候选答案，覆盖此前 review 发现的问题。

## 训练过程

5 个 rollout step 全部完成，每步 2 个问题、8 条 trajectory，共 40 条训练 trajectory。5/5 个 step 都产生 optimizer update。

训练总体：

- 平均 reward：0.1565
- correct rate：0.2000
- format rate：0.6000
- 平均搜索次数：2.5000
- duplicate penalty 命中：2 条，合计 0.06
- empty penalty 命中：3 条，合计 0.03
- bad max-search penalty 命中：3 条，合计 0.03
- date granularity penalty 命中：0 条
- multi-candidate penalty 命中：1 条，合计 0.02
- SwanLab 上传完成：531 条 records
- 训练耗时：约 3 分 37 秒

逐步观察：

| Step | reward | correct | format | avg search |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.3250 | 3/8 | 4/8 | 2.50 |
| 2 | 0.2250 | 2/8 | 6/8 | 1.38 |
| 3 | 0.3613 | 3/8 | 7/8 | 2.12 |
| 4 | -0.1125 | 0/8 | 0/8 | 3.50 |
| 5 | -0.0163 | 0/8 | 7/8 | 3.00 |

训练输出：

- `my-search-r1/outputs/train_pytrio/reward-penalty-v3-followup-aware-5step-20260721/`

## Dev Eval 结果

| 指标 | v2_20 | max001 | max0005 | v3 |
| --- | ---: | ---: | ---: | ---: |
| trajectories | 70 | 70 | 70 | 70 |
| correct count | 25 | 21 | 19 | 21 |
| `em/macro` | 0.3571 | 0.3000 | 0.2714 | 0.3000 |
| format count | 51 | 61 | 57 | 60 |
| `format/rate` | 0.7286 | 0.8714 | 0.8143 | 0.8571 |
| `rollout/search_calls` | 2.6143 | 1.4571 | 1.5714 | 1.5143 |
| `behavior/max_search_no_answer_rate` | 0.2714 | 0.0857 | 0.1143 | 0.0857 |
| `behavior/too_many_search_no_gain_rate` | 0.4286 | 0.1000 | 0.1714 | 0.1286 |
| `behavior/bad_max_search_loop_rate` | 0.1286 | 0.0000 | 0.0429 | 0.0286 |
| `helpful_followup_query` | 42 | 16 | 14 | 15 |
| `missing_followup_query` | 1 | 5 | 5 | 6 |
| `answer_granularity_miss` | 0 | 2 | 0 | 0 |
| `multi_candidate_answer` | 3 | 2 | 3 | 1 |
| Zhihu requests | 183 | 102 | 110 | 106 |
| Zhihu success rate | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| tool failures | 0 | 0 | 0 | 0 |

Eval 输出：

- `my-search-r1/eval_results/reward_train_compare_2026-07-21/penalty_v3_followup_aware_5step_dev.jsonl`
- `my-search-r1/eval_results/reward_train_compare_2026-07-21/penalty_v3_followup_aware_5step_dev.md`

## V3 Gained/Lost

相对 max001：

- gained 3：`test_97`、`test_494`、`test_4020`
- lost 3：`dev_3412`、`test_2231`、`test_8542`

相对 max0005：

- gained 3：`dev_2407`、`test_494`、`test_4020`
- lost 1：`test_99`

相对 v2_20：

- gained 3：`test_108`、`dev_2407`、`dev_2223`
- lost 7：`dev_4869`、`test_29`、`test_99`、`dev_174`、`test_2231`、`test_7310`、`test_8542`

关键样本判断：

- `test_97`：v3 修复 max001 的日期粒度问题，输出 `October 2, 1869`。
- `test_4020`：v3 修复 max001/max0005 的多候选答案，输出 `Buddha`。
- `test_494`：v3 修复 max001/max0005 的 query drift，单次搜索答 `David Villa`。
- `dev_3412`：v3 仍过早停在 `American Idol`，没有像 max001 一样追到 `Lari White -> You Can Be a Star`。
- `test_99`：v3 从 max0005 的 `June 5, 2004` 退化为 `February 5, 2004`，不是粒度问题，而是事实读取错误。
- `test_8542`：v3 答 `Damson plums`，语义接近但 strict EM 判错。

## 结论

`penalty_v3_followup_aware` 达到了部分目标：

- 追平 max001 的 EM 0.3000。
- format 0.8571，略低于 max001 的 0.8714，但高于 max0005 的 0.8143。
- 平均搜索 1.5143，仍显著低于 v2_20 的 2.6143。
- bad max-search loop 从 v2_20 的 9 条降到 2 条。
- 日期粒度问题为 0，多候选答案从 max001 的 2 条、max0005 的 3 条降到 1 条。

但 v3 没解决核心多跳 follow-up 风险：

- `missing_followup_query=6`，高于 max001/max0005 的 5，也明显高于 v2_20 的 1。
- `dev_3412` 和 `dev_4869` 仍是典型必要二跳缺失。

当前决策：

- v3 是比 max0005 更好的折中，和 max001 基本持平，但还不能替代 v2_20 的最高 EM checkpoint。
- v3 的 final answer 粒度/唯一性方向有效；继续保留。
- 下一步不应继续调 penalty 权重，而应增加正向 follow-up 信号：对多跳/关系题中引入关键中间实体的 query 给小 bonus，或在 prompt/rollout 层要求先锁定中间实体再回答。
