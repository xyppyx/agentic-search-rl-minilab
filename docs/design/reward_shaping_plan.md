# Reward Shaping Plan

本文记录 Search-R1 MiniLab 的 reward/penalty 设计版本、动机、实验结果和后续决策。它不替代 `docs/status/` 的事实源；已验证结果仍同步写入 `PROJECT_COMPLETED.md`，待办和风险仍同步写入 `PROJECT_TODO.md`。

## 当前原则

1. 保留严格 EM 作为主指标，但报告中必须区分 strict EM false negative、答案粒度问题和真实检索/推理退化。
2. 不再把“少搜索”本身当作目标。目标是减少重复、空转、无收益搜索，同时保护必要 follow-up query。
3. Reward 变更必须先有离线诊断依据，再进入小预算训练。
4. 每个 reward 版本必须记录参数、预期影响、实际结果和是否继续采用。

## 版本记录

| Version | 状态 | 参数 / 机制 | 目的 | 已观测结果 | 决策 |
| --- | --- | --- | --- | --- | --- |
| `base_reward_v0` | 已完成 | `Answer:` 格式校验 + exact match；无额外 penalty | 建立 Search-R1 MiniLab 的原始 reward baseline | base Zhihu dev EM 0.2429；5-step 原始 reward EM 0.3000；20-step 原始 reward EM 0.2714、format 0.8571 | 继续作为主对照 |
| `penalty_v1` | 已完成 | `duplicate_query_penalty=0.05`；`empty_result_penalty=0.03`；`max_search_no_answer_penalty=0.05`；`verbose_answer_penalty=0.02`；`verbose_answer_token_threshold=8` | 降低重复搜索、空结果、过多搜索和冗长错误答案 | 5-step penalty 降低搜索次数和 `too_many_search_no_gain_rate`，但 EM 0.2286，低于 base 和 5-step 原始 reward | 作为负向参考，不直接扩大训练 |
| `diagnostic_v1` | 已实现并验证 | 离线标注 `possible_alias_match`、`answer_granularity_miss`、`missing_followup_query` | 区分 strict EM false negative、答案粒度问题和必要二跳搜索缺失 | 已在 base、5-step、penalty、20-step eval JSONL 上跑通；20-step 标出 1 条答案粒度问题和 6 条 missing follow-up 风险，覆盖 `test_97`、`dev_4869`、`dev_3412` 等人工 review 样本 | 进入默认复盘流程 |
| `sensitivity_v1` | 已实现并验证 | 离线重评分 `base_reward_v0`、`penalty_v1`、`penalty_v2_candidate`、`penalty_v2_no_empty`，并支持少量自定义配置 | 在训练前比较 penalty 配置对 reward 分布和误伤样本的影响 | 四份 Zhihu dev eval JSONL 上，`penalty_v1` mean delta 约 -0.0094 到 -0.0150；`penalty_v2_candidate` mean delta 约 -0.0004 到 -0.0014，未扣到 missing-followup、alias 或答案粒度样本 | 后续 reward 变更先跑 sensitivity，再决定是否训练 |
| `penalty_v2_candidate` | 已完成 5-step 与 20-step | `duplicate_query_penalty=0.03`、`empty_result_penalty=0.01`、关闭 `max_search_no_answer_penalty` 和 `verbose_answer_penalty`；后续可加 answer granularity penalty 或 follow-up query bonus | 保守减少坏搜索，避免压掉必要二跳 | 5-step Zhihu dev EM 0.2714、format 0.8286、平均搜索 1.7571；20-step EM 0.3571、format 0.7286、平均搜索 2.6143、`too_many_search_no_gain_rate=0.4286` | 20-step 作为最高 EM checkpoint 保留，但不能作为默认最优策略；下一步做 no-empty ablation 或重新约束 max-search/format |
| `penalty_v2_plus_max_search_001` | 已完成 5-step | `duplicate_query_penalty=0.03`、`empty_result_penalty=0.01`、`max_search_no_answer_penalty=0.01`，关闭 verbose | 压住 v2 20-step 暴露出的 max-search 空转和格式退化 | 5-step Zhihu dev EM 0.3000、format 0.8714、平均搜索 1.4571、`too_many_search_no_gain_rate=0.1000`；但 offline diagnostics 显示 `missing_followup_query=5`、`answer_granularity_miss=2` | 作为高 format/高效率候选保留；先做 gained/lost review 或更温和 0.005 版本，不直接推 50-step |
| `penalty_v2_plus_max_search_0005` | 已完成 5-step | `duplicate_query_penalty=0.03`、`empty_result_penalty=0.01`、`max_search_no_answer_penalty=0.005`，关闭 verbose | 验证比 0.01 更温和的 max-search penalty 是否减少过早回答和答案粒度风险 | 5-step Zhihu dev EM 0.2714、format 0.8143、平均搜索 1.5714、`too_many_search_no_gain_rate=0.1714`；offline diagnostics 显示 `missing_followup_query=5`、`answer_granularity_miss=0` | 不作为优先扩大训练候选；0.01 版本的 format/效率更好，但仍需 gained/lost review |
| `penalty_v3_followup_aware` | 已完成 5-step | `duplicate_query_penalty=0.03`、`empty_result_penalty=0.01`、`bad_max_search_penalty=0.01`、`date_granularity_penalty=0.05`、`multi_candidate_answer_penalty=0.02`；关闭传统 max-search 和 verbose | 只扣明显搜索空转，并约束日期粒度和多候选答案 | 5-step Zhihu dev EM 0.3000、format 0.8571、平均搜索 1.5143、`bad_max_search_loop=2`、`answer_granularity_miss=0`、`multi_candidate_answer=1`，但 `missing_followup_query=6` | final answer 粒度/唯一性方向有效；follow-up 风险未解决。下一步做正向 follow-up bonus 或 prompt/rollout 约束 |
| `reward_v4_followup_bonus` | 已完成 5-step | 在 v3 基础上增加 `helpful_followup_bonus=0.02`；正确答案和 invalid format 不加 bonus | 从 penalty-only 转向正向行为信号，鼓励已经出现的有用 follow-up query | 离线预检查 `correct_boosted=0`；5-step dev EM 0.2714、format 0.7714、平均搜索 1.6286，`missing_followup_query=4`、`answer_granularity_miss=0`，Zhihu success rate 1.0 | 不扩大到 20/50-step；follow-up bonus 有局部收益但 format 退化，下一步优先做 prompt/rollout 约束或 `group_size=8` 稳定性对照 |
| `penalty_v3_group_size8` | 已完成 5-step | 沿用 v3 reward，`group_size=8`、`questions_per_batch=2` | 验证更大 rollout group 是否降低 5-step 方差 | 5-step dev EM 0.2714、format 0.6857、平均搜索 2.1143，`helpful_followup_query_rate=0.4429`、`missing_followup_query=3`，Zhihu success rate 1.0 | 不扩大训练；helpful follow-up 增加但 format/search 明显退化，说明单纯增大 group 不能替代更强的 format/final-answer 约束 |
| `prompt_search_first` | 已完成 dev 70 | system prompt 明确要求最终回答前先看到至少一次 search result，不要凭记忆直接答 | 修复 prompt+v3 训练暴露出的 no-search 退化，并继续保护必要 follow-up | base Zhihu dev EM 0.3714、format 0.8000、平均搜索 2.0571、`missing_followup_query=0`、`multi_candidate_answer=0`、`answer_granularity_miss=0`、`bad_max_search_loop=3`，Zhihu success rate 1.0 | 当前最高 EM base，保留；下一步重点补 format/date completeness，而不是继续增强 follow-up |
| `reward_v5_no_search_guard` | 已完成实现、离线 sensitivity、20-step；50-step 尝试中断 | `duplicate_query_penalty=0.02`、`empty_result_penalty=0.0`、`bad_max_search_penalty=0.005`、`date_granularity_penalty=0.05`、`multi_candidate_answer_penalty=0.02`、`no_search_penalty=0.03` | 防止 prompt+v3 训练学成“格式正确但过早不搜”，同时轻度约束 bad-loop | 离线检查在 prompt+v3 失败 checkpoint 上扣到 20 个 no-search wrong-valid，未扣正确；20-step final dev EM 0.3000、format 0.7286、平均搜索 2.5857、`missing_followup_query=0`、`answer_granularity_miss=5`；50-step 首次尝试在 step 41 因 prompt reconstruction failure 中断，已修复为单 trajectory stop reason | 不扩大到 50-step；no-search guard 防住了 no-search collapse，但诱发过度搜索和答案粒度退化。后续优先做 format/date completeness 约束 |
| `prompt_search_budget_guard` | 已完成 dev-5/dev 70 | 在 `prompt_search_first` 基础上增加搜索预算提醒：尽量 3 次搜索内完成，3 次后用最佳证据输出短答案，不继续请求第 5 次搜索 | 保留 search-first 与必要 follow-up，同时修复 max-search 类 format 失败和平均搜索偏高 | Zhihu dev EM 0.4143、format 0.8857、平均搜索 1.9429、no-search 0、`missing_followup_query=0`、`multi_candidate_answer=0`、`answer_granularity_miss=0`、`bad_max_search_loop=4`，Zhihu success rate 1.0；相对 `prompt_search_first` gained 4/lost 1，invalid format 14 降到 8 | 当前最强 prompt base；暂不做 50-step 训练。后续优先做多 seed/搜索预算 ablation 或基于该 prompt 的 5-step smoke，门槛不低于 EM 0.4143、format 0.8857 |
| `grpo_kl_std_5step` | 已完成实现和 5-step，已设为训练默认 | 在 `prompt_search_budget_guard` 上启用 `advantage_normalization=standardize`、`advantage_clip=2.0`、`kl_coef=0.01`、`policy_ratio_clip=0.2`、`learning_rate=1e-5`；KL 为 sampled-token logprob drift penalty | 抑制训练后策略漂移，同时稳定不同 group 的 advantage 尺度 | 5-step final dev EM 0.4286、format 0.9429、平均搜索 1.9429、no-search 0、`missing_followup_query=0`、`answer_granularity_miss=0`、`multi_candidate_answer=0`，相对 prompt-only best gained 2/lost 1、invalid format 8 降到 4；后续消融因 PyTRIO sampling 阻塞未完成，项目暂将 KL/std 作为合理必备技术手段 | 当前最强 checkpoint 证据；默认用于后续训练，PyTRIO 恢复后优先跑 20-step；不直接 50-step |

## Offline Diagnostics

`diagnostic_v1` 是下一轮 reward 改造前的低成本检查。它只读取已有 eval JSONL，不调用模型、搜索 API 或训练服务。

三类标注：

- `possible_alias_match`：strict EM 判错，但 final answer 与 gold 可能是别名、全名/短名或拼写变体，例如 `Dexter King` vs `Dexter`。
- `answer_granularity_miss`：模型推理基本包含正确信息，但 final answer 粒度不足，例如 gold 是 `October 2, 1869`，final answer 只写 `1869`。
- `missing_followup_query`：单次搜索后在多跳或实体角色题上过早作答，可能缺少必要 follow-up query，例如先找到母亲或 writer 后没有继续搜关键中间实体。

运行示例：

```bash
PYTHONPATH=my-search-r1 uv run python my-search-r1/scripts/analyse_offline_diagnostics.py \
  --input my-search-r1/eval_results/reward_train_compare_2026-07-21/base_20step_dev.jsonl \
  --jsonl-output my-search-r1/eval_results/reward_train_compare_2026-07-21/base_20step_offline_diagnostics.jsonl \
  --report-output my-search-r1/eval_results/reward_train_compare_2026-07-21/base_20step_offline_diagnostics.md \
  --title '20-step Offline Diagnostics'
```

## Reward Sensitivity

`sensitivity_v1` 复用已有 eval trajectory JSONL，按不同 penalty 配置离线重算 `base_reward` 和 `final_reward`。它用于训练前 sanity check：确认候选 penalty 是否主要扣到重复、空结果等低质量行为，而不是扣到正确样本、必要 follow-up query、strict EM false negative 或答案粒度问题。

默认配置：

- `base_reward_v0`：所有 penalty 为 0。
- `penalty_v1`：duplicate 0.05、empty 0.03、max-search 0.05、verbose 0.02、verbose threshold 8。
- `penalty_v2_candidate`：duplicate 0.03、empty 0.01，关闭 max-search 和 verbose。
- `penalty_v2_no_empty`：duplicate 0.03，关闭 empty、max-search 和 verbose。

运行示例：

```bash
PYTHONPATH=my-search-r1 uv run python my-search-r1/scripts/analyse_reward_sensitivity.py \
  --input my-search-r1/eval_results/reward_train_compare_2026-07-21/base_20step_dev.jsonl \
  --summary-output my-search-r1/eval_results/reward_train_compare_2026-07-21/base_20step_reward_sensitivity_summary.json \
  --jsonl-output my-search-r1/eval_results/reward_train_compare_2026-07-21/base_20step_reward_sensitivity.jsonl \
  --report-output my-search-r1/eval_results/reward_train_compare_2026-07-21/base_20step_reward_sensitivity.md \
  --title '20-step Reward Sensitivity'
```

小规模自定义配置示例：

```bash
PYTHONPATH=my-search-r1 uv run python my-search-r1/scripts/analyse_reward_sensitivity.py \
  --input my-search-r1/eval_results/reward_train_compare_2026-07-21/base_20step_dev.jsonl \
  --config 'dup_only:duplicate=0.02,empty=0,max_search=0,verbose=0,verbose_threshold=0' \
  --summary-output /tmp/reward_sensitivity_summary.json \
  --jsonl-output /tmp/reward_sensitivity.jsonl \
  --report-output /tmp/reward_sensitivity.md
```

2026-07-21 对四份既有 Zhihu dev eval JSONL 的离线结果：

| Run | Config | mean_delta | penalized | correct penalized | wrong-valid penalized | missing-followup penalized | alias penalized | granularity penalized |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base | `penalty_v1` | -0.0150 | 23 | 0 | 8 | 0 | 4 | 0 |
| base | `penalty_v2_candidate` | -0.0009 | 4 | 0 | 0 | 0 | 0 | 0 |
| 5-step 原始 reward | `penalty_v1` | -0.0106 | 15 | 0 | 7 | 1 | 1 | 0 |
| 5-step 原始 reward | `penalty_v2_candidate` | -0.0014 | 5 | 0 | 3 | 0 | 0 | 0 |
| 5-step penalty | `penalty_v1` | -0.0100 | 16 | 0 | 8 | 1 | 2 | 0 |
| 5-step penalty | `penalty_v2_candidate` | -0.0009 | 4 | 0 | 1 | 0 | 0 | 0 |
| 20-step 原始 reward | `penalty_v1` | -0.0094 | 15 | 0 | 6 | 1 | 1 | 0 |
| 20-step 原始 reward | `penalty_v2_candidate` | -0.0004 | 3 | 0 | 0 | 0 | 0 | 0 |

解读：

- `penalty_v1` 在 4 份 JSONL 上都会扣到 wrong-valid 样本，并在 5-step/20-step 上扣到 missing-followup 风险样本，不适合直接扩大训练。
- `penalty_v2_candidate` 明显更温和，主要只扣重复或空结果；当前四组结果中没有扣到正确样本、missing-followup、alias 或答案粒度样本。
- 离线 verbose penalty 使用 final answer 的 whitespace word count 近似 tokenizer token count；真实训练仍以 tokenizer token count 为准。

## 下一轮实验计划

### Step 1: 离线诊断基线

对以下 JSONL 分别运行 `diagnostic_v1`：

- base：`base_dev.jsonl`
- 5-step 原始 reward：`baseline_reward_ckpt_dev.jsonl`
- 5-step penalty：`penalty_reward_ckpt_dev.jsonl`
- 20-step 原始 reward：`base_20step_dev.jsonl`

验收状态：已完成一次 2026-07-21 离线验证。

| Run | wrong_valid | possible_alias_match | answer_granularity_miss | missing_followup_query |
| --- | ---: | ---: | ---: | ---: |
| base | 26 | 10 | 0 | 0 |
| 5-step 原始 reward | 37 | 9 | 0 | 3 |
| 5-step penalty | 35 | 10 | 0 | 2 |
| 20-step 原始 reward | 41 | 8 | 1 | 6 |

后续验收：

- 每个 checkpoint 都有 diagnostic summary。
- 能把 strict EM false negative 与真实检索/推理退化分开记录。
- 更新下一轮 penalty 参数前，必须引用 diagnostic 结果。

### Step 2: 保守 Penalty 小实验

候选配置：

```text
duplicate_query_penalty=0.03
empty_result_penalty=0.01
max_search_no_answer_penalty=0.00
verbose_answer_penalty=0.00
verbose_answer_token_threshold=0
```

训练配置先用：

- `max_steps=5`
- `questions_per_batch=2`
- `group_size=4`
- `backend=zhihu_search`
- `seed=42`

停止条件：

- EM 不低于 base 的 0.2429，优先接近或超过 5-step 原始 reward 的 0.3000。
- format rate 不低于 0.82。
- `too_many_search_no_gain_rate` 低于 base。
- `missing_followup_query` 不高于 20-step 原始 reward。
- `answer_granularity_miss` 不上升。

验收状态：2026-07-21 已完成一次 5-step 训练与 Zhihu dev 70 eval，并完成一次 20-step 稳定性扩展。

| 指标 | 结果 |
| --- | ---: |
| `em/macro` | 0.2714 |
| `format/rate` | 0.8286 |
| `rollout/search_calls` | 1.7571 |
| `behavior/too_many_search_no_gain_rate` | 0.2143 |
| `missing_followup_query` | 2 |
| `answer_granularity_miss` | 0 |

决策：

- 满足进入 `penalty_v2_candidate` 20-step 小扩展的最低门槛。
- 仍低于 5-step 原始 reward 的 EM 0.3000，因此 20-step 应作为稳定性验证，而不是直接替代当前最强 checkpoint。
- `penalty_v2_no_empty` 在本 checkpoint 上离线不扣分，可作为后续因果 ablation。

### Step 3: Penalty v2 20-step 稳定性扩展

验收状态：2026-07-21 已完成。

| 指标 | 5-step penalty v2 | 20-step penalty v2 |
| --- | ---: | ---: |
| `em/macro` | 0.2714 | 0.3571 |
| `format/rate` | 0.8286 | 0.7286 |
| `rollout/search_calls` | 1.7571 | 2.6143 |
| `behavior/too_many_search_no_gain_rate` | 0.2143 | 0.4286 |
| `missing_followup_query` | 2 | 1 |
| `answer_granularity_miss` | 0 | 0 |

决策：

- 20-step penalty v2 是当前最高 EM checkpoint，但搜索行为和 format 明显退化。
- 这说明 v2 的 duplicate/empty penalty 仍不足以约束长训练中的 max-search 空转；继续增加 step 不应作为下一步默认策略。
- 后续优先做 `penalty_v2_no_empty` ablation、重引入更温和的 max-search no-answer penalty，或增加 format/follow-up query 正向信号。

### Step 4: Max-search 轻量约束

验收状态：2026-07-21 已完成 `max_search_no_answer_penalty=0.01` 与 `0.005` 的 5-step 小实验。

| 指标 | 5-step penalty v2 | 20-step penalty v2 | 5-step v2 + max-search 0.01 | 5-step v2 + max-search 0.005 |
| --- | ---: | ---: | ---: | ---: |
| `em/macro` | 0.2714 | 0.3571 | 0.3000 | 0.2714 |
| `format/rate` | 0.8286 | 0.7286 | 0.8714 | 0.8143 |
| `rollout/search_calls` | 1.7571 | 2.6143 | 1.4571 | 1.5714 |
| `behavior/max_search_no_answer_rate` | 0.1143 | 0.2714 | 0.0857 | 0.1143 |
| `behavior/too_many_search_no_gain_rate` | 0.2143 | 0.4286 | 0.1000 | 0.1714 |
| `missing_followup_query` | 2 | 1 | 5 | 5 |
| `answer_granularity_miss` | 0 | 0 | 2 | 0 |

决策：

- `max_search_no_answer_penalty=0.01` 能显著改善 format 和搜索效率，并把 EM 拉回 0.3000。
- 但它也提高了 missing follow-up 和答案粒度风险，说明惩罚仍可能压掉必要二跳或诱导过早回答。
- `max_search_no_answer_penalty=0.005` 消除了本轮观测到的答案粒度风险，但没有改善 missing follow-up，且 EM、format 和搜索效率均弱于 0.01。
- 2026-07-21 gained/lost review 显示，两个 max-search 版本共同修复了 `test_108`、`dev_2223` 这类 v2_20 的 max-search 空转；共同丢失了 `dev_4869`、`dev_174` 等需要 follow-up 或更强 role binding 的样本。`max001` 独有正确 `dev_3412`，但在 `test_97`、`test_99` 上出现日期粒度退化；`max0005` 修复了这两条日期粒度问题，但整体 EM/format 更低。
- 下一步不建议继续扫单一 `max_search_no_answer_penalty` 标量；优先做 follow-up-aware max-search penalty、final answer 唯一性/日期粒度约束，或做 `penalty_v2_no_empty`、`group_size=8` 对照。

### Step 5: Follow-up-aware v3

验收状态：2026-07-21 已完成实现、离线 sensitivity 和 5-step 小实验。

| 指标 | max001 | max0005 | v3 |
| --- | ---: | ---: | ---: |
| `em/macro` | 0.3000 | 0.2714 | 0.3000 |
| `format/rate` | 0.8714 | 0.8143 | 0.8571 |
| `rollout/search_calls` | 1.4571 | 1.5714 | 1.5143 |
| `behavior/bad_max_search_loop_rate` | 0.0000 | 0.0429 | 0.0286 |
| `missing_followup_query` | 5 | 5 | 6 |
| `answer_granularity_miss` | 2 | 0 | 0 |
| `multi_candidate_answer` | 2 | 3 | 1 |

决策：

- v3 追平 max001 的 EM，并显著减少日期粒度和多候选答案问题。
- v3 没有解决必要二跳缺失，`missing_followup_query` 反而升至 6。
- 下一版应从 penalty-only 转向正向 follow-up 信号：对多跳/关系题中引入关键中间实体的 query 给小 bonus，或在 rollout prompt 中要求先锁定中间实体再回答。

### Step 6: Follow-up bonus v4

验收状态：2026-07-22 已完成实现、离线 sensitivity、5-step 训练和 dev 70 eval。

候选配置：

```text
duplicate_query_penalty=0.03
empty_result_penalty=0.01
bad_max_search_penalty=0.01
date_granularity_penalty=0.05
multi_candidate_answer_penalty=0.02
helpful_followup_bonus=0.02
```

离线预检查结果：

| Run | mean_delta | helpful_bonus | boosted | correct_boosted | wrong_valid_boosted | missing_followup_boosted |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `v2_20` | 0.0001 | 14 | 14 | 0 | 14 | 0 |
| `max001_5step` | -0.0003 | 7 | 7 | 0 | 7 | 0 |
| `max0005_5step` | -0.0010 | 5 | 5 | 0 | 5 | 0 |
| `v3_5step` | 0.0006 | 7 | 7 | 0 | 7 | 0 |

决策：

- v4 没有给正确样本加分，离线影响幅度很小，因此已进入 5-step 小预算训练。
- 该 bonus 不会直接修复已标为 `missing_followup_query` 的单搜索过早回答；它的作用是训练时鼓励模型保留有用 follow-up 行为。
- 5-step 结果为 EM 0.2714、format 0.7714、平均搜索 1.6286、`missing_followup_query=4`、`answer_granularity_miss=0`，Zhihu API 无错误。
- 未达 `EM >= 0.3000` 和 `format >= 0.82` 门槛，因此不扩大到 20-step 或 50-step。
- gained/lost 显示 v4 能修复 `dev_3412`、`test_97` 等需要 follow-up 或答案粒度的样本，但同时丢失 `dev_6748`、`test_494`、`test_4020` 等格式、证据读取和答案唯一性样本。

### Step 7: Group-size 8 stability check

验收状态：2026-07-22 已完成 v3 reward、`group_size=8` 的 5-step 训练和 dev 70 eval。

| 指标 | v3 group 4 | v3 group 8 |
| --- | ---: | ---: |
| `em/macro` | 0.3000 | 0.2714 |
| `format/rate` | 0.8571 | 0.6857 |
| `rollout/search_calls` | 1.5143 | 2.1143 |
| `behavior/helpful_followup_query_rate` | 0.2143 | 0.4429 |
| `missing_followup_query` | 6 | 3 |
| `behavior/too_many_search_no_gain_rate` | 0.1286 | 0.3143 |

决策：

- group size 8 提高了 helpful follow-up 行为率，并降低了 missing follow-up 诊断数。
- 但 format、搜索效率和 EM 明显退化，不应继续扩大。
- 这说明问题不只是 rollout 方差；下一步需要更强的 final-answer format/唯一性约束，或 prompt/rollout 层的中间实体锁定流程。

### Step 8: Search budget guard prompt

验收状态：2026-07-22 已完成 prompt-only dev-5 和 dev 70 eval。

| 指标 | prompt constraints | prompt search-first | prompt search-budget guard |
| --- | ---: | ---: | ---: |
| `em/macro` | 0.3429 | 0.3714 | 0.4143 |
| `format/rate` | 0.8714 | 0.8000 | 0.8857 |
| `rollout/search_calls` | 1.7143 | 2.0571 | 1.9429 |
| `rollout/no_search_rate` | 0.0000 | 0.0000 | 0.0000 |
| `missing_followup_query` | 3 | 0 | 0 |
| `answer_granularity_miss` | 0 | 0 | 0 |
| `multi_candidate_answer` | 0 | 0 | 0 |
| `bad_max_search_loop` | 5 | 3 | 4 |

决策：

- `prompt_search_budget_guard` 同时改善 EM、format 和平均搜索次数，是当前最强 base。
- 该 prompt 没有牺牲 `missing_followup_query=0`，说明“3 次后作答”的约束没有明显压掉必要 follow-up。
- 暂停继续扩大 v5/reward 训练；已有 prompt-only 结果优于当前训练 checkpoint。
- 后续更适合做多 seed 稳定性、搜索预算文案 ablation，或只在该 prompt 上跑 5-step smoke，并以 EM 0.4143、format 0.8857 为新门槛。

### Step 9: KL/std-stabilized GRPO

验收状态：2026-07-22 已完成实现、本地 smoke、5-step Zhihu 训练和 dev 70 eval。

配置：

```text
advantage_normalization=standardize
advantage_clip=2.0
kl_coef=0.01
policy_ratio_clip=0.2
learning_rate=1e-5
max_steps=5
group_size=4
questions_per_batch=2
```

| 指标 | prompt search-budget guard | prompt budget + KL/std 5-step |
| --- | ---: | ---: |
| `em/macro` | 0.4143 | 0.4286 |
| `format/rate` | 0.8857 | 0.9429 |
| `rollout/search_calls` | 1.9429 | 1.9429 |
| `rollout/no_search_rate` | 0.0000 | 0.0000 |
| `missing_followup_query` | 0 | 0 |
| `answer_granularity_miss` | 0 | 0 |
| `multi_candidate_answer` | 0 | 0 |
| `bad_max_search_loop` | 4 | 4 |

决策：

- 训练方向重新通过门控：在最强 prompt base 上加 std normalization、clip、KL-style reference 约束和更低学习率后，5-step 可以超过 prompt-only。
- 当前收益主要体现在 format 改善和少量 EM 增益；平均搜索没有变高。
- 不能直接归因到 KL 或 std 的单项效果；`std+clip only` 与 `KL only` ablation 因 PyTRIO sampling 阻塞未完成。
- 用户已决策将 KL/std 组合作为项目当前合理必备技术手段；`train_pytrio.py` 默认值改为 `standardize + clip 2.0 + kl_coef 0.01 + ratio clip 0.2 + lr 1e-5`。
- PyTRIO sampling 恢复后优先扩到 20-step，必须 `--save-every 5` 或更频繁，并以 EM 0.4286、format 0.9429、`missing_followup_query=0` 作为新门槛。

## 面试叙事

当前 reward 改造不是盲调 penalty，而是一个闭环：

```text
训练/评测 trajectory -> gained/lost review -> offline diagnostic 分类错因
-> 设计更保守的 reward -> 小预算对照实验 -> 更新下一轮决策
```

这个闭环能说明项目的重点不是单点指标，而是可观测、可解释、可迭代的 Agentic RL 实验方法。
