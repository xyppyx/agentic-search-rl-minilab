# Turn-Level Evidence Credit v2 实现与 5-Step 评测复盘

记录时间：2026-07-22 19:35 CST

## 目标

在 `helpful_bridge` v1 的基础上，将 turn-level credit 从 query 形态升级到 evidence/bridge/entity 级别，并增加小权重 early-answer penalty：

- `evidence_bridge_search`：wrong-valid 轨迹中，第 2 次及之后的 search turn 需要命中前序 observation 或题面中的 bridge entity，并在当前 observation 中看到答案或关系/实体证据。
- `early_answer_missing_followup`：wrong-valid、单次搜索后直接回答、题目存在多跳/role-binding cue 且 observation 中有多个候选时，对 final answer turn 加小负向 advantage。
- 训练默认仍沿用当前 prompt、std normalization、advantage clip、KL-style reference 约束和 `learning_rate=1e-5`。

## 实现与回归

新增/修改内容：

- 新增 `search_r1_minilab/turn_credit.py`，集中实现 v1 shape、v2 evidence bridge 和 early-answer risk 的共享判定。
- `train_pytrio.py` 新增 `--turn-credit-policy evidence_bridge`、`--evidence-search-turn-bonus`、`--early-answer-turn-penalty`。
- 新增 `scripts/analyse_turn_credit.py`，读取既有 train/eval JSONL，输出 would-credit / would-penalize JSONL 与 Markdown，不调用模型、搜索 API 或训练服务。
- `trajectory_to_record` 继续在 metadata 中记录 `turn_credits`，本轮支持正向和负向 label。

已运行：

```bash
PYTHONPATH=my-search-r1 uv run python -m unittest discover -s my-search-r1/tests -v
PYTHONPATH=my-search-r1 uv run python -m compileall -q my-search-r1/search_r1_minilab my-search-r1/scripts
PYTHONPATH=my-search-r1 uv run python my-search-r1/scripts/train_pytrio.py --help
PYTHONPATH=my-search-r1 uv run python my-search-r1/scripts/analyse_turn_credit.py --help
```

结果：

- 完整 unittest：77 个测试全部通过。
- compileall 通过。
- CLI help 确认 `evidence_bridge`、`--evidence-search-turn-bonus`、`--early-answer-turn-penalty` 已接入。

## Offline Credit Analysis

对历史 `turn_credit_helpful_bridge_5step` 和 `turn_credit_helpful_bridge_20step` dev 70 输出运行 `analyse_turn_credit.py`。

5-step 历史输出：

- total 70，wrong-valid 38。
- v1 shape candidate turns 39，v1 training credit turns 17。
- v2 evidence candidate records 32，candidate turns 54。
- v2 training credit records 13，training credit turns 23。
- early-answer penalty records 0。
- 关键二跳样本均被 evidence candidate 命中：`dev_4869`、`dev_3741`、`dev_2429`、`dev_174`、`test_7511`。

20-step 历史输出：

- total 70，wrong-valid 44。
- v2 evidence candidate records 23，candidate turns 39。
- v2 training credit records 11，training credit turns 19。
- early-answer penalty records 6。
- 历史 20-step lost case 均被 early-answer risk 命中：`dev_4869`、`dev_3741`、`dev_2429`、`dev_174`、`test_7511`；额外命中 `dev_6133`，同样是单跳多跳/role-binding 风险。

产物：

- `my-search-r1/eval_results/reward_train_compare_2026-07-21/turn_credit_evidence_v2_analysis_5step_20260722.jsonl`
- `my-search-r1/eval_results/reward_train_compare_2026-07-21/turn_credit_evidence_v2_analysis_5step_20260722.md`
- `my-search-r1/eval_results/reward_train_compare_2026-07-21/turn_credit_evidence_v2_analysis_20step_20260722.jsonl`
- `my-search-r1/eval_results/reward_train_compare_2026-07-21/turn_credit_evidence_v2_analysis_20step_20260722.md`

## Local Smoke

运行 local BM25 1-step smoke，参数为 `--turn-credit-policy evidence_bridge`、`--evidence-search-turn-bonus 0.10`、`--early-answer-turn-penalty 0.05`，并启用 std/clip/KL 配置。

结果：

- 成功进入 reference logprobs、custom loss、optimizer 和 final checkpoint 路径。
- `mean_reward=0.5000`、`correct_rate=0.5000`、平均搜索 1.0。
- smoke fixture 是单跳问题，未触发 turn credit，符合预期。

## Zhihu 5-Step

训练配置：

- run name：`turn-credit-evidence-bridge-5step-20260722`
- backend：`zhihu_search`
- steps：5
- group size：4
- questions per batch：2
- policy：`evidence_bridge`
- evidence bonus：0.10
- early-answer penalty：0.05
- std normalization、advantage clip 2.0、KL coef 0.01、ratio clip 0.2、learning rate 1e-5
- SwanLab：disabled

训练结果：

- 5/5 step 完成，5/5 step 执行 optimizer。
- 40 条训练 trajectory，训练平均 reward 0.1125、correct rate 0.1250、format rate 0.8750、平均搜索 2.0750。
- 训练 JSONL 记录 24 个 `evidence_bridge_search` 和 4 个 `early_answer_missing_followup`。
- 训练阶段未观察到 PyTRIO sampling 长时间 await、Zhihu timeout、429、credential/http error 或 tool failure。

Dev-5：

- EM 0.4000
- format 0.8000
- 平均搜索 3.0000
- Zhihu requests 15，success rate 1.0

完整 dev 70：

- EM 0.4286，30/70
- format 0.9429
- 平均搜索 1.8143
- no-search rate 0.0143
- helpful follow-up rate 0.4000
- bad max-search loop rate 0.0286
- too many search no gain rate 0.1571
- Zhihu requests 127，success rate 1.0，error/timeout/rate-limit 均为 0
- offline diagnostics：`wrong_valid=36`、`possible_alias_match=8`、`missing_followup_query=0`、`answer_granularity_miss=0`、`bad_max_search_loop=2`、`multi_candidate_answer=0`

Gained/lost：

- 相对 `prompt_budget_kl_std_5step`：gained 1 `test_7511`，lost 1 `dev_7773`。
- 相对 `turn_credit_helpful_bridge_5step`：gained 1 `test_2231`，lost 1 `dev_7773`。
- 四个明确二跳保护样本 `dev_4869`、`dev_2429`、`dev_3741`、`dev_174` 均保持正确。

## 结论

`evidence_bridge` v2 通过了离线门槛，并在 5-step 训练中实际产生了正向 evidence credit 和 early-answer penalty。最终 dev 70 EM 追平当前最高 0.4286，平均搜索优于 `prompt_budget_kl_std_5step` 和 `turn_credit_helpful_bridge_5step`，且没有重新引入 missing follow-up。

但本轮 format 为 0.9429，低于预设 0.95 门槛，因此原始判断是不应直接跑 20-step 或 50-step。随后按用户要求已完成 20-step 扩展，结果见 `docs/interview/lesson/2026-07-22_turn_level_evidence_credit_v2_20step_eval.md`：20-step v2 dev 70 EM 0.4429、format 1.0000、平均搜索 1.7714、`missing_followup_query=0`，相对 5-step v2 gained 1/lost 0，相对 `helpful_bridge` 20-step 拿回 5 个关键二跳/role-binding lost case。最新决策更新为：20-step v2 可作为当前最强 checkpoint 证据，但仍不直接扩到 50-step，下一步优先做多 seed 稳定性或补 bad-loop/final-answer 约束。
