# Turn-Level Evidence Credit v2 20-Step 扩展复盘

记录时间：2026-07-22 19:58 CST

## 背景

`turn_credit_evidence_bridge_v2` 的 5-step 结果为 EM 0.4286、format 0.9429、平均搜索 1.8143。由于 format 低于预设 0.95 门槛，原计划不直接扩到 20-step；本轮按用户要求继续做 20-step 扩展，用于验证 v2 是否能修复 `helpful_bridge` 20-step 中出现的过早单跳回答和二跳样本丢失。

## 训练配置

- run name：`turn-credit-evidence-bridge-20step-20260722`
- backend：`zhihu_search`
- steps：20
- group size：4
- questions per batch：2
- policy：`evidence_bridge`
- evidence bonus：0.10
- early-answer penalty：0.05
- advantage normalization：standardize
- advantage clip：2.0
- KL coef：0.01
- policy ratio clip：0.2
- learning rate：1e-5
- SwanLab：disabled

## 训练结果

- 20/20 step 完成，160 条训练 trajectory 生成。
- 训练阶段未观察到 Zhihu timeout、429、credential/http error、tool failure 或 PyTRIO sampling 长时间阻塞。
- 训练平均 reward 0.3312、correct rate 0.3375、format rate 0.9375、平均搜索 1.8000。
- 训练 metadata 记录 49 个 `evidence_bridge_search` 和 4 个 `early_answer_missing_followup`。
- 15/20 个 step 出现 turn credit；step 8 出现无有效 loss datum 的 skipped-like step。
- step 19 训练批内 reward/correct 较高，但 step 20 reward/correct 回落到 0，说明训练批内波动仍明显，不能只看末端训练 reward 判断泛化。

## Dev 评测

Dev-5：

- EM 0.4000
- format 1.0000
- 平均搜索 2.6000
- helpful follow-up rate 0.8000
- bad max-search loop rate 0.2000
- too many search no gain rate 0.4000
- Zhihu requests 13，success rate 1.0，error/timeout/rate-limit 均为 0

完整 dev 70：

- EM 0.4429，31/70
- format 1.0000
- 平均搜索 1.7714
- no-search rate 0
- helpful follow-up rate 0.4000
- bad max-search loop rate 0.0429
- too many search no gain rate 0.1571
- empty observation rate 0.0403
- Zhihu requests 124，success rate 1.0，error/timeout/rate-limit 均为 0

Offline diagnostics：

- wrong-valid 39
- possible alias match 7
- missing follow-up query 0
- answer granularity miss 0
- multi-candidate answer 0
- bad max-search loop 3

Turn-credit analysis on final dev 70：

- evidence candidate records 32，candidate turns 48
- training-credit-eligible records 15，evidence training credit turns 24
- v1 shape candidate turns 37，v1 training credit turns 18
- early-answer penalty records 0

## Gained / Lost

相对 `turn_credit_evidence_bridge_5step`：

- gained：`dev_7773`
- lost：无

相对 `prompt_budget_kl_std_5step`：

- gained：`test_7511`
- lost：无

相对 `turn_credit_helpful_bridge_5step`：

- gained：`test_2231`
- lost：无

相对 `turn_credit_helpful_bridge_20step`：

- gained：`dev_4869`、`dev_2429`、`dev_3741`、`dev_174`、`test_7511`
- lost：无

关键样本：

- `dev_4869` 正确，2 次 search，答案 `Sextus Aelius Catus`。
- `dev_2429` 正确，2 次 search，答案 `42.5`。
- `dev_3741` 正确，2 次 search，答案 `Dziga Vertov`。
- `dev_174` 正确，2 次 search，答案 `Clifton College`。
- `test_7511` 正确，2 次 search，答案 `Yoon Seok-ho`。
- `test_2231` 正确，1 次 search，答案 `2013`。
- `dev_7773` 正确，3 次 search，答案 `Frankenstein'S Daughter`。

## 结论

20-step v2 通过本轮扩展验证：EM 从 5-step v2 的 0.4286 提升到 0.4429，format 从 0.9429 提升到 1.0000，平均搜索从 1.8143 降到 1.7714，且 `missing_followup_query` 仍为 0。相比 `helpful_bridge` 20-step，v2 明确修复了 5 个二跳/role-binding lost case，说明 evidence/bridge/entity 级 credit 加 early-answer penalty 的方向比纯 query 形态 credit 更能保护多跳二跳。

但不建议继续直接扩到 50-step。剩余风险是训练批内波动仍大、dev-5 已出现 bad loop/过搜信号、完整 dev 70 中 bad max-search loop 为 3 且有 4.03% empty observation。下一步优先做 5-step/20-step 多 seed 稳定性，或在 v2 上补 final answer/format guard 与 bad-loop 轻约束，再决定是否做更长步数。

## 产物

- `my-search-r1/outputs/train_pytrio/turn-credit-evidence-bridge-20step-20260722/`
- `my-search-r1/eval_results/reward_train_compare_2026-07-21/turn_credit_evidence_bridge_20step_dev5_20260722.jsonl`
- `my-search-r1/eval_results/reward_train_compare_2026-07-21/turn_credit_evidence_bridge_20step_dev5_20260722.md`
- `my-search-r1/eval_results/reward_train_compare_2026-07-21/turn_credit_evidence_bridge_20step_dev_20260722.jsonl`
- `my-search-r1/eval_results/reward_train_compare_2026-07-21/turn_credit_evidence_bridge_20step_dev_20260722.md`
- `my-search-r1/eval_results/reward_train_compare_2026-07-21/turn_credit_evidence_bridge_20step_dev_20260722_offline_diagnostics.jsonl`
- `my-search-r1/eval_results/reward_train_compare_2026-07-21/turn_credit_evidence_bridge_20step_dev_20260722_offline_diagnostics.md`
- `my-search-r1/eval_results/reward_train_compare_2026-07-21/turn_credit_evidence_bridge_20step_turn_credit_analysis_20260722.jsonl`
- `my-search-r1/eval_results/reward_train_compare_2026-07-21/turn_credit_evidence_bridge_20step_turn_credit_analysis_20260722.md`
