# Turn-Level Evidence Credit v2 50-Step 评测复盘

记录时间：2026-07-22 21:21 CST

## 背景

`turn_credit_evidence_bridge_v2` 20-step 已达到 dev 70 EM 0.4429、format 1.0000、平均搜索 1.7714。按用户要求继续直接重试 50-step。此前几次 50-step 尝试卡在 PyTRIO actor 初始化，本次 `turn-credit-evidence-bridge-50step-20260722-retry4` 成功越过 `prepare sampler` 并完整跑完。

## 配置

- run name：`turn-credit-evidence-bridge-50step-20260722-retry4`
- backend：`zhihu_search`
- steps：50
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
- save every：5
- SwanLab：disabled

## 训练结果

- 50/50 step 完成，400 条训练 trajectory 生成。
- 保存 step 5/10/15/20/25/30/35/40/45/50 和 final 权重。
- 训练阶段未观察到 Zhihu timeout、429、credential/http error、tool failure 或 PyTRIO sampling 阻塞。
- 训练平均 reward 0.3677、correct rate 0.3700、format rate 0.9775、平均搜索 1.4225。
- 训练 metadata 记录 76 个 `evidence_bridge_search` 和 28 个 `early_answer_missing_followup`。
- 29/50 个 step 出现 turn credit。
- 长训练后部分 batch 的平均搜索降到 0.62-1.0，并出现多次 skipped loss；说明 50-step 有向短路径/低搜索漂移的迹象。

## Dev 评测

Dev-5：

- EM 0.4000
- format 1.0000
- 平均搜索 2.8000
- helpful follow-up rate 0.8000
- bad max-search loop rate 0.2000
- too many search no gain rate 0.4000
- Zhihu requests 14，success rate 1.0，error/timeout/rate-limit 均为 0

完整 dev 70：

- EM 0.4000，28/70
- format 1.0000
- 平均搜索 1.7429
- no-search rate 0
- helpful follow-up rate 0.3714
- bad max-search loop rate 0.0429
- too many search no gain rate 0.1429
- empty observation rate 0.0246
- Zhihu requests 122，success rate 1.0，error/timeout/rate-limit 均为 0

Offline diagnostics：

- wrong-valid 42
- possible alias match 9
- missing follow-up query 0
- answer granularity miss 0
- multi-candidate answer 0
- bad max-search loop 3

Turn-credit analysis on final dev 70：

- evidence candidate records 30，candidate turns 48
- training-credit-eligible records 13，evidence training credit turns 22
- v1 shape candidate turns 37，v1 training credit turns 16
- early-answer penalty records 0

## Gained / Lost

相对 `turn_credit_evidence_bridge_20step`：

- gained：无
- lost：`test_2231`、`test_7511`、`test_8542`

相对 `turn_credit_evidence_bridge_5step`：

- gained：`dev_7773`
- lost：`test_2231`、`test_7511`、`test_8542`

相对 `prompt_budget_kl_std_5step`：

- gained：无
- lost：`test_2231`、`test_8542`

相对 `turn_credit_helpful_bridge_20step`：

- gained：`dev_174`、`dev_2429`、`dev_3741`、`dev_4869`
- lost：`test_2231`、`test_8542`

关键二跳样本仍保持正确：

- `dev_4869` 正确，2 次 search，答案 `Sextus Aelius Catus`。
- `dev_2429` 正确，2 次 search，答案 `42.5`。
- `dev_3741` 正确，2 次 search，答案 `Dziga Vertov`。
- `dev_174` 正确，2 次 search，答案 `Clifton College`。
- `dev_7773` 正确，3 次 search，答案 `Frankenstein'S Daughter`。

主要 lost case：

- `test_2231`：gold 为 `2013`，50-step 答 `1962`，属于实质错答。
- `test_7511`：gold 为 `Yoon Seok-Ho`，50-step 答 `尹锡湖`，属于中文名/英文名别名导致的 strict EM false negative。
- `test_8542`：gold 包含 `Plum/Plums`，50-step 答 `Damson plums`，属于更具体实体与 gold 粒度不一致导致的 strict EM false negative 或粒度问题。

## 结论

50-step 不优于 20-step。它保持了 format 1.0000、没有重新引入 `missing_followup_query`，也继续保护 `dev_4869/dev_2429/dev_3741/dev_174` 这类明确二跳样本；但 dev 70 EM 从 20-step 的 0.4429 回落到 0.4000，净 lost 3 条，且 helpful follow-up rate 从 0.4000 降到 0.3714。

回落原因不是工具失败、格式错误或 missing follow-up，而是长训练后策略向更短、更确定的答案漂移，并在严格 EM 上丢失别名/粒度样本和 1 条实质错答。当前最强 checkpoint 应继续保留为 `turn_credit_evidence_bridge_20step`，50-step 作为负向扩展证据保留，不建议继续扩大步数。

## 产物

- `my-search-r1/outputs/train_pytrio/turn-credit-evidence-bridge-50step-20260722-retry4/`
- `my-search-r1/eval_results/reward_train_compare_2026-07-21/turn_credit_evidence_bridge_50step_dev5_20260722.jsonl`
- `my-search-r1/eval_results/reward_train_compare_2026-07-21/turn_credit_evidence_bridge_50step_dev5_20260722.md`
- `my-search-r1/eval_results/reward_train_compare_2026-07-21/turn_credit_evidence_bridge_50step_dev_20260722.jsonl`
- `my-search-r1/eval_results/reward_train_compare_2026-07-21/turn_credit_evidence_bridge_50step_dev_20260722.md`
- `my-search-r1/eval_results/reward_train_compare_2026-07-21/turn_credit_evidence_bridge_50step_dev_20260722_offline_diagnostics.jsonl`
- `my-search-r1/eval_results/reward_train_compare_2026-07-21/turn_credit_evidence_bridge_50step_dev_20260722_offline_diagnostics.md`
- `my-search-r1/eval_results/reward_train_compare_2026-07-21/turn_credit_evidence_bridge_50step_turn_credit_analysis_20260722.jsonl`
- `my-search-r1/eval_results/reward_train_compare_2026-07-21/turn_credit_evidence_bridge_50step_turn_credit_analysis_20260722.md`
