# Bridge Targeted 三模型评测复盘

记录时间：2026-07-23 09:55 CST

## 目标

在新构造的 `bridge_eval_150.jsonl` 上补齐 targeted bridge 对比：

- prompt-only base
- `turn_credit_evidence_bridge_20step`
- `turn_credit_evidence_bridge_50step`

核心指标为 EM、format、平均搜索次数、`missing_followup_query` 和逐样本 gained/lost。公开复盘不记录 PyTRIO 远端 sampler weights URI。

## 数据与配置

- 数据：`my-search-r1/datasets/bridge_eval_150.jsonl`
- 样本数：150
- backend：`zhihu_search`
- seed：42
- batch size：1
- base model：`Qwen/Qwen3.5-4B`
- 输出目录：`my-search-r1/eval_results/targeted_eval_20260723/`

本轮使用 PyTRIO checkpoint list 在本地定位 20-step 与 50-step 的 final sampler weights；URI 未写入公开文档。

## 运行产物

- prompt-only base：`my-search-r1/eval_results/targeted_eval_20260723/bridge_prompt_base_20260723.jsonl`
- 20-step：`my-search-r1/eval_results/targeted_eval_20260723/bridge_turn_credit_evidence_20step_20260723.jsonl`
- 50-step：`my-search-r1/eval_results/targeted_eval_20260723/bridge_turn_credit_evidence_50step_20260723.jsonl`
- 三组对比：`my-search-r1/eval_results/targeted_eval_20260723/bridge_targeted_three_way_comparison_20260723.md`
- 20-step offline diagnostics：`my-search-r1/eval_results/targeted_eval_20260723/bridge_turn_credit_evidence_20step_20260723_offline_diagnostics.md`
- 50-step offline diagnostics：`my-search-r1/eval_results/targeted_eval_20260723/bridge_turn_credit_evidence_50step_20260723_offline_diagnostics.md`

## 指标

这里同时记录两种 EM：

- EM macro：按数据源分别求 EM 后宏平均，和 `eval_pytrio.py` 终端输出的 `em/macro` 一致。
- EM overall：150 条样本整体 exact match 比例。

| Run | 工具门槛有效 | EM macro | EM overall | Correct | Format | Avg search | Zhihu success | Missing follow-up | Alias risk | Bad max-search loop |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| prompt-only base | yes | 0.4750 | 0.4933 | 74/150 | 0.7200 | 3.3067 | 1.0000 | 3 | 3 | 9 |
| evidence 20-step | yes | 0.4583 | 0.5400 | 81/150 | 0.9400 | 3.0933 | 1.0000 | 5 | 7 | 6 |
| evidence 50-step | no | 0.4733 | 0.5200 | 78/150 | 0.8400 | 3.1733 | 0.9979 | 6 | 5 | 11 |

## Gained / Lost

20-step 相对 prompt-only base：

- gained：12
- lost：5
- net：+7
- gained ids：`dev_10628`、`dev_7767`、`dev_8262`、`dev_8597`、`dev_10322`、`dev_11245`、`dev_11513`、`dev_11777`、`dev_11860`、`dev_12533`、`dev_7495`、`dev_7595`
- lost ids：`train_80307`、`dev_7742`、`dev_2331`、`test_99`、`dev_11691`

50-step 相对 prompt-only base：

- gained：7
- lost：3
- net：+4
- gained ids：`dev_7767`、`dev_7961`、`dev_8262`、`dev_10322`、`dev_11777`、`dev_7495`、`dev_7725`
- lost ids：`train_80307`、`dev_9212`、`dev_2331`

50-step 相对 20-step：

- gained：5
- lost：8
- net：-3
- gained ids：`dev_7742`、`dev_7961`、`test_99`、`dev_11691`、`dev_7725`
- lost ids：`dev_10628`、`dev_8597`、`dev_9212`、`dev_11245`、`dev_11513`、`dev_11860`、`dev_12533`、`dev_7595`

## 工具失败

20-step 工具指标达标：

- Zhihu requests：464
- success rate：1.0000
- error/timeout/rate-limit：0

50-step 未通过工具门槛：

- Zhihu requests：476
- success rate：0.9979
- error rate：0.0021
- 失败样本：`dev_9212`
- 失败 query：`Hills Of Kentucky film director`
- error type：`parse_error: TypeError`

因此 50-step 本轮只作为参考结果，不作为正式 gated comparison 结论。

## 结论

有效结果中，20-step 相对 prompt-only base 明显提高 format 和 overall correct：format 从 0.7200 到 0.9400，正确数从 74/150 到 81/150，平均搜索从 3.3067 降到 3.0933。但 EM macro 从 0.4750 降到 0.4583，说明提升不均匀，可能集中在占比更高的 2WikiMultihopQA，而 HotpotQA/MuSiQue 仍弱。

50-step 在本轮有工具失败，且即使按参考结果看，也比 20-step 少 3 个 overall correct、format 更低、bad max-search loop 更多。因此当前 bridge targeted eval 不支持把 50-step 作为更强 checkpoint。

## 下一步

- 若要完成严格三模型 bridge 对比，应在 Zhihu 工具健康时重跑 50-step bridge eval，要求 success rate 1.0。
- 若不重跑 50-step，可先接受 20-step 是当前 bridge targeted 的有效最强 checkpoint，并转入 `alias_granularity_eval_80.jsonl` 的 prompt/20-step/50-step 评测。
- 后续分析重点应拆分 source-level EM，尤其看 HotpotQA 与 MuSiQue 的 lost case，而不是只看 overall correct。
