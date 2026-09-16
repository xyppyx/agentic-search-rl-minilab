# Project Completed

本文件记录当前可引用的已完成事实、产物、结果和最终决策。压缩前快照位于 `docs/status/archive/2026-09-16_pre_next_experiments/`；更早快照仍在 `docs/status/archive/`，仅供历史追溯，不作为当前状态事实源。

## 当前结论快照

截至 2026-09-16，项目主线是 Search-R1 MiniLab：在 `Qwen/Qwen3.5-4B`、PyTRIO GRPO 和真实/可模拟搜索工具环境下，构建可观测、可诊断、可复盘的搜索型 Agentic RL 小预算实验框架。

当前最终路线固定为：

```text
turn-credit-final-hop-guardfix-20step-20260806
  -> guardfix20-resume-opsd-v2-5step-20260811
```

含义：先用 final-hop / bridge turn-level credit 学多跳搜索和停止策略，再从强 checkpoint 恢复，用 gated OPSD v2 做 5-step 保守 refinement。

关键边界：

- 本项目是个人学习型 POC，不代表 PyTRIO、SwanLab、知乎开放平台、Search-R1 官方实现或原作者参与、委托或认可。
- 早期“不可靠搜索工具/故障注入鲁棒训练”方向已搁置；failure injection 当前只用于 smoke、回归测试和评测边界验证。
- 当前结论不包装成大规模充分训练、完整 test 结论或 SOTA；正式效果优先使用 tool failures 为 0 的 clean run。
- Same-context OPSD 在 base 起点收益不足；OPSD v2 只在 guard-fix 强 checkpoint 上作为小系数 gated auxiliary objective 保留。
- 已验证过“更多训练步数不必然更好”：20-step OPSD v2 方差对照 bridge macro EM 略高，但 correct、format、search 综合弱于 5-step。

## 数据与训练量

公开数据 schema：

```json
{"id": "...", "question": "...", "answers": ["..."], "data_source": "..."}
```

当前数据与评测集：

| 文件 | 规模 | 用途 |
| --- | ---: | --- |
| `my-search-r1/datasets/train.jsonl` | 169,615 | GRPO/OPSD 训练采样池；HotpotQA 90,447，NQ 79,168 |
| `my-search-r1/datasets/dev.jsonl` | 70 | 每个 source 10 条的小预算 health gate |
| `my-search-r1/datasets/test.jsonl` | 51,713 | held-out pool 和 targeted eval 候选池 |
| `my-search-r1/datasets/bridge_eval_150.jsonl` | 150 | 多跳 bridge/final-hop targeted eval |
| `my-search-r1/datasets/bridge_eval_350.jsonl` | 350 | test-only bridge500 过滤 MuSiQue 后的较明确 bridge-hop targeted eval |

最终路线不是全量 epoch。实际训练取样量：

| 阶段 | `max_steps` | `questions_per_batch` | `group_size` | Question slot | Trajectory |
| --- | ---: | ---: | ---: | ---: | ---: |
| guard-fix 20-step | 20 | 2 | 4 | 40 | 160 |
| OPSD v2 5-step resume | 5 | 2 | 4 | 10 | 40 |

## 当前 Baseline

| 场景 | 模型/策略 | EM macro | Correct | Format | Avg search | 备注 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| dev70 | prompt-only best | 0.4143 | - | 0.8857 | - | `prompt_search_budget_guard` |
| dev70 | guard-fix 20-step retry | 0.4571 | 32/70 | 0.9571 | 1.9000 | OPSD 前强基座 |
| dev70 | final route | 0.4857 | 34/70 | 0.9857 | 1.7286 | 当前最高 clean dev70 |
| bridge150 | prompt-only base | 0.4750 | 74/150 | 0.7200 | 3.3067 | independent full run |
| bridge150 | guard-fix 20-step | 0.5142 | 83/150 | 0.8267 | 3.2000 | patched protocol |
| bridge150 | final route | 0.5242 | 87/150 | 0.9067 | 3.1400 | 10 个 clean chunks 合并，tool failures 0 |
| bridge_eval_350 | base+prompt | 0.4711 | 162/350 | 0.7457 | 2.9543 | micro EM 0.4629 |
| bridge_eval_350 | guard-fix 20-step | 0.4911 | 171/350 | 0.8429 | 2.8629 | micro EM 0.4886 |
| bridge_eval_350 | final route | 0.5200 | 184/350 | 0.9200 | 2.7200 | micro EM 0.5257 |
| alias80 | prompt-only base | 0.4500 | 36/80 | 0.9250 | 1.6500 | independent full run |
| alias80 | evidence-v2 20-step | 0.4375 | 35/80 | 0.9625 | 1.5125 | final route 尚未评测 |

结论：当前最强公开口径是 final route 在 dev70、bridge150 和 bridge_eval_350 上的 clean/同集对照。bridge_eval_350 显示 base -> GRPO -> GRPO+OPSD 的 correct、format、avg search 同时改善，但结论限定为较明确 bridge-hop 场景。

## 已验证能力

- 统一搜索工具层：`mock_search`、`local_bm25`、`zhihu_search`、backend registry、多 key 解析、错误脱敏、failure injection。
- 可观测轨迹：trajectory JSONL、Markdown report、工具失败/格式错误/重复搜索/行为 bucket/group comparison。
- Search-R1 训练与评测链路：PyTRIO train/eval CLI、group rollout、reference logprob、ratio clip、advantage standardization、KL-style drift penalty。
- Turn-level credit：evidence bridge、final-hop attribute search、early-answer、missing-final-hop、final-answer/max-search guard。
- Gated OPSD v2：`--opsd-coef`、`--opsd-mask-policy`、`--opsd-positive-policy`、teacher logprob 对齐、OPSD mask 指标和 custom loss。
- 长评测恢复：`eval_pytrio.py --offset` 支持分片恢复；分片仅用于 eval clean chunk 协议，不用于拼接训练。
- 面试/展示材料：已整理数据指标、turn-level credit、Gated OPSD、GRPO/OPSD 问答和简历项目表述；公开边界已统一到小预算 POC 口径。

## 当前公开边界

- 不提交真实 API key、远程 sampler weights URI、SwanLab 私有链接、模型权重、checkpoint、私有服务器地址或账号。
- `my-search-r1/datasets/`、`docs/interview/`、`docs/info/checkpoints.md`、`docs/resume` 等本地/个人材料由 `.gitignore` 保护。
- success rate < 1.0 的真实搜索 full run 不进入正式效果表；patched protocol 必须显式标注组成和边界。
- `dev70` 是 health gate，不是最终无偏 test；`bridge_eval_350` 是 test-only targeted subset，不代表完整 test 或所有复杂多跳问题。
