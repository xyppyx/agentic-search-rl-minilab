# Agentic Search RL MiniLab

面向搜索型 LLM Agent 的训练、评测与轨迹诊断实验框架。

本项目基于原作者 [`KMnO4-zx/llm-agent-rl-lab`](https://github.com/KMnO4-zx/llm-agent-rl-lab) 的教学复现仓库做个人二次开发，当前主线是把“模型主动搜索 - 读取 observation - 继续 follow-up - 输出短答案”的链路，改造成可观测、可诊断、可复盘的小型 Agentic RL 研究框架。

截至 2026-08-11，最终实验路线固定为：

```text
guard-fix 20-step final-hop turn credit
  -> OPSD v2 5-step gated conservative refinement
```

也就是先用 turn-level reward shaping 学 bridge/final-hop 搜索策略，再从该 checkpoint 恢复，用 `credited_turns + positive_advantage` 的 gated OPSD v2 做小步保守微调。最终 checkpoint 为 `guardfix20-resume-opsd-v2-5step-20260811`。

核心实现位于 [my-search-r1/](my-search-r1/)。原 `00-loss-function/`、`01-grpo/`、`02-opd/`、`03-search-r1/` 四个教学目录已归档到 `Backup` 分支，`main` 只保留当前项目主线。

## 当前进展

已经完成的核心能力：

- 统一搜索工具层：支持 `mock_search`、`local_bm25`、`zhihu_search` 和 failure injection。
- 训练与评测轨迹：保存完整 trajectory JSONL，并生成 Markdown report。
- PyTRIO GRPO 链路：支持 train/eval CLI、group rollout、reference logprob、ratio clip、advantage standardization 和 KL-style drift penalty。
- Gated OPSD v2：支持 `--opsd-coef`、`--opsd-mask-policy`、`--opsd-positive-policy` 和 teacher logprob 对齐，用作 GRPO 之外的小系数辅助目标。
- 诊断与复盘：支持 offline diagnostics、reward sensitivity、turn-credit analysis、checkpoint 对比和 gained/lost case review。
- 定向评测集：构建 `dev70`、`bridge_eval_150`、`alias_granularity_eval_80`，分开记录模型策略问题和真实搜索工具失败。

## 方法迭代

项目的主要结论不是“惩罚搜索越多越好”，而是：搜索型 Agent 的 reward shaping 必须先区分错误类型，再给定向信号。

已验证过的迭代路线：

1. 从 base reward 开始，只检查 `Answer:` 格式和 exact match。
2. 尝试 duplicate query、empty result、max-search no-answer 等轻量 penalty。
3. 发现简单 penalty 能减少无效搜索，但可能压掉必要 follow-up，导致 EM 下降。
4. 加入 prompt/rollout 层的多跳 follow-up 和短答案约束。
5. 引入 turn-level credit，奖励 evidence bridge search 和 final-hop attribute search。
6. 加入 early-answer、missing-final-hop、final-answer/max-search guard，定位 format 和停止策略问题。
7. 在 guard-fix 20-step 强 checkpoint 上加入 gated OPSD v2，只蒸馏 positive advantage 且被 turn-credit 命中的 assistant action tokens。

最终结论：OPSD 不能做 naive full-sequence self-distillation。搜索型 Agent 的轨迹包含工具 observation、错误 query、wrong-valid final answer 和 early-answer 行为，必须用 gate/mask 约束蒸馏范围。

## 结果快照

截至 2026-08-11，公开状态文件中记录的核心对比如下：

| 场景                |                                                                  Prompt/Base |                                                                         最终路线 |                                    变化 |
| ------------------- | ---------------------------------------------------------------------------: | -------------------------------------------------------------------------------: | --------------------------------------: |
| `dev70`           |                                    prompt-only best EM 0.4143，format 0.8857 |  OPSD v2 5-step clean EM 0.4857，correct 34/70，format 0.9857，avg search 1.7286 |              EM +0.0714，format +0.1000 |
| `bridge_eval_150` | prompt-only base EM 0.4750，correct 74/150，format 0.7200，avg search 3.3067 | OPSD v2 5-step clean EM 0.5242，correct 87/150，format 0.9067，avg search 3.1400 | EM +0.0492，correct +13，format +0.1867 |

路线实验对比：

| 路线                                |                                         dev70 |                                             bridge_eval_150 | 结论                                                                 |
| ----------------------------------- | --------------------------------------------: | ----------------------------------------------------------: | -------------------------------------------------------------------- |
| prompt-only guard                   |                      EM 0.4143，format 0.8857 |                    EM 0.4750，correct 74/150，format 0.7200 | 强 prompt baseline，但策略学习有限                                   |
| behavior penalty                    |                             部分减少重复/空搜 |                                              未作为最终候选 | 简单 penalty 容易压掉必要 follow-up                                  |
| evidence-v2 20-step                 |   EM 0.4429，format 1.0000，avg search 1.7714 | EM 0.4583，correct 81/150，format 0.9400，avg search 3.0933 | format 最强对照，但 EM/correct 不最高                                |
| guard-fix 20-step                   |       EM 0.4571，correct 32/70，format 0.9571 |            patched EM 0.5142，correct 83/150，format 0.8267 | final-hop credit 提升 EM/correct；bridge 结果需标注 patched protocol |
| guardfix20 + OPSD v2 5-step         | clean EM 0.4857，correct 34/70，format 0.9857 |              clean EM 0.5242，correct 87/150，format 0.9067 | 当前最终路线，clean correct/format/search 综合最好                   |
| guardfix20 + OPSD v2 20-step seed43 | clean EM 0.4571，correct 32/70，format 1.0000 |              clean EM 0.5317，correct 81/150，format 0.8133 | bridge EM macro 略高，但 correct/format/search 综合弱于 5-step       |

重要边界：

- `bridge_eval_150` 的最终路线结果来自 10 个 clean chunks 合并，150 条 trajectory 的 tool failures 为 0。
- `bridge_eval_150` 的 guard-fix 20-step 历史结果按 patched protocol 展示，其中 3 条由工具错误样本 retry 补齐；该口径适合项目分析，但不等同论文式独立全量 success rate 1.0 run。
- 真实 Zhihu Search API 会出现限流、url error、parse error 等外部失败；本项目要求把 tool failure 和模型策略失败分开记录。

完整事实源见 [docs/status/PROJECT_COMPLETED.md](docs/status/PROJECT_COMPLETED.md) 和 [docs/status/PROJECT_TODO.md](docs/status/PROJECT_TODO.md)。

## 快速开始

依赖使用 `uv` 管理：

```bash
uv sync
```

运行离线测试：

```bash
PYTHONPATH=my-search-r1 uv run python -m unittest discover -s my-search-r1/tests -v
```

运行 mock/local BM25 工具 smoke：

```bash
PYTHONPATH=my-search-r1 uv run python my-search-r1/scripts/tool_smoke.py \
  --backend mock_search \
  --jsonl-output /tmp/search-r1-tool-smoke.jsonl \
  --report-output /tmp/search-r1-tool-smoke.md
```

运行训练级 eval smoke：

```bash
PYTHONPATH=my-search-r1 uv run python my-search-r1/scripts/eval_pytrio.py \
  --backend local_bm25 \
  --limit 2 \
  --batch-size 1
```

运行 1-step GRPO smoke：

```bash
PYTHONPATH=my-search-r1 uv run python my-search-r1/scripts/train_pytrio.py \
  --max-steps 1 \
  --questions-per-batch 1 \
  --group-size 2 \
  --backend local_bm25 \
  --swanlab-mode disabled \
  --run-name search-r1-minilab-smoke
```

真实 PyTRIO、Zhihu Search 和 SwanLab 运行需要本地 `.env`；真实密钥、远程 sampler weights URI、SwanLab 私有链接、模型权重和 checkpoint 不提交。

## 仓库结构

```text
my-search-r1/
  search_r1_minilab/      # 工具层、rollout、reward、training、diagnostics
  scripts/                # 数据、训练、评测、分析入口
  tests/                  # unittest 与 fixture
docs/
  design/                 # 设计草案和 reward shaping 计划
  status/                 # 项目事实源：完成项、TODO、长期日志
```

## 文档入口

- [my-search-r1/README.md](my-search-r1/README.md)：实现细节与脚本用法。
- [docs/design/reward_shaping_plan.md](docs/design/reward_shaping_plan.md)：reward shaping 版本和实验决策。

## 项目边界

本项目是个人学习型 POC，不代表 PyTRIO、SwanLab、知乎开放平台、Search-R1 官方实现或原作者参与、委托或认可。所有实验指标均按公开状态文件中已验证记录表述；未通过工具成功率门槛的 run 不包装成正式模型效果结论。

## 上游来源

- 原教学项目：[https://github.com/KMnO4-zx/llm-agent-rl-lab](https://github.com/KMnO4-zx/llm-agent-rl-lab)
- Search-R1 论文：[https://arxiv.org/abs/2503.09516](https://arxiv.org/abs/2503.09516)
- Search-R1 官方实现：[https://github.com/PeterGriffinJin/Search-R1](https://github.com/PeterGriffinJin/Search-R1)

## License

See [LICENSE](LICENSE).
