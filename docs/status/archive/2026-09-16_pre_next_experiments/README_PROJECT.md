# Agentic Search RL MiniLab

面向搜索型 LLM Agent 的训练、评测与轨迹诊断实验框架。

本项目基于原作者 [`KMnO4-zx/llm-agent-rl-lab`](https://github.com/KMnO4-zx/llm-agent-rl-lab) 的教学复现仓库做个人二次开发，当前主线是把“模型主动搜索 - 读取 observation - 继续 follow-up - 输出短答案”的链路，改造成可观测、可诊断、可复盘的小型 Agentic RL 研究框架。早期关于“不可靠搜索工具/故障注入鲁棒训练”的设想已暂时搁置；当前正式路线聚焦多跳搜索策略、turn-level credit、gated OPSD 和 clean 评测口径，工具失败记录只用于隔离外部环境污染。训练使用的基础模型为 `Qwen/Qwen3.5-4B`。

截至 2026-08-13，最终实验路线固定为：

```text
guard-fix 20-step final-hop turn credit
  -> OPSD v2 5-step gated conservative refinement
```

也就是先用 turn-level reward shaping 学 bridge/final-hop 搜索策略，再从该 checkpoint 恢复，用 `credited_turns + positive_advantage` 的 gated OPSD v2 做小步保守微调。最终 checkpoint 为 `guardfix20-resume-opsd-v2-5step-20260811`。

核心实现位于 [my-search-r1/](my-search-r1/)。原 `00-loss-function/`、`01-grpo/`、`02-opd/`、`03-search-r1/` 四个教学目录已归档到 `Backup` 分支，`main` 只保留当前项目主线。

## 当前进展

已经完成的核心能力：

- 统一搜索工具层：支持 `mock_search`、`local_bm25`、`zhihu_search` 和用于 smoke/回归的 failure injection。
- 训练与评测轨迹：保存完整 trajectory JSONL，并生成 Markdown report。
- PyTRIO GRPO 链路：支持 train/eval CLI、group rollout、reference logprob、ratio clip、advantage standardization 和 KL-style drift penalty。
- Gated OPSD v2：支持 `--opsd-coef`、`--opsd-mask-policy`、`--opsd-positive-policy` 和 teacher logprob 对齐，用作 GRPO 之外的小系数辅助目标。
- 诊断与复盘：支持 offline diagnostics、reward sensitivity、turn-credit analysis、checkpoint 对比和 gained/lost case review。
- 数据与评测集：准备 `train.jsonl`、`dev.jsonl`、`test.jsonl`、`bridge_eval_150.jsonl` 和扩展的 `bridge_eval_350.jsonl`，分开记录模型策略问题和真实搜索工具失败；正式效果表只采用 clean 或显式标注的 patched 口径。

## 数据与评测集

数据采用 Search-R1 风格 JSONL，每条样本包含：

```json
{"id": "...", "question": "...", "answers": ["..."], "data_source": "..."}
```

本地数据目录 `my-search-r1/datasets/` 被 `.gitignore` 忽略，公开仓库不提交完整训练/评测数据文件。数据准备入口：

```bash
PYTHONPATH=my-search-r1 uv run python my-search-r1/scripts/prepare_data.py
```

当前使用的数据与评测集：

| 文件 | 规模 | 用途 |
| --- | ---: | --- |
| `my-search-r1/datasets/train.jsonl` | 169,615 | PyTRIO GRPO/OPSD 训练采样池 |
| `my-search-r1/datasets/dev.jsonl` | 70 | 小预算 checkpoint 选择、health/eval 和路线快速对照 |
| `my-search-r1/datasets/test.jsonl` | 51,713 | 保留评测集与 targeted eval 候选池 |
| `my-search-r1/datasets/bridge_eval_150.jsonl` | 150 | 多跳 bridge/final-hop 搜索能力 targeted eval |
| `my-search-r1/datasets/bridge_eval_350.jsonl` | 350 | 多跳 targeted eval，用于验证较明确多跳场景下的搜索策略收益 |

`dev.jsonl` 按 data source 均衡抽样，每类 10 条，共 70 条；`bridge_eval_150.jsonl` 由 `my-search-r1/scripts/build_targeted_eval_sets.py` 从 dev/test/train 候选池中按 multi-hop source 和 bridge cue 规则筛出，重点测试多跳搜索、role binding 和 follow-up search。`bridge_eval_350.jsonl` 是扩展多跳 targeted eval，用于更稳定地展示训练后的搜索策略收益。

评测报告统一记录 EM macro、format、平均搜索次数、tool failures、tool success rate 和失败类型。这里记录工具失败是为了排除外部 API 波动对模型效果的污染，不表示当前主线在训练模型适应故障工具。真实搜索 full run 的 tool success rate 低于 1.0 时，不进入正式结果表；patched protocol 会单独标注。

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

截至 2026-08-13，公开状态文件中记录的核心对比如下：

| 场景                |                                                                  Prompt/Base |                                                                         最终路线 |                                    变化 |
| ------------------- | ---------------------------------------------------------------------------: | -------------------------------------------------------------------------------: | --------------------------------------: |
| `dev70`           |                                    prompt-only best EM 0.4143，format 0.8857 |               OPSD v2 5-step clean EM 0.4857，format 0.9857，avg search 1.7286 |              EM +0.0714，format +0.1000 |
| `bridge_eval_150` |                   prompt-only base EM 0.4750，format 0.7200，avg search 3.3067 |              OPSD v2 5-step clean EM 0.5242，format 0.9067，avg search 3.1400 |              EM +0.0492，format +0.1867 |
| `bridge_eval_350` |                   base+prompt EM 0.4711，format 0.7457，avg search 2.9543 |              GRPO+OPSD v2 EM 0.5200，format 0.9200，avg search 2.7200 | EM +0.0489，format +0.1743 |

EM 口径：结果快照中的 EM 统一使用 macro 口径。`bridge_eval_350` 的逐样本总体比例可通过下方 `Correct` 列换算，三方明细表也同时列出 Micro EM 便于复核。

`bridge_eval_350` 三方对照更清楚地展示了训练收益链路：

| 路线 | Micro EM | Macro EM | Correct | Format | Avg search | 结论 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| base+prompt | 0.4629 | 0.4711 | 162/350 | 0.7457 | 2.9543 | 强 prompt 基线 |
| GRPO guard-fix 20-step | 0.4886 | 0.4911 | 171/350 | 0.8429 | 2.8629 | turn-level credit 带来中间增益 |
| GRPO + OPSD v2 5-step | 0.5257 | 0.5200 | 184/350 | 0.9200 | 2.7200 | 最终路线，EM/format/search 综合最好 |

这组结果说明：在更明确的多跳搜索场景中，训练不是只改善输出格式；从 base 到 GRPO，再到 GRPO+OPSD v2，correct、format 和搜索效率都呈递进改善。更复杂多跳问题仍需要继续做 case review 和边界分析。

路线实验对比：

| 路线                                |                                         dev70 |                                             bridge_eval_150 | 结论                                                                 |
| ----------------------------------- | --------------------------------------------: | ----------------------------------------------------------: | -------------------------------------------------------------------- |
| prompt-only guard                   |                      EM 0.4143，format 0.8857 |                                  EM 0.4750，format 0.7200 | 强 prompt baseline，但策略学习有限                                   |
| behavior penalty                    |                             部分减少重复/空搜 |                                              未作为最终候选 | 简单 penalty 容易压掉必要 follow-up                                  |
| evidence-v2 20-step                 |   EM 0.4429，format 1.0000，avg search 1.7714 |              EM 0.4583，format 0.9400，avg search 3.0933 | format 最强对照，但 EM 不最高                                       |
| guard-fix 20-step                   |                  EM 0.4571，format 0.9571 |                         patched EM 0.5142，format 0.8267 | final-hop credit 提升 EM；bridge 结果需标注 patched protocol         |
| guardfix20 + OPSD v2 5-step         |            clean EM 0.4857，format 0.9857 |                           clean EM 0.5242，format 0.9067 | 当前最终路线，clean EM/format/search 综合最好                        |
| guardfix20 + OPSD v2 20-step seed43 |            clean EM 0.4571，format 1.0000 |                           clean EM 0.5317，format 0.8133 | bridge EM macro 略高，但 format/search 综合弱于 5-step               |

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
- [docs/design/idea.md](docs/design/idea.md)：公开版系统设计概览。
- [docs/design/reward_shaping_plan.md](docs/design/reward_shaping_plan.md)：公开版 reward 与辅助目标设计原则。
- [docs/design/evaluation_design.md](docs/design/evaluation_design.md)：公开版 dev70 与 bridge150 评测设计。

## 项目边界

本项目是个人学习型 POC，不代表 PyTRIO、SwanLab、知乎开放平台、Search-R1 官方实现或原作者参与、委托或认可。所有实验指标均按公开状态文件中已验证记录表述；未通过工具成功率门槛的 run 不包装成正式模型效果结论。

## 上游来源

- 原教学项目：[https://github.com/KMnO4-zx/llm-agent-rl-lab](https://github.com/KMnO4-zx/llm-agent-rl-lab)
- Search-R1 论文：[https://arxiv.org/abs/2503.09516](https://arxiv.org/abs/2503.09516)
- Search-R1 官方实现：[https://github.com/PeterGriffinJin/Search-R1](https://github.com/PeterGriffinJin/Search-R1)

## License

See [LICENSE](LICENSE).
