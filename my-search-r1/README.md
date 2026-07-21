# my-search-r1

本目录用于实现 Robust Search-R1 MiniLab，是当前项目区别于上游教学复现的主要工作区。

近期目标：

```text
让 Search-R1 smoke/5-step run 产生可复查的 trajectory JSONL 和报告。
```

计划模块：

- `search_r1_minilab/tools/`：统一搜索工具接口、backend registry、Zhihu backend、mock backend、local BM25 和 failure wrapper。
- `trajectories/`：trajectory schema、JSONL 序列化和报告生成。
- `rewards/`：正确性、格式、工具调用与重复 query 等 reward/penalty 组件。
- `backend/`：PyTRIO 训练后端薄封装。
- `configs/`：可公开的 smoke/eval/train 配置模板。
- `scripts/`：数据准备、rollout、训练、评测和报告命令入口。

实现前以 [../03-search-r1/](../03-search-r1/) 为基线参考，以 [../docs/design/idea.md](../docs/design/idea.md) 为路线依据。reward shaping 版本和下一轮实验计划维护在 [../docs/design/reward_shaping_plan.md](../docs/design/reward_shaping_plan.md)。真实凭据写入本地 `.env`，公开模板写入 `.env.example`。

## 当前已实现

首版搜索工具层已经具备：

- `MockSearchBackend`：固定 fixture 查询，未知 query 返回空结果。
- `ZhihuSearchBackend`：适配现有知乎全局搜索 API，支持多 key 解析、轮转、有限重试和错误脱敏。
- `LocalBM25Backend`：读取本地 JSONL 小语料，使用轻量 BM25 做离线可复现检索。
- `FailureWrapperBackend`：用固定 seed 对任意 backend 注入 timeout、empty_result、noisy_result、rate_limited。
- `ToolRegistry`：按 `tool_name` 分发调用，使 rollout 后续不直接依赖具体搜索 client。
- `search_r1_minilab/trajectories/`：统一 trajectory JSONL 读写、分类统计和 Markdown 报告。
- `search_r1_minilab/protocol.py`：Search-R1 chat template、工具调用解析和 tool message 构造。
- `search_r1_minilab/rewards.py`：`Answer:` 格式校验与 exact-match reward。
- `search_r1_minilab/rollout_smoke.py`：PyTRIO sampler + ToolRegistry 的最小 rollout/eval 状态机。
- `search_r1_minilab/rollout.py`：训练级 group rollout，保存 old logprob、advantage 和可报告的 tool 事件。
- `search_r1_minilab/training.py`：PyTRIO GRPO Datum 构建、micro-batch 装箱、advantage 权重缩放和训练指标。
- `search_r1_minilab/tooling.py`：训练、评测和 smoke 共用的 backend 构造，支持 failure injection。
- `search_r1_minilab/prepare_data.py`：固定 ModelScope 数据版本的 train/test/dev JSONL 准备逻辑。
- `search_r1_minilab/analysis.py`：checkpoint eval JSONL 指标读取和 EM/format 曲线绘图。
- `scripts/prepare_data.py`：下载并清洗 Search-R1 训练、测试和固定 dev 数据。
- `scripts/tool_smoke.py`：无需模型或真实 API key，使用 mock/local BM25 生成 trajectory JSONL 和报告。
- `scripts/rollout_smoke_eval.py`：使用 PyTRIO 真实模型采样，默认通过 local BM25 生成 trajectory JSONL 和 Markdown 报告。
- `scripts/eval_pytrio.py`：base/checkpoint 共用的训练级 rollout 评测入口，输出统一 JSONL 与 Markdown 报告。
- `scripts/train_pytrio.py`：PyTRIO GRPO 训练入口，默认 local BM25 和公开 fixture，可跑 1-step smoke。
- `scripts/analyse_trajectories.py`：从已有 trajectory JSONL 生成 Markdown 报告。
- `scripts/analyse_checkpoints.py`：从 eval JSONL 生成 checkpoint EM/format 对比图，兼容 summary JSONL 和纯 trajectory JSONL。
- `scripts/analyse_offline_diagnostics.py`：从 eval JSONL 离线标注 alias、答案粒度和 missing follow-up query 风险。
- `scripts/analyse_reward_sensitivity.py`：从 eval JSONL 离线重评分，对比不同 reward penalty 配置的分布影响和误伤风险。

运行工具层测试：

```bash
PYTHONPATH=my-search-r1 uv run python -m unittest discover -s my-search-r1/tests -v
```

运行工具 smoke 并生成报告：

```bash
PYTHONPATH=my-search-r1 uv run python my-search-r1/scripts/tool_smoke.py \
  --backend mock_search \
  --jsonl-output /tmp/search-r1-tool-smoke.jsonl \
  --report-output /tmp/search-r1-tool-smoke.md
```

运行 PyTRIO rollout smoke/eval 并生成报告：

```bash
PYTHONPATH=my-search-r1 uv run python my-search-r1/scripts/rollout_smoke_eval.py
```

默认使用 `Qwen/Qwen3.5-4B`、2 条公开 smoke fixture、`local_bm25` 和
`my-search-r1/tests/fixtures/bm25_corpus.jsonl`。真实 PyTRIO 凭据写在本地
`my-search-r1/.env` 的 `PYTRIO_API_KEY`，输出默认落到被 git 忽略的
`my-search-r1/outputs/rollout_smoke/`。

从已有 JSONL 重新生成报告：

```bash
PYTHONPATH=my-search-r1 uv run python my-search-r1/scripts/analyse_trajectories.py \
  --input /tmp/search-r1-tool-smoke.jsonl \
  --output /tmp/search-r1-tool-smoke-report.md
```

运行训练级 PyTRIO eval：

```bash
PYTHONPATH=my-search-r1 uv run python my-search-r1/scripts/eval_pytrio.py \
  --backend local_bm25 \
  --limit 2 \
  --batch-size 1
```

运行 1-step PyTRIO GRPO train smoke：

```bash
PYTHONPATH=my-search-r1 uv run python my-search-r1/scripts/train_pytrio.py \
  --max-steps 1 \
  --questions-per-batch 1 \
  --group-size 2 \
  --backend local_bm25 \
  --swanlab-mode disabled \
  --run-name search-r1-minilab-smoke
```

训练和评测默认使用 `local_bm25`，可通过 `--backend mock_search` 或
`--backend zhihu_search --env-file my-search-r1/.env` 切换。failure injection
通过 `--p-timeout`、`--p-empty`、`--p-noise`、`--p-rate-limited` 和
`--failure-seed` 控制。

准备完整 Search-R1 数据：

```bash
PYTHONPATH=my-search-r1 uv run python my-search-r1/scripts/prepare_data.py
```

默认写入被 git 忽略的 `my-search-r1/datasets/`，包含 `train.jsonl`、
`test.jsonl` 和按来源均衡抽样的 `dev.jsonl`。

生成 checkpoint EM/format 对比图：

```bash
PYTHONPATH=my-search-r1 uv run python my-search-r1/scripts/analyse_checkpoints.py \
  --result-dir my-search-r1/eval_results \
  --output my-search-r1/eval_results/checkpoint_em_format.png
```

默认会优先使用原 Search-R1 checkpoint 文件名；如果目录里只有
`trajectories.jsonl`，会自动按 `Current=trajectories.jsonl` 生成单点图。
也可以重复传入 `--checkpoint 'Step 20=eval_results_rl_step_20.jsonl'` 指定列表。

运行离线诊断：

```bash
PYTHONPATH=my-search-r1 uv run python my-search-r1/scripts/analyse_offline_diagnostics.py \
  --input my-search-r1/eval_results/reward_train_compare_2026-07-21/base_20step_dev.jsonl \
  --jsonl-output my-search-r1/eval_results/reward_train_compare_2026-07-21/base_20step_offline_diagnostics.jsonl \
  --report-output my-search-r1/eval_results/reward_train_compare_2026-07-21/base_20step_offline_diagnostics.md
```

运行 reward 敏感性分析：

```bash
PYTHONPATH=my-search-r1 uv run python my-search-r1/scripts/analyse_reward_sensitivity.py \
  --input my-search-r1/eval_results/reward_train_compare_2026-07-21/base_20step_dev.jsonl \
  --summary-output my-search-r1/eval_results/reward_train_compare_2026-07-21/base_20step_reward_sensitivity_summary.json \
  --jsonl-output my-search-r1/eval_results/reward_train_compare_2026-07-21/base_20step_reward_sensitivity.jsonl \
  --report-output my-search-r1/eval_results/reward_train_compare_2026-07-21/base_20step_reward_sensitivity.md
```

默认对比 `base_reward_v0`、`penalty_v1`、`penalty_v2_candidate` 和
`penalty_v2_no_empty`。可重复传入 `--config` 做小规模自定义配置：

```bash
PYTHONPATH=my-search-r1 uv run python my-search-r1/scripts/analyse_reward_sensitivity.py \
  --input my-search-r1/eval_results/reward_train_compare_2026-07-21/base_20step_dev.jsonl \
  --config 'dup_only:duplicate=0.02,empty=0,max_search=0,verbose=0,verbose_threshold=0' \
  --summary-output /tmp/reward_sensitivity_summary.json \
  --jsonl-output /tmp/reward_sensitivity.jsonl \
  --report-output /tmp/reward_sensitivity.md
```
