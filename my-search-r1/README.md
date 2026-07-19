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

实现前以 [../03-search-r1/](../03-search-r1/) 为基线参考，以 [../docs/design/idea.md](../docs/design/idea.md) 为路线依据。真实凭据写入本地 `.env`，公开模板写入 `.env.example`。

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
- `scripts/tool_smoke.py`：无需模型或真实 API key，使用 mock/local BM25 生成 trajectory JSONL 和报告。
- `scripts/rollout_smoke_eval.py`：使用 PyTRIO 真实模型采样，默认通过 local BM25 生成 trajectory JSONL 和 Markdown 报告。
- `scripts/analyse_trajectories.py`：从已有 trajectory JSONL 生成 Markdown 报告。

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
