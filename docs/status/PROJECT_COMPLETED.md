# Project Completed

本文件记录已完成且有验证证据的事实、产物、结果和最终决策。未验证的代码或实验不得写入完成项。

## 2026-07-19

- 项目定位已明确为基于上游 `llm-agent-rl-lab` 的 Search-R1/Agentic RL 学习与改进项目；当前主线是 `docs/design/idea.md` 中的 Robust Search-R1 MiniLab。
  - 相关路径：`AGENTS.md`、`README.md`、`docs/status/`、`my-search-r1/`
  - 验证方式：读取仓库结构与 `docs/design/idea.md`；检查所有一级目录均具备 `AGENTS.md` 和 `README.md`；运行 `git diff --check` 无空白错误。
  - 关键结果：协作规范已从旧医疗后训练项目语境迁移到 Search-R1 MiniLab 项目语境，状态事实源和 `my-search-r1/` 目录规则已建立。

- `my-search-r1/` 首版搜索工具层已实现。
  - 相关路径：`my-search-r1/search_r1_minilab/tools/`、`my-search-r1/tests/test_tools.py`、`my-search-r1/tests/fixtures/bm25_corpus.jsonl`
  - 验证方式：运行 `PYTHONPATH=my-search-r1 uv run python -m unittest discover -s my-search-r1/tests -v`。
  - 关键结果：12 个 unittest 全部通过；已覆盖 mock fixture/空结果、Zhihu key 解析和错误脱敏、local BM25 排序/空 query/无命中、failure wrapper 固定 seed 与失败类型指标、registry 分发，以及 mock/local BM25 smoke record schema 一致性。

- `my-search-r1/` trajectory JSONL 保存与 Markdown 报告已实现。
  - 相关路径：`my-search-r1/search_r1_minilab/trajectories/`、`my-search-r1/scripts/analyse_trajectories.py`、`my-search-r1/scripts/tool_smoke.py`、`my-search-r1/tests/test_trajectories.py`
  - 验证方式：运行 `PYTHONPATH=my-search-r1 uv run python -m unittest discover -s my-search-r1/tests -v`；运行 `tool_smoke.py` 分别验证 `mock_search` 和 `local_bm25` 输出 JSONL 与 Markdown；运行 `analyse_trajectories.py` 从已有 JSONL 重新生成 Markdown。
  - 关键结果：16 个 unittest 全部通过；`/tmp/search-r1-tool-smoke.jsonl`、`/tmp/search-r1-tool-smoke.md`、`/tmp/search-r1-bm25-smoke.jsonl`、`/tmp/search-r1-bm25-smoke.md` 和 `/tmp/search-r1-tool-smoke-reanalyse.md` 均成功生成。报告支持 summary、correct、wrong、invalid_format、tool_failure、repeated_search 和 group comparison 区块。

- `my-search-r1/` 最小 PyTRIO rollout smoke/eval 链路已跑通，并能生成 Markdown 报告。
  - 相关路径：`my-search-r1/search_r1_minilab/protocol.py`、`my-search-r1/search_r1_minilab/rewards.py`、`my-search-r1/search_r1_minilab/rollout_smoke.py`、`my-search-r1/scripts/rollout_smoke_eval.py`、`my-search-r1/tests/fixtures/smoke_eval.jsonl`、`my-search-r1/tests/test_rollout_smoke.py`
  - 验证方式：运行 `PYTHONPATH=my-search-r1 uv run python -m unittest discover -s my-search-r1/tests -v`；运行 `PYTHONPATH=my-search-r1 uv run python my-search-r1/scripts/rollout_smoke_eval.py`；检查 `my-search-r1/outputs/rollout_smoke/trajectories.jsonl` 和 `my-search-r1/outputs/rollout_smoke/report.md`。
  - 关键结果：19 个 unittest 全部通过；真实 PyTRIO smoke 使用默认 `Qwen/Qwen3.5-4B`、2 条公开 MiniLab fixture 和 `local_bm25` 完成，生成 2 条 trajectory。观测指标：EM 0.5、format rate 1.0、平均搜索次数 0.5、工具失败率 0.0、`local_bm25` 请求 1 次且 success rate 1.0。

## 2026-07-20

- `my-search-r1/` 已迁移 `03-search-r1` 的训练级 rollout 与 PyTRIO GRPO train/eval 链路，并接入统一工具层。
  - 相关路径：`my-search-r1/search_r1_minilab/data.py`、`my-search-r1/search_r1_minilab/rollout.py`、`my-search-r1/search_r1_minilab/training.py`、`my-search-r1/search_r1_minilab/tooling.py`、`my-search-r1/scripts/eval_pytrio.py`、`my-search-r1/scripts/train_pytrio.py`、`my-search-r1/tests/test_rollout_training.py`
  - 验证方式：运行 `PYTHONPATH=my-search-r1 uv run python -m unittest discover -s my-search-r1/tests -v`；运行 `PYTHONPATH=my-search-r1 uv run python -m py_compile my-search-r1/scripts/eval_pytrio.py my-search-r1/scripts/train_pytrio.py`；运行 `PYTHONPATH=my-search-r1 uv run python my-search-r1/scripts/eval_pytrio.py --backend local_bm25 --limit 2 --batch-size 1`；运行 `PYTHONPATH=my-search-r1 uv run python my-search-r1/scripts/train_pytrio.py --max-steps 1 --questions-per-batch 1 --group-size 2 --backend local_bm25 --swanlab-mode disabled --run-name search-r1-minilab-smoke --save-every 0`。
  - 关键结果：25 个 unittest 全部通过；训练级 eval 使用默认 `Qwen/Qwen3.5-4B`、2 条公开 MiniLab fixture 和 `local_bm25` 完成，生成 2 条 trajectory，指标为 EM macro 0.5、format rate 1.0、平均搜索次数 0.5、`local_bm25` 请求 1 次且 success rate 1.0；1-step train smoke 完成 PyTRIO 训练客户端创建、sampler 权重导出、group rollout、trajectory 报告和最终 checkpoint 保存，本次 batch reward 全同导致 GRPO datum 为 0、`train/update_skipped=1.0`，未产生参数更新。

- `my-search-r1/` 已迁移 `03-search-r1` 的数据准备和 checkpoint 分析脚本。
  - 相关路径：`my-search-r1/search_r1_minilab/prepare_data.py`、`my-search-r1/search_r1_minilab/analysis.py`、`my-search-r1/scripts/prepare_data.py`、`my-search-r1/scripts/analyse_checkpoints.py`、`my-search-r1/tests/test_data_analysis.py`
  - 验证方式：运行 `PYTHONPATH=my-search-r1 uv run python -m unittest discover -s my-search-r1/tests -v`；运行 `PYTHONPATH=my-search-r1 uv run python -m py_compile my-search-r1/scripts/prepare_data.py my-search-r1/scripts/analyse_checkpoints.py`；运行两个脚本的 `--help`；运行 `PYTHONPATH=my-search-r1 uv run python my-search-r1/scripts/analyse_checkpoints.py --result-dir my-search-r1/eval_results --output my-search-r1/eval_results/checkpoint_em_format.png --dpi 120`。
  - 关键结果：34 个 unittest 全部通过；`analyse_checkpoints.py` 能从当前纯 trajectory JSONL 直接计算 `em/macro=0.5000`、`format/rate=1.0000` 并生成图；`prepare_data.py` 入口和数据清洗/抽样逻辑已验证，但未实际下载完整 ModelScope 数据集。

- `my-search-r1/` 迁移后真实测试矩阵已完成一次端到端验证。
  - 相关路径：`my-search-r1/datasets/`、`my-search-r1/eval_results/`、`my-search-r1/outputs/train_pytrio/search-r1-minilab-nondegenerate-smoke/`、`my-search-r1/scripts/eval_pytrio.py`
  - 验证方式：运行离线回归 `PYTHONPATH=my-search-r1 uv run python -m unittest discover -s my-search-r1/tests -v`、`PYTHONPATH=my-search-r1 uv run python -m compileall -q my-search-r1/search_r1_minilab my-search-r1/scripts`、`git diff --check`；运行 `prepare_data.py` 下载固定 ModelScope revision `aa2da0496c1b1a50a66af7acabdf09c07a0cb79e`；运行 `eval_pytrio.py` 的 local BM25 fixture、local BM25 dev、Zhihu dev-5、Zhihu dev；运行 `train_pytrio.py` 的 1-step Zhihu 非退化 GRPO smoke；运行 `analyse_checkpoints.py` 生成 local-vs-Zhihu 图。
  - 关键结果：34 个 unittest 全部通过；数据集生成 `train=169615`、`test=51713`、`dev=70`，dev 来源为 7 类各 10 条；local BM25 fixture eval 2 条，EM macro 0.5、format rate 1.0、搜索请求 1 次；local BM25 dev eval 70 条，EM macro 0.0143、format rate 0.1571、搜索请求 253 次、success rate 1.0、empty rate 0.5692；Zhihu dev-5 eval 5 条，EM macro 0.0、format rate 0.2、搜索请求 10 次、success rate 1.0；Zhihu dev eval 70 条，EM macro 0.2143、format rate 0.6714、搜索请求 128 次、success rate 1.0、empty rate 0.0547、无 429/timeout；非退化 train smoke 8 条 trajectory，`train/datums_per_rollout_batch=4`、`train/update_skipped=0`、`trainer/loss_mean=-0.0003145`、搜索请求 19 次、success rate 1.0，已实际执行 backward 和 optimizer step 并保存最终 checkpoint（路径不公开记录）。本轮发现并修复 `eval_pytrio.py` 默认 `--limit=2` 导致完整 dev 命令只评 2 条的问题，现默认 `--limit=0` 表示全量。

- 迁移测试失败案例复盘已完成。
  - 相关路径：`docs/interview/2026-07-20_migration_failure_case_review.md`、`my-search-r1/eval_results/zhihu_dev.jsonl`、`my-search-r1/eval_results/local_bm25_dev.jsonl`、`my-search-r1/outputs/train_pytrio/search-r1-minilab-nondegenerate-smoke/step_000001.jsonl`
  - 验证方式：读取上述 Markdown 报告和 JSONL，使用本地 Python 脚本按 `search_calls`、`exact_match`、`valid_format`、tool observation 空结果、重复 query、`stop_reason` 重新统计分类；运行 `git diff --check` 检查新增文档和状态文件。
  - 关键结果：已整理直接答对、搜索后答对、搜索后答错、搜索为空/弱相关、重复 query、格式错误、工具调用过多但无收益 7 类案例；确认 Zhihu dev 15 个正确中 14 个依赖搜索，local BM25 dev 唯一正确为直接答对；三份产物均无工具失败，主要风险集中在格式收束、query 改写、弱相关/空结果处理、证据阅读和严格 EM 对长答案/日期格式的误伤。

- 行为指标、报告 bucket 与默认关闭的轻量 reward penalty 已实现。
  - 相关路径：`my-search-r1/search_r1_minilab/diagnostics.py`、`my-search-r1/search_r1_minilab/rewards.py`、`my-search-r1/search_r1_minilab/rollout.py`、`my-search-r1/search_r1_minilab/training.py`、`my-search-r1/search_r1_minilab/trajectories/report.py`、`my-search-r1/scripts/eval_pytrio.py`、`my-search-r1/scripts/train_pytrio.py`、`docs/interview/2026-07-20_behavior_penalty_comparison.md`
  - 验证方式：运行 `PYTHONPATH=my-search-r1 uv run python -m unittest discover -s my-search-r1/tests -v`、`PYTHONPATH=my-search-r1 uv run python -m compileall -q my-search-r1/search_r1_minilab my-search-r1/scripts`、`git diff --check`；运行 `analyse_trajectories.py` 从既有 `zhihu_dev.jsonl` 生成新版行为报告；对既有 Zhihu dev trajectory 做离线 penalty 重评分。
  - 关键结果：39 个 unittest 全部通过；新版报告可显示 `direct_correct`、`searched_correct`、`searched_wrong`、空 observation、重复 query、`max_search_no_answer`、`too_many_search_no_gain` 等 bucket 和 rate；`eval_pytrio.py` 与 `train_pytrio.py` 均支持 `--duplicate-query-penalty`、`--empty-result-penalty`、`--max-search-no-answer-penalty`、`--verbose-answer-penalty`、`--verbose-answer-token-threshold`，默认值均不改变旧 reward。基于既有 Zhihu dev 的离线重评分显示平均 reward 从 0.1814 降至 0.1673，EM、format 和行为率不变。

## 2026-07-21

- Reward shaping checkpoint 训练效果对比已完成。
  - 相关路径：`docs/interview/2026-07-21_reward_shaping_checkpoint_comparison.md`、`my-search-r1/eval_results/reward_train_compare_2026-07-21/`、`my-search-r1/outputs/train_pytrio/reward-baseline-5step-20260721/`、`my-search-r1/outputs/train_pytrio/reward-penalty-5step-20260721/`
  - 验证方式：运行 `local_bm25 --limit 1` 和 `zhihu_search --limit 5` smoke；运行 base model Zhihu dev 70 条 eval；分别运行原始 reward 与 penalty reward 的 5-step Zhihu GRPO 训练；使用两个 final sampler weights 分别跑 Zhihu dev 70 条 eval；从 JSONL 和训练 step Markdown 汇总公开指标。
  - 关键结果：PyTRIO sampling 和 Zhihu backend 均恢复可用；三组 dev eval 均生成 70 条 trajectory，工具失败数均为 0，Zhihu success rate 均为 1.0。Base EM macro 0.2429、format rate 0.6143、平均搜索 1.9571；原始 reward checkpoint EM macro 0.3000、format rate 0.8286、平均搜索 1.7714；penalty reward checkpoint EM macro 0.2286、format rate 0.7286、平均搜索 1.5143。5-step 小预算下，penalty checkpoint 降低了搜索次数和 `too_many_search_no_gain_rate`，但未提升 EM/format；原始 reward checkpoint 表现最好。

- 原始 reward 20-step checkpoint 训练与 Zhihu dev 70 条评测已完成。
  - 相关路径：`docs/interview/2026-07-21_baseline_20step_eval.md`、`my-search-r1/eval_results/reward_train_compare_2026-07-21/base_20step_dev.jsonl`、`my-search-r1/eval_results/reward_train_compare_2026-07-21/base_20step_dev.md`、`my-search-r1/outputs/train_pytrio/reward-baseline-20step-20260721/`
  - 验证方式：运行 `train_pytrio.py` 的 20-step Zhihu GRPO 原始 reward 训练；使用 final sampler weights 运行 `eval_pytrio.py` 的 Zhihu dev 70 条评测；检查训练 step JSONL/Markdown、eval JSONL/Markdown 和终端指标；真实 PyTRIO checkpoint URI 不写入公开文档。
  - 关键结果：20 个 rollout step 全部完成，共 160 条训练 trajectory，17/20 个 optimizer update，skipped step 为 4、10、19；训练平均 reward 0.2888、correct rate 0.3125、format rate 0.7625、平均搜索 1.825。20-step checkpoint 的 Zhihu dev eval 生成 70 条 trajectory，工具失败数 0，Zhihu success rate 1.0，EM macro 0.2714、format rate 0.8571、平均搜索 1.7000；相对 base 净增 2 条正确、净增 17 条格式合规，但低于 5-step 原始 reward checkpoint 的 EM 0.3000。

- 5-step 与 20-step 原始 reward checkpoint 的 gained/lost case review 已完成。
  - 相关路径：`docs/interview/2026-07-21_5step_20step_gained_lost_review.md`、`my-search-r1/eval_results/reward_train_compare_2026-07-21/baseline_reward_ckpt_dev.jsonl`、`my-search-r1/eval_results/reward_train_compare_2026-07-21/base_20step_dev.jsonl`
  - 验证方式：离线逐行读取两个 eval JSONL，按同一 dev 顺序比较 `exact_match`，抽取 question、gold、final answer、search calls、queries 和 stop reason；未重新运行模型、搜索 API 或训练服务。
  - 关键结果：20-step 相对 5-step gained 2 条、lost 4 条。4 条 lost 样本为 `dev_4869`、`test_97`、`dev_2407`、`dev_3412`；其中 `dev_4869` 和 `dev_3412` 是真实退化，主要原因是少做必要二跳搜索或实体角色绑定错误；`test_97` 和 `dev_2407` 主要是严格 EM/答案粒度问题。结论是 20-step 格式更稳、搜索更少，但多跳 follow-up query 和最终答案粒度更脆。
