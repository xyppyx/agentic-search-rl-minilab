# Search-R1 MiniLab Ideas

本文记录把上游 Search-R1 教学复现扩展成个人项目的设计想法。原 `03-search-r1/` 教学目录已归档到 `Backup` 分支，`main` 的正式实现位于 `my-search-r1/`。

项目主题可以暂定为：

```text
Robust Search-R1 MiniLab:
低成本、多工具、不可靠环境下的搜索型 Agent RL 实验框架
```

核心目标不是追求论文级分数，而是建立一套更可控、可观测、可扩展的 Agentic RL 实验环境。

## 总体优先级

```text
P0: 搜索工具封装 + 轨迹可视化
P1: reward / penalty 组件化
P2: 概率失败工具环境
P3: PyTRIO 后端封装
P4: 阶段化训练
```

原因：

- 先做工具封装，后续才能自然支持知乎、其他低成本搜索、本地 mock、失败注入。
- 先做轨迹可视化，才能解释 reward 曲线背后的真实行为。
- reward 改进需要轨迹支持，否则很难判断模型是在变好、变坏，还是工具环境污染了信号。
- PyTRIO 封装有价值，但可以先做薄封装，不急着迁移训练后端。
- 阶段化训练可能有用，但需要稳定评测和可视化后再验证。

## 1. 搜索工具封装

当前 `rollout.py` 直接依赖 `ZhihuSearchClient`。可以抽象出统一工具层，让 rollout 只关心“调用工具并得到 observation”。

建议结构：

```text
my-search-r1/
  tools/
    base.py
    registry.py
    zhihu.py
    mock.py
    failure.py
```

接口草图：

```python
class SearchBackend:
    name: str

    def search(self, query: str) -> SearchResult:
        ...


class ToolRegistry:
    def call(self, tool_name: str, arguments: dict) -> SearchResult:
        ...
```

第一批 backend：

```text
zhihu_search     现有知乎 API
mock_search      本地固定/规则返回，用于无费用调试
local_bm25       可选，本地小语料检索
serp_search      可选，其他低成本搜索服务
```

收益：

- 减少 `rollout.py` 对具体搜索服务的耦合。
- 可以在不改训练逻辑的情况下替换搜索环境。
- 后续能做“多工具选择”和“工具失败鲁棒性”实验。

## 2. 概率失败工具环境

在工具封装之后，为任意 backend 包一层 `FailureWrapperBackend`，模拟真实工具环境的不稳定性。

研究问题：

```text
当搜索工具可能失败、超时、返回空结果或噪声结果时，
模型是否能学会重试、换 query、停止搜索，或基于已有证据回答？
```

失败类型：

```text
timeout
rate_limited
empty_result
noisy_result
stale_result
wrong_result
```

配置示例：

```yaml
tool_failure:
  enabled: true
  p_timeout: 0.05
  p_empty: 0.10
  p_noise: 0.10
```

需要新增指标：

```text
tool/failure_rate
tool/timeout_rate
tool/empty_rate
tool/noise_rate
tool/retry_after_failure_rate
tool/final_answer_after_failure_rate
tool/repeated_failed_query_rate
```

注意：

- 不要只做一个总的 `p_fail`，失败类型应该可区分。
- 失败注入应该能关闭，作为 clean environment baseline。
- 训练时要区分“模型策略差”和“工具环境失败”，否则 reward 会被污染。

## 3. 轨迹可视化

轨迹可视化应该尽早做。Agentic RL 的很多问题不会直接体现在平均 reward 中，而是在具体 trajectory 里。

第一步先保存 JSONL，不急着做复杂 Web UI。

记录格式草图：

```json
{
  "question": "...",
  "answers": ["..."],
  "data_source": "hotpotqa",
  "turns": [
    {
      "role": "assistant",
      "text": "...",
      "tool_call": {"name": "search", "query": "..."}
    },
    {
      "role": "tool",
      "tool_name": "search",
      "ok": true,
      "items": []
    },
    {
      "role": "assistant",
      "text": "Answer: ..."
    }
  ],
  "reward": 1.0,
  "advantage": 0.37,
  "valid_format": true,
  "exact_match": true,
  "search_calls": 2,
  "tool_failures": 0
}
```

报告可以先做 Markdown 或静态 HTML：

```text
正确案例
错误案例
格式错误案例
搜索失败后成功案例
重复搜索案例
高 advantage / 低 advantage 对比
同一问题 group 内多轨迹对比
```

收益：

- 快速定位模型是否学会了停止搜索并输出 `Answer:`。
- 观察 query 是否重复、过长、空泛或被错误 observation 误导。
- 对比 base、step 5、step 20 checkpoint 的行为变化。

## 4. 阶段化训练

阶段化训练暂时存疑，但值得作为后续实验。

合理假设：

```text
先学会格式和基础搜索，再学多跳搜索，可能降低探索难度。
```

可能设计：

```text
Phase 1: NQ / 单跳问题，max_search_calls=1-2
Phase 2: HotpotQA / 多跳问题，max_search_calls=4
Phase 3: mixed replay，混合 NQ + HotpotQA，避免遗忘和分布偏移
```

风险：

- 模型可能先学会“少搜或单跳就回答”，到多跳阶段不愿继续搜索。
- 数据分布和训练配置改变后，reward 曲线很难与普通 baseline 公平比较。
- 如果没有轨迹可视化，很难判断阶段化到底带来了什么。

建议：

- 暂不作为第一阶段工作。
- 等工具环境、轨迹报告、固定评测集稳定后，再做 `curriculum vs mixed` 对照。

## 5. Reward 与 Penalty 改进

当前 reward 很简单：

```text
格式合法且答案正确: 1.0
格式合法但答案错误: 0.0
格式非法或没有答案: -0.1
```

可以先把 reward 拆成组件，而不是一开始就加很多惩罚。

组件草图：

```text
correct_reward
format_reward
invalid_tool_penalty
repeat_query_penalty
empty_query_penalty
tool_failure_handling_reward
```

示例：

```python
reward = correctness + 0.1 * format_score - 0.02 * invalid_tool_calls
```

需要谨慎的点：

- 不要过早惩罚搜索次数。Search-R1 的目标就是学会何时搜索，过强的 search penalty 可能把模型训成不敢搜索。
- 先记录坏行为指标，再决定是否惩罚。
- penalty 应该小而可解释，避免盖过最终答案正确性。

优先记录的行为指标：

```text
rollout/search_calls
rollout/no_search_rate
rollout/repeated_query_count
rollout/empty_query_count
rollout/invalid_tool_call_count
rollout/answer_after_tool_failure_rate
```

## 6. PyTRIO 封装

当前 `train.py` 直接调用 PyTRIO API。短期可以做薄封装，把训练后端细节收束到一个模块。

建议结构：

```text
my-search-r1/
  backend/
    base.py
    pytrio_backend.py
```

接口草图：

```python
class TrainingBackend:
    def create_or_resume_trainer(self, args): ...
    def get_tokenizer(self): ...
    def make_sampler(self): ...
    def forward_backward(self, datums, loss_fn: str): ...
    def optim_step(self, adam_params): ...
    def save_checkpoint(self, name: str): ...
```

短期收益：

- `train.py` 更像算法流程，不被 PyTRIO API 细节占据。
- 之后如果尝试 AutoDL / Transformers / TRL / vLLM，本地训练后端有替换入口。

注意：

- 第一阶段不需要真的迁移出 PyTRIO。
- 封装先薄一些，不要为了“未来可能支持很多后端”过度抽象。

## 推荐路线

第一阶段：可观测 + 可插拔

```text
1. 抽象 SearchBackend / ToolRegistry
2. 保留 Zhihu backend
3. 增加 Mock backend
4. rollout 保存完整 trajectory JSONL
5. 增加 trajectory report
```

第二阶段：鲁棒工具环境

```text
1. 增加 FailureWrapperBackend
2. 支持 p_timeout / p_empty / p_noise
3. 记录 tool failure 指标
4. 跑 5-step 对比：无失败 vs 20% 失败
```

第三阶段：reward 组件化

```text
1. 拆出 reward components
2. 添加 invalid tool penalty
3. 添加 repeat query penalty
4. 对比 reward 曲线与轨迹报告
```

第四阶段：训练框架与数据策略

```text
1. PyTRIO backend 薄封装
2. 阶段化训练实验
3. 更大评测集与更多 checkpoint 对比
```

## 近期最小可做版本

建议先做一个很小但完整的版本：

```text
目标：让 Search-R1 smoke/5-step run 产生可复查的轨迹报告。

任务：
1. 给 Trajectory 增加可序列化导出函数
2. train.py 增加 --trajectory-output 参数
3. eval.py 也保存同格式 trajectory JSONL
4. 新增 analyse_trajectories.py，输出 Markdown 报告
5. 报告按 correct / wrong / invalid_format / tool_failure 分类
```

这个版本不会动训练算法，但能显著提高理解和 debug 能力。
