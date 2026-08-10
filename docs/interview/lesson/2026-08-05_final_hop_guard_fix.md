# Final-Hop Guard 修正记录

记录时间：2026-08-05 CST

## 目标

修正 `turn_credit_final_hop_bridge_v3` 暴露的两个问题：

- format/max-search/final-answer guard：搜过之后仍然 max-search 不答或输出 invalid final answer 时，对最后一轮 assistant turn 给负向 credit。
- missing final-hop attribute：让真实 bridge wrong-valid 样本中“没有查最终属性就回答”的情况能被 `missing_final_hop_attribute` 命中。

本轮只做代码、离线分析和 local BM25 1-step smoke；没有重新运行 Zhihu 训练或 eval。

## 实现

- `TurnCreditConfig` 新增 `final_answer_guard_turn_penalty`，默认 `0.0`，旧策略和旧默认行为不变。
- `train_pytrio.py` 新增 `--final-answer-guard-turn-penalty`。
- `final_hop_bridge` 在 wrong invalid/max-search 样本上允许标注 `final_answer_guard`，不再因为 `valid_format=False` 直接跳过所有 final-turn guard。
- `detect_missing_final_hop_risk(...)` 的覆盖判断改为以 query 显式属性词为主；date 类不再因为 observation 中出现任意年份就视为已覆盖。
- 补充 `birthdate`、`deathdate` 等合写 query term，避免 older/death date 题被误伤。
- `analyse_turn_credit.py` 新增 `final_answer_guard_penalty_records` 和对应案例区。

## 离线检查

对本轮已有 bridge eval 结果运行：

```bash
PYTHONPATH=my-search-r1 uv run python my-search-r1/scripts/analyse_turn_credit.py \
  --input my-search-r1/eval_results/final_hop_bridge_20260805/final_hop_5step_bridge_20260805.jsonl \
  --jsonl-output /tmp/final-hop-guard-analysis.jsonl \
  --report-output /tmp/final-hop-guard-analysis.md \
  --title 'Final-Hop Guard Fixed Analysis'
```

结果：

| Metric | Value |
| --- | ---: |
| total | 150 |
| wrong_valid | 37 |
| final_hop_training_credit_turns | 41 |
| missing_final_hop_penalty_records | 1 |
| final_answer_guard_penalty_records | 34 |

`missing_final_hop_attribute` 命中的真实样本是 `test_65`：问题要求 national capital city 的 established year，但 query 只有 `most populous national capital city`，没有查 established/founded/year 属性；final answer 为 `969`，gold 为 `1045 BC`。

`final_answer_guard` 主要命中 bridge 中 stop_reason 为 `max_search_calls` 且无合法 `Answer:` 的样本，覆盖本轮 format 退化的主因。

## 验证

- `PYTHONPATH=my-search-r1 uv run python -m unittest my-search-r1/tests/test_rollout_training.py -v`：34 tests OK。
- `PYTHONPATH=my-search-r1 uv run python -m unittest discover -s my-search-r1/tests -v`：87 tests OK。
- `PYTHONPATH=my-search-r1 uv run python -m compileall -q my-search-r1/search_r1_minilab my-search-r1/scripts`：通过。
- `git diff --check`：通过。
- local BM25 1-step smoke：完成 rollout、reference logprobs、custom loss、optimizer 和 final state/weights 保存；公开文档不记录远端 URI。

## 下一步

下一轮小预算训练建议在原 final-hop v3 参数上增加：

```text
--final-answer-guard-turn-penalty 0.06
```

仍保留：

```text
--turn-credit-policy final_hop_bridge
--evidence-search-turn-bonus 0.05
--final-hop-search-turn-bonus 0.10
--early-answer-turn-penalty 0.05
--missing-final-hop-turn-penalty 0.08
```

训练前仍需 Zhihu dev-5 health，success rate 必须为 1.0；若重新跑 bridge eval，目标不变：bridge EM macro 高于 0.4750、correct 高于 74/150、format 不低于 0.9000。
