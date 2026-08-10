# Turn-Level Search Credit 5/20-Step 训练复盘

记录时间：2026-07-22 19:08 CST

## 目标

在 `exp/turn-reward-helpful-search` 分支上验证默认关闭的 turn-level search credit 是否能提升多跳搜索动作学习：

- final answer turn 仍使用原 trajectory advantage。
- wrong-valid trajectory 中，保守判定为 helpful bridge follow-up 的搜索 turn 使用 `max(advantage, 0.0) + 0.10`。
- 不接入 LLM judge，不做人工在线标注；人工只用于训练后抽查与复盘。

## 本地 Smoke

PyTRIO sampling 恢复后，重跑 local BM25 1-step smoke：

```bash
PYTHONPATH=my-search-r1 uv run python my-search-r1/scripts/train_pytrio.py \
  --data my-search-r1/tests/fixtures/smoke_eval.jsonl \
  --max-steps 1 \
  --questions-per-batch 1 \
  --group-size 2 \
  --backend local_bm25 \
  --turn-credit-policy helpful_bridge \
  --helpful-search-turn-bonus 0.10 \
  --advantage-normalization standardize \
  --advantage-clip 2.0 \
  --kl-coef 0.01 \
  --policy-ratio-clip 0.2 \
  --learning-rate 1e-5 \
  --swanlab-mode disabled \
  --save-every 0 \
  --run-name turn-credit-local-smoke-20260722-retry
```

结果：

- 进入 reference logprobs、custom loss、optimizer 和 final checkpoint 路径。
- `mean_reward=0.5000`、`correct_rate=0.5000`、平均搜索 1.0。
- 说明 turn-credit 代码路径与 KL/std custom loss 可共同运行。

## Zhihu 5-Step

训练配置：

- run name：`turn-credit-helpful-bridge-5step-20260722`
- base：当前 prompt budget guard + KL/std 默认训练配置
- backend：`zhihu_search`
- steps：5
- group size：4
- turn credit：`helpful_bridge`
- bonus：0.10
- SwanLab：disabled

训练结果：

- 5/5 step 完成，4/5 step 执行 optimizer update，step 5 因无有效 datum 跳过。
- 训练 JSONL 中共记录 20 个 credited helpful search turn，覆盖 16 个 step-local trajectory。
- 训练阶段未观察到 Zhihu timeout、429、credential/http error 或 tool failure。

Dev-5：

- EM 0.4000
- format 0.8000
- 平均搜索 3.0000
- helpful follow-up rate 1.0000
- Zhihu success rate 1.0

完整 dev 70：

- EM 0.4286，30/70
- format 0.9714
- 平均搜索 1.8571
- no-search rate 0
- helpful follow-up rate 0.4143
- bad max-search loop rate 0.0286
- too many search no gain rate 0.1571
- Zhihu requests 130，success rate 1.0
- offline diagnostics：`wrong_valid=38`、`possible_alias_match=8`、`missing_followup_query=0`、`answer_granularity_miss=0`、`bad_max_search_loop=2`、`multi_candidate_answer=0`

相对当前最强 `prompt_budget_kl_std_5step`：

- EM 持平：0.4286 vs 0.4286
- format 更高：0.9714 vs 0.9429
- 平均搜索更低：1.8571 vs 1.9429
- gained 1：`test_7511`
- lost 1：`test_2231`

判断：5-step turn credit 不是明确 EM 提升，但改善 format 和搜索效率，且保持 `missing_followup_query=0`、`answer_granularity_miss=0`。可以作为一个正向但弱增益的小预算实验结果。

## Zhihu 20-Step

训练配置：

- run name：`turn-credit-helpful-bridge-20step-20260722`
- 其余参数沿用 5-step，仅将 `--max-steps` 改为 20，`--save-every` 为 5。

训练结果：

- 20/20 step 完成，保存 step 5/10/15/20/final。
- 14/20 step 执行 optimizer update。
- 训练 JSONL 中共记录 36 个 credited helpful search turn，覆盖 33 个 step-local trajectory。
- 训练阶段未观察到 PyTRIO sampling 长时间 await、Zhihu timeout、429、credential/http error 或 tool failure。

Dev-5：

- EM 0.4000
- format 1.0000
- 平均搜索 2.8000
- helpful follow-up rate 0.8000
- bad max-search loop rate 0.2000
- Zhihu success rate 1.0

完整 dev 70：

- EM 0.3714，26/70
- format 1.0000
- 平均搜索 1.6000
- no-search rate 0
- helpful follow-up rate 0.3000
- bad max-search loop rate 0.0286
- too many search no gain rate 0.1143
- empty observation rate 0.0179
- Zhihu requests 112，success rate 1.0
- offline diagnostics：`wrong_valid=44`、`possible_alias_match=8`、`missing_followup_query=3`、`answer_granularity_miss=0`、`bad_max_search_loop=2`、`multi_candidate_answer=0`

相对 turn-credit 5-step：

- gained 1：`test_2231`
- lost 5：`dev_174`、`dev_2429`、`dev_3741`、`dev_4869`、`test_7511`

相对当前最强 `prompt_budget_kl_std_5step`：

- gained 0
- lost 4：`dev_174`、`dev_2429`、`dev_3741`、`dev_4869`

判断：20-step 继续训练提高了 format 并降低平均搜索，但 EM 明显退化，`missing_followup_query` 从 0 回升到 3，说明它仍然会把模型推向更保守、更早收束的搜索策略。该配置不扩到 50-step。

## 结论

Turn-level search credit 的首版结论是：

- 机制有效生效：训练 artifact 中能看到 credited turn，且 wrong-valid 全 0 reward batch 也能产生部分 optimizer update。
- 5-step 可以追平当前最强 EM，并改善 format / 平均搜索，是可讲述的“小正向实验”。
- 20-step 不通过扩大训练门控；它复现了 KL/std 20-step 的核心问题，即更高 format 和更少搜索不等于更高 EM，必要 follow-up/role binding 仍会被继续训练压掉。

下一轮不应简单增加训练步数。更有价值的方向是把 turn credit 从“query 形态合理”升级到“query 是否命中 bridge entity / 是否减少候选歧义”的证据级 credit，或做 5-step 多 seed 稳定性验证。
