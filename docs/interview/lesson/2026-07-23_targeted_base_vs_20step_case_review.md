# Targeted Base vs 20-Step Case Review

记录时间：2026-07-23 11:05 CST

## 目标

在两个新构造的 targeted eval 集上，对 prompt-only base 与 `turn_credit_evidence_bridge_20step` 的有效结果做逐 case 复盘：

- `my-search-r1/datasets/bridge_eval_150.jsonl`
- `my-search-r1/datasets/alias_granularity_eval_80.jsonl`

本轮只做离线 review，不重新运行模型、训练、搜索 API 或 PyTRIO checkpoint。公开记录不包含远端 sampler weights URI。

## 输入产物

Bridge：

- base：`my-search-r1/eval_results/targeted_eval_20260723/bridge_prompt_base_20260723.jsonl`
- 20-step：`my-search-r1/eval_results/targeted_eval_20260723/bridge_turn_credit_evidence_20step_20260723.jsonl`
- diagnostics：`my-search-r1/eval_results/targeted_eval_20260723/bridge_prompt_base_20260723_offline_diagnostics.jsonl`、`my-search-r1/eval_results/targeted_eval_20260723/bridge_turn_credit_evidence_20step_20260723_offline_diagnostics.jsonl`
- aggregate comparison：`my-search-r1/eval_results/targeted_eval_20260723/bridge_targeted_three_way_comparison_20260723.md`

Alias/granularity：

- base：`my-search-r1/eval_results/targeted_eval_20260723/alias_prompt_base_retry_20260723.jsonl`
- 20-step：`my-search-r1/eval_results/targeted_eval_20260723/alias_turn_credit_evidence_20step_20260723.jsonl`
- diagnostics：`my-search-r1/eval_results/targeted_eval_20260723/alias_prompt_base_retry_20260723_offline_diagnostics.jsonl`、`my-search-r1/eval_results/targeted_eval_20260723/alias_turn_credit_evidence_20step_20260723_offline_diagnostics.jsonl`
- aggregate comparison：`my-search-r1/eval_results/targeted_eval_20260723/alias_base_vs_20step_comparison_20260723.md`

两组 base 与 20-step eval 的 Zhihu success rate 均为 1.0，因此可以进入 case review。

## Bridge 总体判断

20-step 相对 base：

- overall correct：74/150 -> 81/150，净增 7 条。
- EM macro：0.4750 -> 0.4583，下降 0.0167。
- format：0.7200 -> 0.9400，显著提升。
- avg search：3.3067 -> 3.0933，略降。
- gained/lost：12 gained、5 lost。

source-level 解释了 overall correct 上升但 macro EM 下降的原因：

| Source | Base EM | 20-step EM | Delta |
| --- | ---: | ---: | ---: |
| 2WikiMultihopQA | 0.5000 | 0.6000 | +0.1000 |
| HotpotQA | 0.5000 | 0.4333 | -0.0667 |
| Bamboogle | 0.6000 | 0.5000 | -0.1000 |
| MuSiQue | 0.3000 | 0.3000 | +0.0000 |

20-step 的收益集中在样本数最多的 2Wiki director comparison 子集，所以 overall correct 增加；但 HotpotQA 和 Bamboogle 下滑，按 source 宏平均后 EM macro 反而下降。

## Bridge Gained

12 个 gained 全部来自 2WikiMultihopQA，且 base 全部是 `valid_format=False`、`stop_reason=max_search_calls`；20-step 全部 `valid_format=True`、`stop_reason=answer`。这不是单纯检索能力提升，更准确地说是 20-step 在 bridge 题上改善了“搜索后收束成短答案”的能力。

| ID | Gold | Base 行为 | 20-step 行为 | 归因 |
| --- | --- | --- | --- | --- |
| `dev_10628` | `The Carousel Of Death` | 4 search 后继续第 5 次 query，未答 | 4 search 后答对 | 格式/停止策略修复 |
| `dev_7767` | `North Of Nevada` | 4 search 后重复 `Ricardo A. Solla birth date`，未答 | 3 search 后答对 | 去掉冗余 query，收束更好 |
| `dev_8262` | `Roads Of Kiarostami` | 4 search 后重复 Ryan Carroll birth date，未答 | 4 search 后答对 | 格式/停止策略修复 |
| `dev_8597` | `Grandmother's War Story` | 4 search 后继续 Lloyd Bacon death date，未答 | 4 search 后答对 | 格式/停止策略修复 |
| `dev_10322` | `Morecambe Church Lads' Brigade At Drill` | 4 search 后继续查 Mitchell and Kenyon founders，未答 | 4 search 后答对 | 格式/停止策略修复 |
| `dev_11245` | `Legion Of Terror` | 4 search 后继续查 Charles C. Coleman death date，未答 | 4 search 后答对 | 格式/停止策略修复 |
| `dev_11513` | `The Fiddler Of Florence` | 4 search 后继续查 Il Contratto director，未答 | 4 search 后答对 | 格式/停止策略修复 |
| `dev_11777` | `Ven Mi Corazón Te Llama` | 4 search 后继续查片名导演，未答 | 3 search 后答对 | 去掉冗余 query，收束更好 |
| `dev_11860` | `Temptations Of A Shop Girl` | 4 search 后继续查 Dui Jibon director，未答 | 4 search 后答对 | 格式/停止策略修复 |
| `dev_12533` | `Once Upon A Time In America` | 4 search 后继续查 Aashrayam director，未答 | 3 search 后答对 | 去掉冗余 query，收束更好 |
| `dev_7495` | `The Woman From Last Night` | 4 search 后继续查 Massy Tadjedin birth date，未答 | 4 search 后答对 | 格式/停止策略修复 |
| `dev_7595` | `God's Good Man` | 4 search 后继续查 Morris Elvey birth year，未答 | 4 search 后答对 | 格式/停止策略修复 |

关键结论：这些 gained 不说明 20-step 找到了 base 找不到的证据。base 多数已经走到了有效 follow-up query，但被 max-search 和 answer formatting 卡住；20-step 的主要收益是终止决策、短答案输出和格式约束。

## Bridge Lost

| ID | Source | Gold | Base answer | 20-step answer | 归因 |
| --- | --- | --- | --- | --- | --- |
| `train_80307` | HotpotQA | `Mutant Enemy Productions` | `Mutant Enemy Productions` | `The Mighty Boosh` | 实质退化。20-step 只查 `Joss Whedon founder organization` 就早答，没做 base 的第二次聚焦 query；属于 single-search role binding/证据阅读失败。 |
| `dev_7742` | 2Wiki | `The Heart Of St. Pauli` | `The Heart Of St. Pauli` | `Lukket Avdeling` | 实质退化。两组 query 基本相同，20-step 不是少搜，而是在同等证据下比较方向或出生日期绑定出错。 |
| `dev_2331` | HotpotQA | `Tony Award` | `Tony Award` | `Tony Awards` | strict EM 误伤为主。20-step 答案是复数形式，语义接近 gold；不应视为强能力退化，但暴露 answer normalization/alias 规则不足。 |
| `test_99` | Bamboogle | `June 5, 2004` | `June 5, 2004` | `March 2004` | 实质退化。20-step 在识别 Ronald Reagan 后少查 `Ronald Reagan death date`，提前给出错误月份。 |
| `dev_11691` | 2Wiki | `Les Mutinés De L'Elseneur` | `Les Mutinés De L'Elseneur` | `Kaatru Veliyidai` | 实质退化。20-step 查了 Pierre Chenal birth date 后提前比较，少查 Mani Ratnam birth date，属于必要二跳不足和比较方向错误。 |

Bridge lost 中，3 条是少做必要 follow-up 或早答，1 条是同证据比较错误，1 条是 strict EM/复数形式误伤。也就是说，20-step 的 format 收束变强后，副作用主要是更早停止，少数样本压掉了必要证据链。

## Alias/Granularity 总体判断

20-step 相对 base：

- overall correct：36/80 -> 35/80，净减 1 条。
- EM macro：0.4500 -> 0.4375。
- format：0.9250 -> 0.9625。
- avg search：1.6500 -> 1.5125。
- `possible_alias_match`：10 -> 10。
- `answer_granularity_miss`：0 -> 0。
- `multi_candidate_answer`：2 -> 2。
- gained/lost：1 gained、2 lost。

该集合上没有观察到新的答案粒度退化，也没有增加 alias risk；但 20-step 的更少搜索带来 2 个实体国籍题早答错。

## Alias/Granularity Changed Cases

| ID | Gold | Base answer | 20-step answer | 归因 |
| --- | --- | --- | --- | --- |
| `dev_12115` | `Stanisław Leszczyński` 等多个别名 | invalid/no answer | `Stanisław Leszczyński` | 真实收益。base 已查到 Sophie of France -> Marie Leszczyńska -> Stanisław Leszczyński 线索，但继续查父亲并触发 max-search；20-step 在 3 search 后正确收束。 |
| `dev_11271` | `Germany` 等多个别名 | `Germany` | `United States` | 实质退化。base 先查 1927 film director，再查 Gustav Pauli nationality；20-step 只查一次后绑定到 Lewis D. Collins 并早答美国。不是 strict EM 问题，因为 gold 已含 Germany 多种别名。 |
| `dev_8490` | `Canada` 等多个别名 | `Canada` | `United States` | 实质退化。base 通过 Chris Alexander nationality/birthplace 多次 follow-up 得到 Canada；20-step 两次 query 后早答 United States。不是 alias/granularity 误伤。 |

Alias/granularity 集的结论与 bridge 一致：20-step 更会停止并输出规范答案，但有时把“少搜”推进过头。对国籍/实体绑定题，当前 evidence_bridge 仍不足以稳定保护最后一跳消歧 query。

## 面向下一轮的结论

- 20-step 的最大确定性收益是格式和停止策略：bridge 12 gained 与 alias 1 gained 都是 base max-search/no-answer 被修复。
- 20-step 的主要风险是 early answer：bridge lost 的 `train_80307`、`test_99`、`dev_11691`，以及 alias lost 的 `dev_11271`、`dev_8490` 都指向必要 follow-up 被压缩。
- strict EM false negative 当前只明确命中 `dev_2331`，不是主风险；但应补 answer normalization，把 `Tony Award`/`Tony Awards` 这类变体纳入诊断。
- evidence_bridge v2 对 dev 70 的二跳保护有效，但 targeted eval 暴露了新类型：director comparison 下的日期比较、国籍题的 director-to-nationality 最后一跳、组织 founder role binding。下一轮不应盲目加大训练步数，而应补更细的 early-answer/final-hop 诊断与 credit。

## 建议

1. 保留 `turn_credit_evidence_bridge_20step` 作为当前有效 checkpoint，但在 targeted eval 结论中写清：它提升 overall correct 和 format，不提升 bridge macro EM，也不提升 alias/granularity EM。
2. 下一轮优先做 final-hop guard，而不是继续 50-step：如果问题类型包含 `country/nationality/death date/birth date/older/died earlier/founder organization`，且当前 query 序列没有查到目标属性 query，降低 early answer 的 turn/trajectory reward。
3. 增加 answer normalization 诊断：至少覆盖单复数、冠词、大小写、常见国家/组织别名，先作为 offline diagnostics，不直接改主 EM。
4. 若继续完整三模型 targeted eval，应先重跑 bridge 50-step 使 Zhihu success rate 达到 1.0；否则 50-step 仍只能作为参考，不进入正式 gated comparison。

## 验证方式

本轮离线读取并对齐以下文件，抽取 `exact_match`、`valid_format`、`stop_reason`、`search_calls`、query 序列、diagnostic reasons 和 final answer：

```bash
python - <<'PY'
import json
from pathlib import Path

base_dir = Path('my-search-r1/eval_results/targeted_eval_20260723')
paths = [
    base_dir / 'bridge_prompt_base_20260723.jsonl',
    base_dir / 'bridge_turn_credit_evidence_20step_20260723.jsonl',
    base_dir / 'bridge_prompt_base_20260723_offline_diagnostics.jsonl',
    base_dir / 'bridge_turn_credit_evidence_20step_20260723_offline_diagnostics.jsonl',
    base_dir / 'alias_prompt_base_retry_20260723.jsonl',
    base_dir / 'alias_turn_credit_evidence_20step_20260723.jsonl',
    base_dir / 'alias_prompt_base_retry_20260723_offline_diagnostics.jsonl',
    base_dir / 'alias_turn_credit_evidence_20step_20260723_offline_diagnostics.jsonl',
]
for path in paths:
    with path.open() as f:
        print(path, sum(1 for _ in f))
PY
```

另外用临时 Python 片段统计 gained/lost 的 source、format、stop reason 和 search delta。观测结果：bridge gained 12/lost 5，alias gained 1/lost 2；两组有效 run 的工具成功率均为 1.0。本轮未运行训练、模型推理、搜索 API 或单元测试。
