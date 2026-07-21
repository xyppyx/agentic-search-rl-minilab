# 5-Step vs 20-Step Gained/Lost Case Review

复盘时间：2026-07-21 20:35 CST

## 目标

对比同一套 Zhihu dev 70 题上：

- `5step`：原始 reward 5-step checkpoint，`baseline_reward_ckpt_dev.jsonl`
- `20step`：原始 reward 20-step checkpoint，`base_20step_dev.jsonl`

重点找出 20-step 相对 5-step 的 gained/lost case，尤其是 20-step 丢掉的 4 条 5-step 正确样本。

## 方法

逐行读取两个 eval JSONL，按同一 dev 顺序比较 `exact_match`：

- `gained`：5-step 错，20-step 对。
- `lost`：5-step 对，20-step 错。

本次只做离线 JSONL 复盘，没有重新调用模型、搜索 API 或训练服务。

## 汇总

| 类别 | Count | 样本索引 |
| --- | ---: | --- |
| 20-step gained | 2 | 46, 55 |
| 20-step lost | 4 | 6, 19, 21, 30 |

总体指标：

| 指标 | 5-step | 20-step | 变化 |
| --- | ---: | ---: | ---: |
| correct count | 21 | 19 | -2 |
| EM | 0.3000 | 0.2714 | -0.0286 |
| format count | 58 | 60 | +2 |
| format rate | 0.8286 | 0.8571 | +0.0286 |
| avg search calls | 1.7714 | 1.7000 | -0.0714 |

结论：20-step 比 5-step 更会输出合规格式、搜索次数略少，但答对数少 2 条。丢失样本中，有 2 条是严格 EM/答案粒度问题，2 条是真实推理或检索策略退化。

## 20-Step Gained Cases

### Gained 1: Spain national team top scorer

- Index：46
- ID：`test_494`
- Source：`nq`
- Question：`all time highest goal scorer in spain national team?`
- Gold：`David Villa`

| Run | Correct | Format | Search calls | Query | Final answer |
| --- | --- | --- | ---: | --- | --- |
| 5-step | false | true | 1 | `all time highest goal scorer Spain national team` | `Alvaro Morata` |
| 20-step | true | true | 1 | `all time highest goal scorer Spain national team` | `David Villa` |

判断：

- 两个 checkpoint query 相同，差异主要在 evidence reading。
- 5-step 被搜索结果中的近期或干扰信息带偏，回答了 `Alvaro Morata`。
- 20-step 正确选择了 `David Villa`，说明 20-step 在这类单跳事实题上有时更能抵抗干扰。

### Gained 2: Spring Waltz director

- Index：55
- ID：`test_7511`
- Source：`popqa`
- Question：`Who was the director of Spring Waltz?`
- Gold：`Yoon Seok-Ho`

| Run | Correct | Format | Search calls | Query | Final answer |
| --- | --- | --- | ---: | --- | --- |
| 5-step | false | true | 1 | `Spring Waltz film director` | `Yun Seok-ho` |
| 20-step | true | true | 1 | `Spring Waltz film director` | `Yoon Seok-ho` |

判断：

- 两个 checkpoint 找到的是同一实体，差异主要是 strict EM 的英文转写。
- 5-step 的 `Yun Seok-ho` 语义上接近，但没有命中 gold `Yoon Seok-Ho`。
- 20-step 输出了 gold spelling，因此被 EM 判对。

## 20-Step Lost Cases

### Lost 1: Claudia Antonia maternal grandfather

- Index：6
- ID：`dev_4869`
- Source：`2wikimultihopqa`
- Question：`Who is the maternal grandfather of Claudia Antonia?`
- Gold：`Sextus Aelius Catus`

| Run | Correct | Format | Search calls | Queries | Final answer |
| --- | --- | --- | ---: | --- | --- |
| 5-step | true | true | 2 | `Claudia Antonia maternal grandfather`; `Aelia Paetina father` | `Sextus Aelius Catus` |
| 20-step | false | true | 1 | `Claudia Antonia maternal grandfather` | `Gaius Silius` |

判断：

- 这是最典型的 20-step 退化样本。
- 5-step 先定位 Claudia Antonia 的母亲 Aelia Paetina，再追问 `Aelia Paetina father`，完成二跳检索。
- 20-step 只搜第一跳就作答，把母系父亲误判成 `Gaius Silius`。
- 归因：20-step 的搜索次数下降在这里损伤了必要二跳检索；需要奖励“证据不足时继续查关键中间实体”，而不是单纯压低搜索次数。

### Lost 2: Quit India speech speaker birth date

- Index：19
- ID：`test_97`
- Source：`bamboogle`
- Question：`When was the person who delivered the "Quit India" speech born?`
- Gold：`October 2, 1869`

| Run | Correct | Format | Search calls | Query | Final answer |
| --- | --- | --- | ---: | --- | --- |
| 5-step | true | true | 1 | `who delivered Quit India speech` | `October 2, 1869` |
| 20-step | false | true | 1 | `Quit India speech speaker` | `1869` |

判断：

- 20-step 实际推理内容包含完整生日，但最终 `Answer:` 只输出年份。
- 这是严格 EM 下的答案粒度退化，不是搜索失败。
- 归因：训练后模型更倾向压缩最终答案，但对日期题压缩过度；需要 final answer 粒度约束，比如日期题保留完整日期。

### Lost 3: Martin Luther King III vs Dexter

- Index：21
- ID：`dev_2407`
- Source：`hotpotqa`
- Question：`Who is younger, Martin Luther King III or his brother Dexter?`
- Gold：`Dexter`

| Run | Correct | Format | Search calls | Queries | Final answer |
| --- | --- | --- | ---: | --- | --- |
| 5-step | true | true | 2 | `Martin Luther King III birth date`; `Dexter King birth date` | `Dexter` |
| 20-step | false | true | 2 | `Martin Luther King III birth date`; `Dexter King birth date` | `Dexter King` |

判断：

- 两个 checkpoint 的 query 和推理都基本正确。
- 20-step 输出 `Dexter King`，gold 是 `Dexter`，严格 EM 判错。
- 这是答案别名/粒度问题，不应被解释为真实能力下降。
- 后续评测需要补充 alias-aware metric 或至少在报告里区分 strict EM false negative。

### Lost 4: I Will Not Say Goodbye writer talent competition

- Index：30
- ID：`dev_3412`
- Source：`hotpotqa`
- Question：`"I Will Not Say Goodbye" is a song written in part by a music artist who first gained national attention as a winner of what talent competition?`
- Gold：`You Can Be a Star`

| Run | Correct | Format | Search calls | Queries | Final answer |
| --- | --- | --- | ---: | --- | --- |
| 5-step | true | true | 2 | `"I Will Not Say Goodbye" song writer talent competition winner`; `Lari White American Idol winner` | `You Can Be a Star` |
| 20-step | false | true | 1 | `"I Will Not Say Goodbye" song writer talent competition winner` | `American Idol` |

判断：

- 这是另一个真实退化样本。
- 题目问的是“这首歌的某位 writer first gained attention as winner of what talent competition”，关键实体是 writer `Lari White`，不是 recorded artist `Danny Gokey`。
- 5-step 做了第二次搜索并锁定 Lari White，得到 `You Can Be a Star`。
- 20-step 只做一次搜索后跳到 Danny Gokey/American Idol，属于 evidence reading 和实体角色绑定错误。
- 归因：20-step 的搜索行为更收敛，但对多实体问题更容易过早作答。

## 归因分类

| 类型 | Cases | 说明 |
| --- | --- | --- |
| 严格 EM/答案粒度 false negative | Lost 2, Lost 3, Gained 2 | 日期只答年份、`Dexter` vs `Dexter King`、转写差异都会影响 EM。 |
| 必要二跳搜索被压缩 | Lost 1, Lost 4 | 20-step 少搜一次后过早作答，损伤多跳问题。 |
| Evidence reading 抗干扰提升 | Gained 1 | 20-step 在同 query 下选对事实，5-step 被干扰项带偏。 |
| 格式收束提升但可能过度简化 | Lost 2 | 20-step 格式更稳，但最终答案有时过短。 |

## 对下一轮 Reward 的启发

1. 不宜继续单纯压低搜索次数。20-step 搜索更少，但 Lost 1 和 Lost 4 都说明多跳题需要允许第二次关键检索。
2. Reward shaping 应区分“无收益重复搜索”和“必要中间实体搜索”。可以惩罚 duplicate query，但不要惩罚不同实体的 follow-up query。
3. 增加 final answer 类型约束：日期题保留完整日期；比较题可接受短名和全名，但最终答案应优先贴近候选项或 gold 粒度。
4. 增加 alias-aware 或 relaxed metric 作为辅助报告。严格 EM 仍保留，但需要单独统计 `Dexter King` 这类 false negative，避免误判训练方向。
5. 多实体问题需要 role binding 信号：writer、recorded artist、director、speaker 等关系词应该进入 query/reasoning 检查。

## 结论

20-step 丢掉的 4 条 5-step 正确样本分别是：

- `dev_4869`：Claudia Antonia maternal grandfather，少一次二跳搜索，真实退化。
- `test_97`：Quit India speech birth date，最终答案只输出年份，严格 EM/粒度退化。
- `dev_2407`：Martin Luther King III vs Dexter，输出全名导致 strict EM 判错。
- `dev_3412`：I Will Not Say Goodbye，实体角色从 writer 跳到 recorded artist，真实退化。

因此，20-step 并非整体变差，而是“格式更稳、搜索更少、但多跳实体绑定和最终答案粒度更脆”。下一轮应优先做必要 follow-up query 与 answer granularity 的 reward/diagnostic，而不是加大搜索 penalty。
