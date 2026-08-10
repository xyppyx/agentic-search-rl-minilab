# Penalty v2 Max-Search Gained/Lost Case Review

复盘时间：2026-07-21 22:19 CST

## 目标

对比同一套 Zhihu dev 70 题上的三个 checkpoint：

- `v2_20`：`penalty_v2_candidate` 20-step checkpoint，`penalty_v2_20step_dev.jsonl`
- `max001`：`penalty_v2_plus_max_search_001` 5-step checkpoint，`penalty_v2_maxsearch001_5step_dev.jsonl`
- `max0005`：`penalty_v2_plus_max_search_0005` 5-step checkpoint，`penalty_v2_maxsearch0005_5step_dev.jsonl`

重点确认：

1. 哪些样本从 max-search penalty 中受益。
2. 哪些样本因为过早停止、答案粒度或答案过宽而损失。
3. `max_search_no_answer_penalty=0.005` 是否比 `0.01` 更稳。

本次只读取既有 eval JSONL 和 offline diagnostics JSONL，没有重新调用模型、搜索 API 或训练服务。

## 总体对比

| 指标 | v2_20 | max001 | max0005 |
| --- | ---: | ---: | ---: |
| correct count | 25 | 21 | 19 |
| `em/macro` | 0.3571 | 0.3000 | 0.2714 |
| format count | 51 | 61 | 57 |
| `format/rate` | 0.7286 | 0.8714 | 0.8143 |
| avg search calls | 2.6143 | 1.4571 | 1.5714 |
| `missing_followup_query` | 1 | 5 | 5 |
| `answer_granularity_miss` | 0 | 2 | 0 |
| `possible_alias_match` | 8 | 9 | 11 |

Exact-match 三元模式：

| v2_20 | max001 | max0005 | Count | 说明 |
| --- | --- | --- | ---: | --- |
| false | false | false | 41 | 三者都错 |
| false | true | false | 2 | 只有 max001 对 |
| false | true | true | 2 | 两个 max-search 版本都修复 |
| true | false | false | 6 | 两个 max-search 版本都丢失 |
| true | false | true | 2 | 只有 max001 丢失 |
| true | true | false | 2 | 只有 max0005 丢失 |
| true | true | true | 15 | 三者都对 |

## Pairwise Gained/Lost

| 对比 | Gained | Lost | 净变化 |
| --- | ---: | ---: | ---: |
| v2_20 -> max001 | 4 | 8 | -4 |
| v2_20 -> max0005 | 2 | 8 | -6 |
| max001 -> max0005 | 2 | 4 | -2 |

## Max-Search 共同受益样本

这两条是 `v2_20` 错、`max001` 和 `max0005` 都对。

### `test_108`: Song 后继王朝末代皇帝

- Gold：`Toghon Temür`
- v2_20：4 次搜索后 `max_search_calls`，无最终答案。
- max001：1 次搜索，答 `Toghon Temür`。
- max0005：1 次搜索，答 `Toghon Temür`。

判断：

- 这是 max-search 约束真正有收益的样本。
- v2_20 在找到“元朝是宋后继王朝”后继续改写 query，后续被 `Zhu Di` 等干扰带偏，最终用满 search budget。
- 两个 max-search 版本更快停在有效证据上，格式和答案都正确。

### `dev_2223`: Italian navigator 的 child

- Gold：`Sebastian Cabot`
- v2_20：4 次搜索后 `max_search_calls`，无最终答案。
- max001：3 次搜索，答 `Sebastian Cabot`。
- max0005：2 次搜索，答 `Sebastian Cabot`。

判断：

- 这是另一个有效压制空转的样本。
- v2_20 多次围绕错误实体 `Francisco Vázquez de Coronado` 打转；max-search 版本更快改写到 “Italian navigator sailed for England explored east coast of North America”，锁定 John Cabot / Sebastian Cabot 关系。
- `max0005` 在这里比 `max001` 更省一次搜索，但两者都正确。

## max001 独有受益样本

### `dev_2407`: Martin Luther King III vs Dexter

- Gold：`Dexter`
- v2_20：2 次搜索，答 `Martin Luther King III`，真实错误。
- max001：2 次搜索，答 `Dexter`，正确。
- max0005：2 次搜索，答 `Dexter King`，strict EM 判错，但语义上是同一人。

判断：

- max001 这里是真正答对，主要来自比较阅读更准确，不是搜索次数差异。
- max0005 不是能力显著退化，而是 strict EM/答案粒度问题：`Dexter King` 与 gold `Dexter` 指向同一候选。

### `dev_3412`: I Will Not Say Goodbye writer talent competition

- Gold：`You Can Be a Star`
- v2_20：3 次搜索，答 `American Idol`。
- max001：4 次搜索，答 `You Can Be a Star`。
- max0005：1 次搜索，答 `American Idol`，diagnostic 标为 `missing_followup_query`。

判断：

- max001 是三个 checkpoint 中唯一真正完成实体角色绑定的版本。
- 关键是从 song writer 追到 `Lari White`，再追她 first gained national attention 的比赛。
- max0005 在第一跳后过早作答，和 v2_20 一样落到 `American Idol` 干扰项；这是 0.005 的真实 lost case。

## max0005 修复 max001 的样本

这两条都是 max001 的 `answer_granularity_miss`，max0005 修复。

### `test_99`: Tear Down This Wall president death date

- Gold：`June 5, 2004`
- v2_20：2 次搜索，答 `June 5, 2004`。
- max001：1 次搜索，答 `2004`，粒度不足。
- max0005：1 次搜索，答 `June 5, 2004`。

判断：

- max001 的错误不是检索失败，而是日期题最终答案过度压缩。
- max0005 只用一次搜索也能给出完整日期，说明 `0.005` 确实缓解了部分 answer granularity 风险。

### `test_97`: Quit India speech speaker birth date

- Gold：`October 2, 1869`
- v2_20：3 次搜索，答 `October 2, 1869`。
- max001：1 次搜索，答 `1869`，同时标为 `answer_granularity_miss` 和 `missing_followup_query`。
- max0005：1 次搜索，答 `October 2, 1869`。

判断：

- max001 读到了正确实体但最终只输出年份，属于答案粒度损失。
- max0005 修复了完整日期输出；这里支持“0.005 比 0.01 更少诱导过短答案”的假设。
- 但这类修复只有 2 条，不足以抵消 max0005 相对 max001 丢掉的 4 条。

## Max-Search 共同丢失样本

这 6 条是 `v2_20` 正确、`max001` 和 `max0005` 都错误。

### `dev_4869`: Claudia Antonia maternal grandfather

- Gold：`Sextus Aelius Catus`
- v2_20：3 次搜索，答 `Sextus Aelius Catus`。
- max001：1 次搜索，答 `Aelius Sejanus`，标为 `missing_followup_query`。
- max0005：1 次搜索，答 `Aelius Sejanus`，标为 `missing_followup_query`。

判断：

- 这是最明确的过早停止退化。
- 正确路径需要先确认 Claudia Antonia 的母亲 Aelia Paetina，再追 Aelia Paetina 的父亲。
- 两个 max-search 版本都只搜泛化 query 后作答，实体角色绑定错误。

### `test_29`: Indiana Jones main actor licensed what

- Gold：`pilot`
- v2_20：2 次搜索，答 `pilot`。
- max001：1 次搜索，答 `actor`。
- max0005：1 次搜索，格式无效，无最终答案。

判断：

- 这是单跳题中的 query/evidence reading 退化。
- max001 搜了 `licensed profession`，但把 actor profession 和 licensed pilot 混淆。
- max0005 甚至没有给出合规最终答案。

### `dev_174`: Acting Company founder studied where

- Gold：`Clifton College`
- v2_20：4 次搜索，答 `Clifton College`。
- max001：1 次搜索，答 `Juilliard School`。
- max0005：1 次搜索，答 `Juilliard School`。

判断：

- 这是 role binding 问题。问题问 founder of the Acting Company 的学习经历，搜索结果中容易混入 Acting Company 与 Juilliard 的机构关系。
- v2_20 虽然搜索更多，但最终追到 John Houseman 的背景；两个 max-search 版本过早接受了干扰学校。

### `test_494`: Spain national team all-time top scorer

- Gold：`David Villa`
- v2_20：3 次搜索，答 `David Villa`。
- max001：4 次搜索后 `max_search_calls`，无最终答案。
- max0005：4 次搜索后 `tool_observation_budget`，无最终答案。

判断：

- 这不是“少搜导致错”，而是 max-search 版本 query drift。
- 两个 max-search 版本都把 `Lionel Messi` 等干扰项引入 Spain national team 语境，最后没有收束答案。
- 说明 max-search penalty 不能替代 query quality 控制。

### `test_7310`: My Last Day director

- Gold：`Barry Cook`
- v2_20：1 次搜索，答 `Barry Cook`。
- max001：1 次搜索，答 `Barry Cook (for the 2011 film) or Liu Bicheng (for the 2020 film)`。
- max0005：1 次搜索，答 `Barry Cook (for the 2011 film) or Liu Bicheng (for the 2020 film)`。

判断：

- 这是答案过宽导致 strict EM 判错。
- 两个 max-search 版本都包含 gold，但附加了第二个候选，说明需要最终答案唯一性约束。
- diagnostic 同时标为 alias 和 missing follow-up，但人工看更像“消歧不足后过宽回答”，不是典型少搜。

### `test_4020`: Siddhartha Gautama better known name

- Gold 包含 `The Buddha`、`Buddha` 等。
- v2_20：答 `The Buddha`，正确。
- max001：答 `The Buddha (or Shakyamuni Buddha)`，strict EM 判错。
- max0005：答 `The Buddha (or Shakyamuni Buddha)`，strict EM 判错。

判断：

- 这是答案过宽/多候选输出，不是知识错误。
- 对面试叙事应归为 strict EM false negative 或 final-answer formatting 问题。

## max0005 独有丢失样本

### `test_2231`: Astros changed league

- Gold：`2013`
- v2_20：1 次搜索，答 `2013`。
- max001：1 次搜索，答 `2013`。
- max0005：1 次搜索，答 `1962`。

判断：

- max0005 把 “franchise founded/joined MLB” 与 “moved from NL to AL” 混淆。
- 搜索次数相同，主要是 evidence reading 和 question decomposition 退化。

### `test_8542`: Slivovitz fruit

- Gold 包含 `Plum` / `Plums`。
- v2_20：答 `Plum`，正确。
- max001：答 `Plum`，正确。
- max0005：答 `Damson plums`，strict EM 判错，diagnostic 标为 alias。

判断：

- 语义上 `Damson plums` 是 plum 的具体品种，不能简单视作完全错误。
- 但对 strict EM 来说，它偏离了 gold 粒度。
- 这是 max0005 的答案粒度/别名边界问题。

## 风险归因

| 类型 | 样本 | 涉及 checkpoint | 判断 |
| --- | --- | --- | --- |
| 有效减少空转 | `test_108`, `dev_2223` | max001, max0005 | max-search 约束让模型更快收束，修复 v2_20 的 max-search 无答案。 |
| 必要 follow-up 被压掉 | `dev_4869`, `dev_3412`, `dev_6133` | max001/max0005，尤其 max0005 | 多跳实体角色题需要继续查中间实体；单纯惩罚 max-search 会提高过早作答风险。 |
| 日期答案粒度损失 | `test_97`, `test_99` | max001 | 0.01 版本会把完整日期压成年份；0.005 修复了这两条。 |
| 答案过宽/多候选 | `test_7310`, `test_4020` | max001, max0005 | 包含 gold 但输出多个候选，strict EM 判错；需要 final answer 唯一性约束。 |
| Strict EM alias/粒度 false negative | `dev_2407`, `test_8542`, `test_7511`, `test_42`, `test_4753` | 各 checkpoint | 不应直接当作真实能力下降，应在报告中单列 relaxed/alias bucket。 |
| Query drift / 干扰项 | `test_494`, `test_2231` | max001/max0005 | 搜索次数不是主因，query 改写和证据阅读才是瓶颈。 |

## 结论

1. `max001` 是当前更好的 max-search 折中：相对 `max0005` 多 2 条 correct、format 更高、平均搜索更少，尽管有 2 条日期粒度损失。
2. `max0005` 修复了 `test_97` 和 `test_99` 的日期粒度问题，但没有减少 `missing_followup_query`，还丢掉了 `dev_3412`、`test_2231`、`test_8542` 等样本。
3. `v2_20` 的 EM 最高，因为它保留了更多必要多跳搜索和证据阅读机会；但搜索效率和 format 明显退化。
4. 下一轮不应继续调大或调小单一 max-search penalty。更合理的方向是：
   - 保留 duplicate/empty 轻 penalty。
   - 对 `max_search_no_answer` 只惩罚“重复/同义 query 空转”，不要惩罚不同实体的必要 follow-up。
   - 增加 final answer 唯一性和日期粒度检查。
   - 增加 alias-aware/relaxed metric，避免把 `Dexter King`、`Damson plums`、`Yun Seok-ho` 全部混作真实错误。

当前决策：

- 不扩大 `max0005`。
- `max001` 可作为高 format/高效率候选保留，但必须先解决日期粒度与多候选答案问题。
- 下一组实验优先做 `penalty_v2_no_empty` 或 `group_size=8` 对照；reward 设计上优先实现 follow-up-aware max-search penalty，而不是继续扫 `max_search_no_answer_penalty` 标量。
