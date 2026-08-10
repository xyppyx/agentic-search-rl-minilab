# 迁移测试失败案例复盘

复盘时间：2026-07-20 20:49 CST

## 复盘范围

本次复盘不重新运行 eval/train，只读取迁移测试已经生成的报告和 trajectory JSONL：

- `my-search-r1/eval_results/zhihu_dev.md`
- `my-search-r1/eval_results/zhihu_dev.jsonl`
- `my-search-r1/eval_results/local_bm25_dev.md`
- `my-search-r1/eval_results/local_bm25_dev.jsonl`
- `my-search-r1/outputs/train_pytrio/search-r1-minilab-nondegenerate-smoke/step_000001.md`
- `my-search-r1/outputs/train_pytrio/search-r1-minilab-nondegenerate-smoke/step_000001.jsonl`

数据和 backend：

- dev eval：70 条，来自 `2wikimultihopqa`、`bamboogle`、`hotpotqa`、`musique`、`nq`、`popqa`、`triviaqa` 各 10 条。
- backend：`zhihu_search` 与 `local_bm25`。
- train smoke：8 条 `nq` trajectory，`zhihu_search`，1-step 非退化 GRPO smoke。
- 模型：报告沿用迁移测试默认 PyTRIO 配置，项目状态中记录为 `Qwen/Qwen3.5-4B`。

## 总体结论

| 产物 | 样本 | EM | 格式正确 | 平均搜索 | 工具失败 | 空结果 observation | 重复 query trajectory |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Zhihu dev | 70 | 15/70 | 47/70 | 1.83 | 0 | 7/128 | 2 |
| local BM25 dev | 70 | 1/70 | 11/70 | 3.61 | 0 | 144/253 | 8 |
| train smoke step 1 | 8 | 0/8 | 3/8 | 2.38 | 0 | 1/19 | 0 |

关键判断：

- `zhihu_search` 有真实搜索收益：15 个正确里 14 个发生过搜索。
- `local_bm25` 基本不能作为完整 dev 的可用知识源：唯一正确是模型没搜直接答对，搜索后正确为 0。
- 当前最大问题不是工具失败；三份产物工具失败均为 0。主要问题是格式收束、query 改写、弱相关/空结果处理和证据阅读。
- 不应对所有搜索调用做统一惩罚。Zhihu 的多跳正确样本里有 3-4 次搜索才答对的情况；更合理的是轻微惩罚重复 query、空结果后同类 query、达到 `max_search_calls` 仍无 `Answer:`。

## 案例分类

### 1. 直接答对

定义：`search_calls == 0` 且 `exact_match == true`。

观测：

- Zhihu dev：1 条。
- local BM25 dev：1 条。
- train smoke：0 条。

代表案例：

- 问题：`By what name is Siddhartha Gautama better known?`
- gold 包含：`The Buddha` / `Buddha`
- 模型输出：`Answer: The Buddha`
- 搜索次数：0

复盘：

- 这类问题可以由模型参数知识直接解决，不需要工具。
- 后续做 Search-R1 强化时，这类样本不宜强行奖励搜索；否则会把简单题训练成无意义工具调用。
- 可以考虑在评测报告里显式保留 `direct_correct`，作为“工具不必要样本”的估计。

### 2. 搜索后答对

定义：`search_calls > 0` 且 `exact_match == true`。

观测：

- Zhihu dev：14 条。
- local BM25 dev：0 条。
- train smoke：0 条。

代表案例：

| 问题 | 搜索次数 | query 路径 | 答案 |
| --- | ---: | --- | --- |
| `What nationality is Isabelle Coutant-Peyre's husband?` | 3 | `Isabelle Coutant-Peyre husband nationality` -> `Isabelle Coutant-Peyre husband name` -> `Carlos the Jackal nationality` | `Venezuelan` |
| `The husband of Lady Godiva was Earl of which Anglic kingdom?` | 1 | `husband of Lady Godiva title` | `Mercia` |
| `When was the person after which the Hubble Space Telescope is named after born?` | 1 | `Edwin Hubble birth date` | `November 20, 1889` |
| `When did the president who said Tear Down This Wall die?` | 4 | 先定位 quote 到 Ronald Reagan，再查死亡日期 | `June 5, 2004` |

复盘：

- Zhihu backend 能支撑 Search-R1 目标行为：模型通过逐步拆解实体和属性拿到最终答案。
- `Isabelle Coutant-Peyre` 与 `Tear Down This Wall` 说明多跳搜索有必要，不能只用搜索次数作为负向信号。
- 这类样本应作为后续 reward/penalty 改造的保护集：惩罚策略不能显著压低搜索后答对样本。

### 3. 搜索后答错

定义：`search_calls > 0`、`valid_format == true`、`exact_match == false`。

观测：

- Zhihu dev：32 条。其中粗略按 gold 是否出现在 observation 分：13 条 evidence 可能已出现、19 条 evidence 未出现或无法按字符串命中。
- local BM25 dev：10 条，均未在 observation 中命中 gold。
- train smoke：2 条，都是 `do veins carry blood to the heart or away?`，证据和输出语义接近 gold，但 EM 失败。

根因拆分：

1. 证据已足够，模型读证据或推理错。
   - 问题：`Who is younger, Jule Mallonee or Mimí Lazo?`
   - Zhihu query：`Jule Mallonee birth date`、`Mimí Lazo birth date`
   - observation 给出 Jule Mallonee 1900 年出生、Mimí Lazo 1954 年出生。
   - 模型输出：`Answer: Jule Mallonee`
   - gold：`Mimí Lazo`
   - 判断：query 和结果都可用，模型把“younger”比较方向做反了。

2. query 过宽或缺少关键实体，结果弱相关。
   - 问题：`The main actor of Indiana Jones is a licensed what?`
   - Zhihu query：`Indiana Jones main actor licensed profession`
   - top titles 包括 George Lucas、Harrison Ford still acting、电影看点，未稳定指向 `Harrison Ford pilot license`。
   - 模型输出：`Answer: actor`
   - gold：`pilot`
   - 判断：需要先确定 main actor = Harrison Ford，再搜 `Harrison Ford pilot license`；当前 query 混合了角色定位和属性，结果被“actor”表面词带偏。

3. 多跳关系读错，把上一跳实体的母亲当成目标。
   - 问题：`Who is the mother of the father of Barack Obama?`
   - Zhihu query 包含 `Barack Obama father mother`、`Barack Obama father name`。
   - 模型输出：`Answer: S. Ann Dunham`
   - gold：`Habiba Akumu Nyanjango`
   - 判断：模型识别到 Barack Obama Sr.，但把 Barack Obama 的母亲当成 Barack Obama Sr. 的母亲，属于关系链跟踪失败。

4. 严格 EM 对冗长或等价答案不友好。
   - 问题：`Where was the place of death of the director of film The Ages Of Lulu?`
   - 模型输出：`Answer: La Riera de Gaià, Catalonia, Spain`
   - gold：`La Riera de Gaia` / `La Riera de Gaià`
   - 判断：语义上基本正确，但输出包含上级行政区，严格 EM 判错。
   - 类似样本：`Prince John Konstantinovich Of Russia's mother die?` 输出 `March 24, 1927`，gold 为 `24 March 1927`，还带了人物全称前缀。

5. train smoke 中答案语义正确但格式/EM 不匹配。
   - 问题：`do veins carry blood to the heart or away?`
   - gold：`to`
   - 模型输出多为 `Answer: Veins carry blood to the heart.`
   - 判断：任务 gold 期望极短答案，模型输出完整句导致 EM 失败。后续需要在 prompt 或 reward 中强化“Answer 后只写最短规范答案”。

### 4. 搜索为空 / 结果弱相关

观测：

- Zhihu dev：7/128 个 tool observation 为空，空结果率 5.47%。
- local BM25 dev：144/253 个 tool observation 为空，空结果率 56.92%。
- train smoke：1/19 个 tool observation 为空，空结果率 5.26%。

backend 问题：

- local BM25 的 dev 覆盖明显不足。很多合理 query 直接空：
  - `Jule Mallonee birth date`
  - `Mimí Lazo birth date`
  - `Isabelle Coutant-Peyre husband nationality`
  - `Harrison Ford pilot license`
- local BM25 还会返回明显弱相关内容：
  - 问题：`Who was the last emperor of the dynasty that succeeded the Song dynasty?`
  - query：`dynasty that succeeded the Song dynasty`
  - local BM25 top titles：`The Little Prince`、`Paris`、`Antoine de Saint-Exupery`
  - 判断：这是 toy/local corpus 与完整 dev 不匹配，不应当解释为模型不会搜。

query 问题：

- Zhihu 上空结果少，但有些 query 被错误假设带偏。
  - 问题：`Who did Fredric Rieders tesify agains ... 60 patients ... Florence Colorado?`
  - gold：`Michael Swango`
  - 首次 query 过长，top results 已弱相关；后续 query 引入 `Dr. Hooten`，连续空结果。
  - 判断：模型从弱结果中产生了错误中间实体，后续 query 进入死路。
- 长尾实体也存在 backend/query 混合问题。
  - 问题：`How many students attend the Swiss University Philip Kraft has lectured at?`
  - 多个 `Philip Kraft ...` query 为空或命中无关 `Philip T. KREIN`。
  - 判断：需要更好的 query 退火策略，例如先搜 exact name，再加上下文，不要一开始塞入完整多跳问题。

### 5. 重复 query

观测：

- Zhihu dev：2 条 trajectory 出现重复 query。
- local BM25 dev：8 条 trajectory 出现重复 query。
- train smoke：0 条。

代表案例：

- local BM25：`What nationality is Isabelle Coutant-Peyre's husband?`
  - query 序列：`Isabelle Coutant-Peyre husband nationality`、`Isabelle Coutant-Peyre husband name`、`Isabelle Coutant-Peyre`、`Isabelle Coutant-Peyre`、`Isabelle Coutant-Peyre`
  - 前 4 个 observation 均为空，模型继续重复同一实体 query。
- Zhihu：`when did the astros change from the national league to the american league?`
  - query 序列最后重复 `Houston Astros moved from National League to American League 1962`
  - 问题 gold 是 `2013`，模型把错误候选 `1962` 写入 query，之后重复验证错误假设。

复盘：

- 重复 query 主要发生在没有明确“无收益停止”机制时。
- 后续可以在 rollout 层记录 `duplicate_query_count`，并对完全相同 query 或只改停用词的近重复 query 加轻微 penalty。
- penalty 应只在最终未答对或没有新 observation 信息时启用，避免误伤正常多跳。

### 6. 格式错误

观测：

- Zhihu dev：23/70。
- local BM25 dev：59/70。
- train smoke：5/8。

典型形态：

1. 单个 assistant turn 输出多个 tool call，被解析为 `invalid_format`，`search_calls == 0`。
   - 三个 dev 样本在两个 backend 都出现：
     - `Which film has the director who died earlier, Max And Helen or Held Einer Nacht?`
     - `Which film whose director is younger, Era D'Estate or Mr. Baseball?`
     - `Which film has the director born later, Jungle Ka Jawahar or Frankenstein'S Daughter?`
   - 模型一次输出两个 `<tool_call>`，没有遵守单步工具协议。

2. 达到 `max_search_calls` 后仍输出 tool call，没有唯一 `Answer:`。
   - local BM25 dev：56 条 stop_reason 为 `max_search_calls`。
   - Zhihu dev：12 条 stop_reason 为 `max_search_calls`。
   - train smoke：4 条 stop_reason 为 `max_search_calls`。

3. 内容正确但没有 `Answer:`。
   - train smoke 的 `do veins carry blood to the heart or away?` 有一条输出 `veins carry blood to the heart...`，语义接近正确，但无唯一 `Answer:`，reward 为 `-0.1`。

复盘：

- 格式问题和搜索质量问题交织：local BM25 空结果多，模型更容易一直搜到上限；但 Zhihu 也有纯协议问题。
- 现有 format reward 已经能惩罚，但还需要在采样 prompt 或 rollout stop policy 上更硬地约束：接近搜索上限时必须输出最终 `Answer:`，且一次只允许一个 tool call。

### 7. 工具调用过多但没收益

统计口径：`search_calls >= 3` 且 `exact_match == false`。

观测：

- Zhihu dev：15/70。
- local BM25 dev：63/70。
- train smoke：4/8。

代表案例：

- `roller derby first appear in the press?`
  - train smoke 4 条同组样本全部搜到上限。
  - query 多次围绕 1935、first game、newspaper，gold 是 `1922`。
  - 搜索结果不断强化 1935 起源叙事，模型没有找到题目要的 first press appearance。
- `Who was the last emperor of the dynasty that succeeded the Song dynasty?`
  - Zhihu 能检索到 Yuan 相关结果，但模型继续搜索 `Zhu Di` 这类错误中间假设，最后没有 answer。
  - local BM25 则完全是 corpus 不匹配和重复 query。

改造建议：

- 需要轻微 penalty，但不建议用简单线性 `search_call_penalty`。
- 建议分解为：
  - `duplicate_query_penalty`：完全重复 query，最终未答对时扣分。
  - `empty_query_penalty`：空结果后继续同一实体或同一关系 query，最终未答对时扣分。
  - `max_search_no_answer_penalty`：达到搜索上限仍无唯一 `Answer:` 时扣分。
  - `concise_answer_bonus` 或 answer-length 约束：`Answer:` 后只输出最短答案，降低严格 EM 误伤。
- 对 3-4 次搜索后答对的样本保留正向奖励，避免模型学成“少搜但乱猜”。

## 后续优先级

1. 先补报告指标：`direct_correct`、`searched_correct`、`searched_wrong`、`empty_observation_count`、`duplicate_query_count`、`max_search_no_answer_count`。
2. 再做 reward shaping：重复 query、空结果无改写、达到上限无答案的轻微 penalty，保持格式错误负分。
3. 加强 prompt/协议：一次只发一个 tool call；达到最后一次搜索预算后必须输出唯一 `Answer:`；`Answer:` 后只写短答案。
4. local BM25 只作为 smoke/mock backend，不用于说明完整 dev 搜索能力；完整 dev 对照应优先看 Zhihu 或后续更完整的离线检索库。
5. 对严格 EM 误伤样本增加复盘标签，不直接改高分，但在面试讲解中说明“评测指标保守，部分 wrong 是归一化或过长答案问题”。
