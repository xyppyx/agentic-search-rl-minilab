"""使用 PyTRIO 和知乎搜索训练 Qwen3.5-4B Search-R1。

本文件是一条完整的 Search-R1 训练 pipeline。可以先按下面的流程图读，
再回到具体函数看实现：

1. 读取命令行参数：parse_args()
   - 决定训练步数、每步多少问题、每题采样几条轨迹、LoRA rank、学习率等。
2. 加载训练问题：main() -> shuffled_examples() / take_batch()
   - 数据已经由 prepare_data.py 整理成 JSONL，本文件只负责读取和打乱。
3. 创建或恢复 PyTRIO 训练客户端：main() -> trio.ServiceClient()
   - PyTRIO 远端持有基础模型、LoRA 权重、optimizer state。
4. 每个训练 step 先导出当前策略的采样客户端：
   main() -> training_client.save_weights_and_get_sampling_client()
   - rollout 必须来自“更新前”的当前策略，这些采样 logprob 后面会进入 loss。
5. 执行多轮工具交互采样：
   main() -> rollout_batch()
   - rollout.py 负责让模型生成 search tool call、调用知乎搜索、继续生成，
     最后按 reward.py 打分并计算同题组内 advantage。
6. 把每条完整轨迹转成 PyTRIO 可训练样本：
   main() -> build_training_datums() -> build_datum()
   - 只让 assistant 生成的 token 带 advantage；system/user/tool observation 的
     advantage 为 0，它们只作为上下文参与前向计算。
7. 把长短不一的轨迹拆成 micro-batch：
   main() -> pack_micro_batches() -> weight_micro_batch_for_global_mean()
   - 先在完整 group 上算 advantage，再拆小批；拆批时缩放 advantage，
     让多个 forward_backward 累计起来等价于一个 logical batch 的均值。
8. 远端反传并更新 LoRA：
   main() -> training_client.forward_backward(..., loss_fn="importance_sampling")
          -> training_client.optim_step()
   - forward_backward 累积梯度，optim_step 才真正更新参数。
9. 记录指标和保存 checkpoint：
   main() -> rollout_metrics() / merge_trainer_metrics() / swanlab.log()
          -> save_checkpoint()
   - state 用于续训，sampler weights 用于 eval.py 评测或后续采样。

在 03-search-r1 目录下运行正式训练：
uv run python train.py \
    --max-steps 100 \
    --questions-per-batch 8 \
    --group-size 8 \
    --save-every 50 \
    --swanlab-mode online \
    --run-name search-r1-qwen35-4b

小规模训练:
uv run python train.py \
    --max-steps 20 \
    --questions-per-batch 8 \
    --group-size 8 \
    --save-every 5 \
    --swanlab-mode online

小规模测试:
uv run python train.py \
    --max-steps 20 \
    --questions-per-batch 2 \
    --group-size 8 \
    --save-every 5 \
    --swanlab-mode disabled
"""

import argparse
from collections import defaultdict
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pytrio as trio
import swanlab
from tqdm import tqdm

from data import shuffled_examples, take_batch
from rollout import RolloutConfig, Trajectory, rollout_batch
from search import ZhihuSearchClient


MAX_TRAIN_CONTEXT_TOKENS = 8192  # 单个训练 Datum 允许的最大 token 数。
MAX_MICRO_BATCH_ITEMS = 32  # 单个 micro-batch 最多容纳的 Datum 数。
MAX_MICRO_BATCH_PADDED_TOKENS = 64_000  # 单批 padding 矩形允许的最大 token 数。


class TrainingDatum:
    """Pipeline 第 6-7 步：保存一条可训练轨迹及其真实长度。

    输入：一个 PyTRIO Datum，以及该 Datum 的未 padding token 数。
    输出：轻量容器对象，供 pack_micro_batches() 按长度装箱。
    作用：PyTRIO Datum 本身不暴露一个方便排序的长度字段，所以额外保存
    num_tokens，用来估算 micro-batch padding 后会占多少 token。
    """

    def __init__(self, datum: trio.Datum, num_tokens: int) -> None:
        """Pipeline 第 6 步：把训练样本和长度绑定在一起。

        输入：datum 是 PyTRIO 训练样本；num_tokens 是右移后的输入长度。
        输出：初始化后的 TrainingDatum。
        作用：后续 first-fit decreasing 装箱只需要看 num_tokens。
        """
        self.datum = datum
        self.num_tokens = num_tokens


def build_datum(trajectory: Trajectory) -> TrainingDatum:
    """Pipeline 第 6 步：把一条多轮搜索轨迹转成一个 PyTRIO Datum。

    输入：rollout.py 生成的一条 Trajectory，里面包含每轮 assistant 的
    prompt_tokens、completion_tokens、采样时 old logprobs，以及该轨迹的
    group-relative advantage。
    输出：TrainingDatum，内部的 trio.Datum 可直接交给 forward_backward()。
    作用：把“对话轨迹”变成“自回归语言模型训练样本”。关键规则是：
    assistant token 是模型动作，带 trajectory.advantage；system/user/tool
    observation 只是上下文，advantage 为 0，不对这些 token 做策略更新。
    """
    if not trajectory.turns:
        raise ValueError("不能用没有 assistant turn 的轨迹构造训练 Datum")

    # 三个列表始终一一对齐：第 i 个 token 对应第 i 个 old logprob 和 advantage。
    full_tokens: list[int] = []
    old_logprobs_by_token: list[float] = []
    advantages_by_token: list[float] = []
    assistant_token_count = 0

    for turn_index, turn in enumerate(trajectory.turns):
        if len(turn.completion_tokens) != len(turn.logprobs):
            raise ValueError(
                f"第 {turn_index + 1} 个 assistant turn 的 token 与 logprob 长度不一致"
            )

        # rollout.py 里每次 assistant 生成前都会保存“本轮完整 prompt”。
        # 第 0 轮之前没有历史，所以 turn.prompt_tokens 全部都是上下文。
        # 第 1 轮及以后，prompt 应该是 full_tokens 的前缀扩展：
        #   旧轨迹 full_tokens
        #   + 上一轮 tool observation / chat template 补充 token
        # 新增出来的那一截就是 delta_observation，它不是模型新生成的动作。
        if turn_index == 0:
            delta_observation = turn.prompt_tokens
        elif turn.prompt_tokens[: len(full_tokens)] == full_tokens:
            delta_observation = turn.prompt_tokens[len(full_tokens) :]
        else:
            raise ValueError(
                f"第 {turn_index + 1} 个 assistant turn 的 prompt "
                "不是已有轨迹的前缀扩展，无法安全对齐采样 logprob"
            )

        # 先追加环境/历史上下文，再追加本轮 assistant 生成内容。
        full_tokens.extend(delta_observation)
        full_tokens.extend(turn.completion_tokens)

        # old logprob 只对“采样出来的 assistant token”有意义。
        # observation 不是模型采样动作，填 0 只是为了长度对齐；它的 advantage
        # 也是 0，因此不会进入 policy loss。
        old_logprobs_by_token.extend([0.0] * len(delta_observation))
        old_logprobs_by_token.extend(turn.logprobs)

        # Search-R1 的 outcome reward 只在最终答案算一次，然后给整条轨迹的
        # assistant token 同一个 advantage。tool observation token 保留在上下文里，
        # 但 advantage 为 0，避免把搜索结果当成模型要学习生成的文本。
        advantages_by_token.extend([0.0] * len(delta_observation))
        advantages_by_token.extend(
            [trajectory.advantage] * len(turn.completion_tokens)
        )
        assistant_token_count += len(turn.completion_tokens)

    if assistant_token_count == 0:
        raise ValueError("不能用没有 assistant token 的轨迹构造训练 Datum")
    if not (
        len(full_tokens)
        == len(old_logprobs_by_token)
        == len(advantages_by_token)
    ):
        raise ValueError("完整轨迹的 token、logprob 和 advantage 长度不一致")

    # 自回归 LM 训练需要“右移一位”：
    #   input_tokens  = [t0, t1, ..., t(n-2)]
    #   target_tokens = [t1, t2, ..., t(n-1)]
    # 模型读到 input_tokens 的第 i 个位置时，要预测 target_tokens 的第 i 个 token。
    #
    # old_logprobs 和 advantages 也必须做同样右移，因为它们描述的是“目标 token”
    # 在 rollout 时的旧策略概率和训练权重。举例：target_tokens[i] 如果是 tool
    # observation，对应 advantage 就是 0；如果是 assistant 生成 token，对应
    # advantage 就是 trajectory.advantage。
    input_tokens = full_tokens[:-1]
    target_tokens = full_tokens[1:]
    old_logprobs = old_logprobs_by_token[1:]
    advantages = advantages_by_token[1:]
    if not (
        len(input_tokens)
        == len(target_tokens)
        == len(old_logprobs)
        == len(advantages)
    ):
        raise ValueError("Datum 的 input、target、logprobs 和 advantages 长度不一致")
    if len(input_tokens) > MAX_TRAIN_CONTEXT_TOKENS:
        raise ValueError(f"Datum 超过 {MAX_TRAIN_CONTEXT_TOKENS} token")
    datum = trio.Datum(
        # PyTRIO ModelInput.from_ints() 把 Python token id 列表包装成远端模型
        # 能接收的输入对象；这里不会在本地跑模型前向。
        model_input=trio.ModelInput.from_ints(input_tokens),
        loss_fn_inputs={
            # importance_sampling loss 需要三个逐 token 输入：
            # target_tokens：当前位置要预测的 token；
            # logprobs：rollout 时旧策略对 target token 的 logprob；
            # advantages：这个 target token 的策略梯度权重。
            "target_tokens": np.asarray(target_tokens, dtype=np.int64),
            "logprobs": np.asarray(old_logprobs, dtype=np.float32),
            "advantages": np.asarray(advantages, dtype=np.float32),
        },
    )
    return TrainingDatum(datum, len(input_tokens))


def build_training_datums(trajectories: list[Trajectory]) -> list[TrainingDatum]:
    """Pipeline 第 6 步：筛出有训练信号的轨迹并构造 Datum。

    输入：一个 rollout batch 的所有 Trajectory。
    输出：若干 TrainingDatum；advantage 为 0 的轨迹会被跳过。
    作用：group 内所有 reward 都一样时，advantage 全部为 0，这个问题没有
    相对优劣信号，送去训练只会浪费 token。
    """
    datums: list[TrainingDatum] = []
    for trajectory in trajectories:
        if trajectory.advantage == 0.0:
            continue
        if any(turn.completion_tokens for turn in trajectory.turns):
            datums.append(build_datum(trajectory))
    return datums


def datum_size(item: TrainingDatum) -> int:
    """Pipeline 第 7 步：返回装箱排序使用的 Datum token 数。

    输入：TrainingDatum。
    输出：未 padding 的 token 数。
    作用：作为 sorted(..., key=datum_size) 的 key，让长样本优先装箱。
    """
    return item.num_tokens


def datum_loss_token_count(item: TrainingDatum) -> int:
    """Pipeline 第 9 步：统计一条 Datum 中实际参与 policy loss 的 token 数。

    输入：TrainingDatum。
    输出：advantage 非零的 token 数。
    作用：用于日志指标。input_tokens 包含上下文和 observation，但真正产生
    RL 更新的只有 advantage 非零的 assistant token。
    """
    # PyTRIO tensor/array 包装对象转回 numpy，方便本地统计非零 advantage。
    advantages = item.datum.loss_fn_inputs["advantages"].to_numpy()
    return int(np.count_nonzero(advantages))


def pack_micro_batches(datums: list[TrainingDatum]) -> list[list[TrainingDatum]]:
    """Pipeline 第 7 步：把长短不一的 Datum 装成多个 micro-batch。

    输入：build_training_datums() 产出的 TrainingDatum 列表。
    输出：list[list[TrainingDatum]]，每个内部列表是一次 forward_backward 请求。
    作用：远端训练通常会把同一 batch padding 到相同长度。如果把 8k token
    和 500 token 的样本随便混在一起，padding 会浪费很多 token。本函数用
    first-fit decreasing：先放长样本，再把短样本塞进还能容纳它的已有批次。
    """
    batches: list[list[TrainingDatum]] = []
    batch_max_tokens: list[int] = []
    # 长样本优先，能减少“一个很长样本把一批短样本全部 padding 到很长”的浪费。
    for item in sorted(datums, key=datum_size, reverse=True):
        if item.num_tokens > MAX_TRAIN_CONTEXT_TOKENS:
            raise ValueError("单条 Datum 超过训练上下文限制")
        for index, batch in enumerate(batches):
            next_items = len(batch) + 1
            next_max_tokens = max(batch_max_tokens[index], item.num_tokens)
            # 估算这个 micro-batch 远端 padding 后的矩形面积：
            # 样本数 * 该批最长序列长度。
            next_padded_tokens = next_items * next_max_tokens
            fits_items = next_items <= MAX_MICRO_BATCH_ITEMS
            fits_tokens = next_padded_tokens <= MAX_MICRO_BATCH_PADDED_TOKENS
            if fits_items and fits_tokens:
                batch.append(item)
                batch_max_tokens[index] = next_max_tokens
                break
        else:
            batches.append([item])
            batch_max_tokens.append(item.num_tokens)
    return batches


def weight_micro_batch_for_global_mean(
    micro_batch: list[TrainingDatum],
    total_samples: int,
) -> list[trio.Datum]:
    """Pipeline 第 7-8 步：缩放 micro-batch 的 advantage 以保持全局均值。

    输入：一个 micro-batch，以及完整 rollout batch 的轨迹总数 total_samples。
    输出：新的 trio.Datum 列表，内容与原样本相同，但 advantages 乘了权重。
    作用：PyTRIO 的 forward_backward() 会对“本次请求内的样本”取 mean。
    如果一个 logical batch 被拆成 3 个 micro-batch，直接累积会让每个
    micro-batch 权重相同，而不是让每条轨迹权重相同。这里乘 n_k / N，
    让 sum_k [n_k / N * mean(loss_k)] 等价于 mean(loss_all)。
    """
    if not micro_batch:
        return []
    if total_samples <= 0:
        raise ValueError("全局样本数必须大于零")
    if len(micro_batch) > total_samples:
        raise ValueError("micro-batch 样本数不能超过全局样本数")

    # 远端对每次 forward_backward 内的样本取 mean。多个大小不同的
    # micro-batch 直接累积会让小批次权重过大，因此将第 k 批乘以 n_k / N：
    # sum_k [n_k / N * mean(loss_k)] = mean(loss_global)。
    micro_batch_weight = np.float32(len(micro_batch) / total_samples)
    weighted_datums: list[trio.Datum] = []
    for item in micro_batch:
        loss_inputs = item.datum.loss_fn_inputs
        weighted_datums.append(
            trio.Datum(
                model_input=item.datum.model_input,
                loss_fn_inputs={
                    "target_tokens": loss_inputs["target_tokens"].to_numpy(),
                    "logprobs": loss_inputs["logprobs"].to_numpy(),
                    "advantages": (
                        loss_inputs["advantages"].to_numpy() * micro_batch_weight
                    ),
                },
            )
        )
    return weighted_datums


def mean(values: list[float]) -> float:
    """Pipeline 第 9 步：计算日志指标用的均值。

    输入：float 列表。
    输出：平均值；空列表返回 0.0。
    作用：让日志汇总逻辑不用在每个指标处单独处理空列表。
    """
    return sum(values) / len(values) if values else 0.0


def source_reward(trajectories: list[Trajectory], source_name: str) -> float:
    """Pipeline 第 9 步：按数据来源统计平均 reward。

    输入：所有轨迹和来源名称，如 "nq" 或 "hotpotqa"。
    输出：该来源轨迹的平均 reward。
    作用：训练集混合了 NQ 和 HotpotQA，分来源看 reward 能发现某一类题
    是否单独变好或变坏。
    """
    rewards = [
        trajectory.reward
        for trajectory in trajectories
        if source_name in trajectory.example.data_source.lower()
    ]
    return mean(rewards)


def degenerate_group_count(trajectories: list[Trajectory]) -> int:
    """Pipeline 第 9 步：统计没有有效相对训练信号的问题数。

    输入：一个 rollout batch 的所有 Trajectory。
    输出：整组 advantage 都为 0 的问题数量。
    作用：GRPO/Search-R1 依赖同题多条轨迹的相对好坏。如果同一道题的
    group reward 完全一样，advantage 全为 0，这道题不会贡献梯度。
    """
    groups: dict[int, list[float]] = defaultdict(list)
    for trajectory in trajectories:
        groups[trajectory.question_index].append(trajectory.advantage)
    return sum(all(advantage == 0.0 for advantage in values) for values in groups.values())


def rollout_metrics(
    trajectories: list[Trajectory],
    datums: list[TrainingDatum],
    micro_batches: list[list[TrainingDatum]],
    question_count: int,
) -> dict[str, float]:
    """Pipeline 第 9 步：汇总本地 rollout 和装箱指标。

    输入：本 step 的轨迹、训练 Datum、micro-batch 列表、问题数量。
    输出：可直接传给 SwanLab 的指标字典。
    作用：让你能在网页上同时看到模型结果、搜索行为、有效训练 token、
    padding 浪费和 degenerate group 比例。
    """
    tool_attempts = sum(
        "<tool_call>" in turn.text
        for trajectory in trajectories
        for turn in trajectory.turns
    )
    valid_tool_calls = sum(trajectory.search_calls for trajectory in trajectories)
    trajectory_lengths = [
        len(trajectory.turns[-1].prompt_tokens)
        + len(trajectory.turns[-1].completion_tokens)
        for trajectory in trajectories
        if trajectory.turns
    ]
    micro_batch_padded_tokens = [
        max((item.num_tokens for item in batch), default=0) * len(batch)
        for batch in micro_batches
    ]
    input_tokens = sum(item.num_tokens for item in datums)
    loss_tokens = sum(datum_loss_token_count(item) for item in datums)
    padded_tokens = sum(micro_batch_padded_tokens)
    return {
        "reward/mean": mean([trajectory.reward for trajectory in trajectories]),
        "reward/correct": mean([float(trajectory.exact_match) for trajectory in trajectories]),
        "reward/format": mean([float(trajectory.valid_format) for trajectory in trajectories]),
        "reward/nq": source_reward(trajectories, "nq"),
        "reward/hotpotqa": source_reward(trajectories, "hotpotqa"),
        "rollout/turns": mean([float(len(trajectory.turns)) for trajectory in trajectories]),
        "rollout/search_calls": mean(
            [float(trajectory.search_calls) for trajectory in trajectories]
        ),
        "rollout/trajectory_tokens": mean([float(value) for value in trajectory_lengths]),
        "rollout/valid_tool_call_rate": valid_tool_calls / max(tool_attempts, 1),
        "rollout/degenerate_group_rate": degenerate_group_count(trajectories)
        / max(question_count, 1),
        "train/datums_per_rollout_batch": float(len(datums)),
        "train/micro_batches_per_step": float(len(micro_batches)),
        "train/tokens_per_rollout_batch": float(input_tokens),
        "train/loss_tokens_per_rollout_batch": float(loss_tokens),
        "train/padded_tokens_per_rollout_batch": float(padded_tokens),
        "train/max_micro_batch_padded_tokens": float(
            max(micro_batch_padded_tokens, default=0)
        ),
    }


def merge_trainer_metrics(results: list[Any]) -> dict[str, float]:
    """Pipeline 第 9 步：合并 PyTRIO 远端返回的 trainer 指标。

    输入：每次 forward_backward() 返回的结果列表。
    输出：统一加上 "trainer/" 前缀的指标字典。
    作用：一个 logical step 可能拆成多个 micro-batch，因此远端会返回多份
    loss/吞吐等指标。普通指标取平均；mean loss 因为前面已经按 n_k / N
    缩放过 advantage，所以这里要求和才对应整个 logical batch。
    """
    values: dict[str, list[float]] = defaultdict(list)
    for result in results:
        # PyTRIO result.metrics 是远端训练服务返回的指标集合；转成 dict 后
        # 只收集数值型字段，避免字符串/结构化字段进入 SwanLab 标量曲线。
        for key, value in dict(result.metrics).items():
            if isinstance(value, (int, float, np.number)):
                values[key].append(float(value))
    merged: dict[str, float] = {}
    for key, items in values.items():
        # 提交前的 advantage 已乘以 n_k / N，因此各 micro-batch 返回的
        # mean loss 相加才是整个 logical batch 的 global mean loss。
        if key in {"loss_mean", "loss/mean"}:
            merged[f"trainer/{key}"] = sum(items)
        else:
            merged[f"trainer/{key}"] = mean(items)
    return merged


def pick_mean_loss_metric(metrics: dict[str, float]) -> float | None:
    """Pipeline 第 9 步：从合并后的指标中找一个适合终端显示的 mean loss。

    输入：rollout_metrics() 和 merge_trainer_metrics() 合并后的指标字典。
    输出：mean loss；找不到则返回 None。
    作用：不同 PyTRIO 版本可能把 mean loss 命名为 loss_mean 或 loss/mean。
    这里只接受 mean loss，不混用 sum loss，避免终端数字随 batch 大小误导人。
    """
    for key in (
        "trainer/loss_mean",
        "trainer/loss/mean",
    ):
        if key in metrics:
            return float(metrics[key])
    return None


def serializable_config(args: argparse.Namespace) -> dict[str, Any]:
    """Pipeline 第 9 步：把命令行配置转成 SwanLab 可序列化格式。

    输入：argparse 解析出的 Namespace。
    输出：普通 dict，其中 Path 被转成字符串。
    作用：SwanLab 会记录 run config，方便之后知道某条曲线对应什么参数。
    """
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def save_checkpoint(training_client: Any, name: str) -> None:
    """Pipeline 第 9 步：保存训练状态和可采样权重。

    输入：PyTRIO training_client，以及 checkpoint 名称前缀。
    输出：无返回；在终端打印两个 trio:// 路径。
    作用：state 保存 optimizer/LoRA 等完整训练状态，用于 --resume-state；
    sampler weights 只用于采样/评测，传给 eval.py 的 --model-path。
    """
    # save_state() 在 PyTRIO 远端保存“可续训”的完整状态。
    state = training_client.save_state(name=f"{name}-state").result()
    # save_weights_for_sampler() 在 PyTRIO 远端保存“可推理/采样”的权重快照。
    weights = training_client.save_weights_for_sampler(name=f"{name}-weights").result()
    print(f"Saved state: {state.path}")
    print(f"Saved sampler weights: {weights.path}")


def parse_args() -> argparse.Namespace:
    """Pipeline 第 1 步：解析训练、rollout、日志和 checkpoint 参数。

    输入：命令行参数。
    输出：argparse.Namespace。
    作用：把实验中常改的变量都放到 CLI，避免为了试 group size、步数、
    采样温度或 SwanLab 模式而改代码。
    """
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-steps",
        type=int,
        required=True,
        help="最多执行多少个 GRPO 训练 step",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=base_dir / "datasets" / "train.jsonl",
        help="训练数据 JSONL 文件路径",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=0,
        help="最多使用多少条训练问题；0 表示使用全部数据",
    )
    parser.add_argument(
        "--base-model",
        default="Qwen/Qwen3.5-4B",
        help="创建 LoRA 训练客户端使用的基础模型",
    )
    parser.add_argument(
        "--resume-state",
        help="从 save_state() 保存的训练状态恢复；设置后不再新建 LoRA 客户端",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=32,
        help="新建 LoRA 训练客户端时使用的 rank",
    )
    parser.add_argument(
        "--questions-per-batch",
        type=int,
        default=8,
        help="每个训练 step 选取的问题数量",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=8,
        help="同一道问题采样的轨迹数量，用于计算组内相对 advantage",
    )
    parser.add_argument(
        "--max-search-calls",
        type=int,
        default=4,
        help="每条轨迹最多调用搜索工具的次数",
    )
    parser.add_argument(
        "--max-assistant-turns",
        type=int,
        default=6,
        help="每条轨迹最多生成的 assistant 回合数",
    )
    parser.add_argument(
        "--max-trajectory-tokens",
        type=int,
        default=8192,
        help="整条轨迹允许使用的最大 token 数",
    )
    parser.add_argument(
        "--max-assistant-tokens",
        type=int,
        default=1024,
        help="单个 assistant 回合最多生成的 token 数",
    )
    parser.add_argument(
        "--max-tool-response-tokens",
        type=int,
        default=1024,
        help="单次搜索结果最多保留的 token 数",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="rollout 采样温度；越高随机性越强",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="rollout 核采样的累积概率阈值",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=4e-5,
        help="Adam 优化器学习率",
    )
    parser.add_argument(
        "--beta1",
        type=float,
        default=0.9,
        help="Adam 优化器的一阶动量系数",
    )
    parser.add_argument(
        "--beta2",
        type=float,
        default=0.95,
        help="Adam 优化器的二阶动量系数",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=50,
        help=(
            "每隔多少个 step 同时保存断点续训 state 和推理 sampler weights；"
            "0 表示只在训练结束时保存"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="数据打乱、LoRA 初始化和 rollout 采样使用的随机种子",
    )
    parser.add_argument(
        "--run-name",
        default="search-r1-qwen35-4b",
        help="SwanLab 实验名称，同时作为 checkpoint 名称前缀",
    )
    parser.add_argument(
        "--swanlab-project",
        default="llm-agent-rl-lab-search-r1",
        help="SwanLab 项目名称",
    )
    parser.add_argument(
        "--swanlab-mode",
        choices=["online", "local", "offline", "disabled"],
        default="online",
        help="SwanLab 日志模式",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """Pipeline 第 2-9 步：运行完整 Search-R1 训练循环。

    输入：parse_args() 产出的 Namespace。
    输出：无直接返回；会向 PyTRIO 写入 LoRA 更新、向 SwanLab 写指标、
    并周期性保存 checkpoint。
    作用：把数据读取、远端 trainer/sampler、rollout、reward/advantage、
    Datum 构造、micro-batch 反传、optimizer step 和日志串成一个实验。
    """
    # 先固定打乱训练问题；max_train_samples 只用于限制本次实际参与训练的数据量。
    examples = shuffled_examples(args.data, args.seed)
    if args.max_train_samples > 0:
        examples = examples[: args.max_train_samples]
    if not examples:
        raise ValueError("训练数据为空，请先运行 prepare_data.py")

    # PyTRIO 的 ServiceClient 是本地 Python 连接远端训练/采样服务的入口。
    # 本文件不在本地加载 Qwen3.5-4B 权重，也不在本地做 backward。
    service_client = trio.ServiceClient()

    # 有 state 就恢复完整训练状态，否则从基础模型新建一个 LoRA 训练客户端。
    if args.resume_state:
        # create_training_client_from_state() 会恢复 LoRA、optimizer 等训练状态。
        training_client = service_client.create_training_client_from_state(
            args.resume_state
        )
    else:
        # create_lora_training_client() 在 PyTRIO 远端基于 base_model 新建 LoRA 训练任务。
        training_client = service_client.create_lora_training_client(
            base_model=args.base_model,
            rank=args.lora_rank,
            seed=args.seed,
        )

    # tokenizer 来自训练客户端，保证 rollout 采样、prompt 拼接、Datum 构造使用
    # 同一个 chat template 和 token id 空间。Search-R1 对 token 对齐很敏感。
    tokenizer = training_client.get_tokenizer()

    # 知乎搜索客户端是本地环境的一部分。模型生成 query 后，本地 Python 调 API，
    # 再把 observation 作为 tool message 接回下一轮 prompt。
    search_client = ZhihuSearchClient.from_env(Path(__file__).resolve().parent / ".env")

    # 将命令行中的采样、搜索次数和轨迹长度限制集中成 rollout 配置。
    rollout_config = RolloutConfig(
        group_size=args.group_size,
        max_search_calls=args.max_search_calls,
        max_assistant_turns=args.max_assistant_turns,
        max_trajectory_tokens=args.max_trajectory_tokens,
        max_assistant_tokens=args.max_assistant_tokens,
        max_tool_response_tokens=args.max_tool_response_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
    )

    # PyTRIO AdamParams 只是把优化器超参数传给远端；实际 optimizer state 在远端。
    # 每个训练 step 的所有 micro-batch 共享同一组 Adam 参数，并只 optim_step 一次。
    adam_params = trio.AdamParams(
        learning_rate=args.learning_rate,
        beta1=args.beta1,
        beta2=args.beta2,
    )

    # swanlab.init() 创建一次实验 run；后续 swanlab.log() 会把标量指标上传/保存。
    run = swanlab.init(
        project=args.swanlab_project,
        name=args.run_name,
        mode=args.swanlab_mode,
        config=serializable_config(args),
    )

    try:
        # 外层进度条统计已完成的训练 step，并自动显示总耗时和预计剩余时间。
        with tqdm(
            total=args.max_steps,
            desc="Training",
            unit="step",
            position=0,
        ) as training_progress:
            for step in range(args.max_steps):
                step_started = perf_counter()

                # 按 questions_per_batch 循环取题；超过数据末尾时 take_batch 会回绕。
                batch = take_batch(
                    examples,
                    step * args.questions_per_batch,
                    args.questions_per_batch,
                )

                # 导出当前 LoRA 权重创建 sampler，确保本 step 的 rollout 来自当前策略。
                # 这是 on-policy RL 的关键：old logprobs 必须来自生成这批轨迹时的策略。
                # save_weights_and_get_sampling_client() 会把 trainer 当前权重快照同步给
                # PyTRIO sampler，并返回一个只负责 sample_async 的 client。
                training_progress.set_postfix(phase="prepare sampler", refresh=True)
                sampling_client = training_client.save_weights_and_get_sampling_client()

                # 内层进度条显示当前 step 已完成多少条轨迹。
                training_progress.set_postfix(phase="rollout", refresh=True)
                with tqdm(
                    total=len(batch) * args.group_size,
                    desc=f"Step {step + 1}/{args.max_steps} rollout",
                    unit="trajectory",
                    position=1,
                    leave=False,
                ) as rollout_progress:
                    # 为每道题采样 group_size 条多轮搜索轨迹，并计算 reward 和组内 advantage。
                    # rollout_batch() 内部会调用 sampling_client.sample_async() 生成文本，
                    # 也会调用 search_client.search() 执行真实工具搜索。
                    trajectories = rollout_batch(
                        sampling_client,
                        tokenizer,
                        search_client,
                        batch,
                        rollout_config,
                        progress_callback=rollout_progress.update,
                    )

                # 每条有训练信号的完整轨迹只构造一个 Datum，再按 padding 矩形动态装箱。
                training_progress.set_postfix(phase="build datums", refresh=True)
                datums = build_training_datums(trajectories)
                micro_batches = pack_micro_batches(datums)

                # 远端对每次请求按样本取 mean；按当前批次占全部 rollout 样本的比例
                # 缩放 advantage 后再累积，保证动态拆批不改变 global mean 梯度。
                training_progress.set_postfix(phase="backward", refresh=True)
                trainer_results = []
                for micro_batch in micro_batches:
                    # forward_backward() 在 PyTRIO 远端跑模型前向、按指定 loss_fn
                    # 计算 loss，并累积梯度；这里还没有更新参数。
                    result = training_client.forward_backward(
                        weight_micro_batch_for_global_mean(
                            micro_batch,
                            total_samples=len(trajectories),
                        ),
                        # importance_sampling 使用 target_tokens、旧 logprobs 和
                        # advantages 计算策略梯度，是本项目 GRPO/Search-R1 的核心 loss。
                        loss_fn="importance_sampling",
                    ).result()
                    # .result() 等待远端任务完成，并取回 loss/吞吐等指标。
                    trainer_results.append(result)

                # 整个 rollout batch 只更新一次参数；没有非零 advantage 时跳过更新。
                if micro_batches:
                    training_progress.set_postfix(phase="optimizer", refresh=True)
                    # optim_step() 才真正把前面累积的梯度应用到 LoRA 权重上。
                    training_client.optim_step(adam_params).result()

                # 汇总 rollout、搜索客户端和远程 trainer 指标。
                metrics = rollout_metrics(
                    trajectories,
                    datums,
                    micro_batches,
                    len(batch),
                )
                metrics.update(search_client.stats.metrics())
                metrics.update(merge_trainer_metrics(trainer_results))
                mean_loss = pick_mean_loss_metric(metrics)

                # 按配置同时保存可续训 state 和可推理 sampler weights。
                # checkpoint 保存耗时也算在当前 step 的完整耗时内。
                if args.save_every > 0 and (step + 1) % args.save_every == 0:
                    training_progress.set_postfix(phase="checkpoint", refresh=True)
                    save_checkpoint(
                        training_client,
                        f"{args.run_name}-step-{step + 1}",
                    )

                # 记录完整 step 耗时，终端和 SwanLab 都能看到同一数值。
                step_seconds = perf_counter() - step_started
                metrics["time/step_seconds"] = step_seconds
                metrics["train/update_skipped"] = float(not micro_batches)
                # swanlab.log() 记录当前 step 的所有标量，网页上看到的曲线来自这里。
                swanlab.log(metrics, step=step)

                if not micro_batches:
                    loss_mean_text = "skipped"
                elif mean_loss is None:
                    loss_mean_text = "missing"
                else:
                    loss_mean_text = f"{mean_loss:.4f}"
                training_progress.update(1)
                training_progress.set_postfix(
                    step_s=f"{step_seconds:.1f}",
                    loss_mean=loss_mean_text,
                    reward=f"{metrics['reward/mean']:.3f}",
                    refresh=True,
                )
                tqdm.write(
                    f"step={step + 1}/{args.max_steps} "
                    f"step_time={step_seconds:.1f}s "
                    f"loss_mean={loss_mean_text} "
                    f"mean_reward={metrics['reward/mean']:.3f} "
                    f"correct_rate={metrics['reward/correct']:.3f} "
                    f"mean_search_calls={metrics['rollout/search_calls']:.2f} "
                    f"input_tokens={int(metrics['train/tokens_per_rollout_batch'])} "
                    f"loss_tokens={int(metrics['train/loss_tokens_per_rollout_batch'])} "
                    f"padded_tokens={int(metrics['train/padded_tokens_per_rollout_batch'])}"
                )

        # 无论周期保存频率如何，正常完成训练后始终保存一次最终 checkpoint。
        save_checkpoint(training_client, f"{args.run_name}-final")
    finally:
        # 即使训练中途抛出异常，也要结束 SwanLab run，避免实验一直显示为运行中。
        run.finish()


if __name__ == "__main__":
    main(parse_args())
