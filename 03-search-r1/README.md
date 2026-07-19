# 03-search-r1

本目录保存上游 Search-R1 教学复现代码，使用 Qwen3.5-4B、PyTRIO 和知乎搜索 API 跑通多轮搜索 RL。

详细文章见 [readme.md](readme.md)。本目录是当前改进项目的 baseline；正式改进实现默认放在 [../my-search-r1/](../my-search-r1/)。

主要文件：

- `prepare_data.py`：准备 NQ/HotpotQA 等训练与评测 JSONL。
- `data.py`：读取和打乱样本。
- `search.py`：知乎搜索客户端。
- `rollout.py`：多轮搜索轨迹采样。
- `reward.py`：格式与 EM reward。
- `train.py`：PyTRIO 训练循环。
- `eval.py`：checkpoint 评测。
- `analyse.py`：评测结果分析。
