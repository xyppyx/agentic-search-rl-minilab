# Turn-Level Evidence Credit v2 50-Step Attempt

记录时间：2026-07-22 19:57 CST

## 目标

在 `turn_credit_evidence_bridge_v2` 20-step 已达到 dev 70 EM 0.4429、format 1.0000、平均搜索 1.7714 后，按用户要求直接扩大到 50-step，验证更长训练是否继续提升 EM，或是否重新出现 bad-loop、过搜、early-answer/follow-up 退化。

## 配置

- run name：`turn-credit-evidence-bridge-50step-20260722`
- backend：`zhihu_search`
- steps：50
- group size：4
- questions per batch：2
- policy：`evidence_bridge`
- evidence bonus：0.10
- early-answer penalty：0.05
- advantage normalization：standardize
- advantage clip：2.0
- KL coef：0.01
- policy ratio clip：0.2
- learning rate：1e-5
- save every：5
- SwanLab：disabled

## 结果

本次 50-step 未启动成功。训练在 step 0 的 `prepare sampler` 阶段失败，PyTRIO actor event submit 在重试后返回 retryable request failure，底层错误为 SSL EOF。该错误发生在 sampler weights 初始化前：

- 未进入 rollout。
- 未生成训练 trajectory。
- 未执行 reference logprobs、backward 或 optimizer。
- 未生成 checkpoint 或可评测 sampler weights。
- 未调用 Zhihu search backend，因此没有搜索 API 指标。

按项目运行规则，本次不继续反复重试远程训练服务，先记录为 PyTRIO actor/网络侧初始化阻塞。

## 决策

50-step 方向尚未被验证，也不能据此判断 50-step 好坏。当前可用最强 checkpoint 仍是 `turn-credit-evidence-bridge-20step-20260722`，dev 70 EM 0.4429、format 1.0000、平均搜索 1.7714、`missing_followup_query=0`。

下一次若继续 50-step，应先做一个最小 1-step 或 5-step PyTRIO health check，确认 actor event submit 和 sampler export 正常，再重启 50-step；不要把这次失败计入 reward 机制效果。
