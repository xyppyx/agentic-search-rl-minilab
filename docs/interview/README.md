# interview

本目录用于记录训练、评测、推理、rollout 或搜索工具实验的面试复盘材料。

根目录保留面试临场速查材料：

- [interview_qa_quick_reference.md](interview_qa_quick_reference.md)：高频追问的 30 秒到 1 分钟口述答案。
- [one_page_project_pitch.md](one_page_project_pitch.md)：2 到 3 分钟项目讲稿、一页纸指标边界和收尾话术。

面试知识文档已迁移到 [../learning/](../learning/)：

- [../learning/project/project_experience_star.md](../learning/project/project_experience_star.md)：按 STAR(T) 法则组织的项目背景、任务、行动、指标和简历表述。
- [../learning/project/technical_implementation_details.md](../learning/project/technical_implementation_details.md)：项目技术实现、reward 设计、训练评测链路和关键指标。
- [../learning/basic/rl/reinforcement_learning_interview_notes.md](../learning/basic/rl/reinforcement_learning_interview_notes.md)：RL、GRPO、reward shaping 和训练稳定性基础。
- [../learning/basic/rl/opd_opsd_interview_notes.md](../learning/basic/rl/opd_opsd_interview_notes.md)：OPD、OPSD、自蒸馏、门控 mask 和本项目最终 OPSD v2 路线。
- [../learning/basic/agentic_rl/agentic_rl_search_r1_interview_notes.md](../learning/basic/agentic_rl/agentic_rl_search_r1_interview_notes.md)：Agentic RL、Search-R1、工具调用和评测口径。

历史实验复盘与可复用经验归档在 `lesson/` 子目录。

每份复盘应至少包含：

- 实验目标。
- 数据、模型、checkpoint 和 backend 版本。
- 关键参数、随机种子和命令。
- loss、reward、EM、format、工具调用和失败统计等实际观测指标。
- 耗时、费用、显存或远程资源使用情况。
- 问题、排查过程、解决方案和可复用经验。

未实际观测到的指标不得补写或猜测。
