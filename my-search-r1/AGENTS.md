# My Search-R1 目录规则

- 本目录是 Robust Search-R1 MiniLab 的正式改进实现区。
- 新增代码优先围绕工具封装、trajectory 可观测性、failure 注入、reward 组件化和 PyTRIO 后端薄封装展开。
- 代码应提供 mock/smoke run 路径，避免必须依赖真实搜索 API 或远程训练服务才能做基础验证。
- 新增训练、评测或 rollout 脚本必须支持通过命令行或配置文件覆盖关键参数，包括 seed、backend、输出目录、步数、样本数和 failure 概率。
- 不提交 `.env`、真实 API key、远程训练凭据、SwanLab 私有链接、数据下载缓存、trajectory 大文件、checkpoint、LoRA 权重或模型权重。
- 完成可验证功能或实验后，同步更新 `docs/status/PROJECT_COMPLETED.md` 与 `docs/status/PROJECT_TODO.md`。
