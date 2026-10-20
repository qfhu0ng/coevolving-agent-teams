# 角色策略系列：本地接口与边界

当前为阶段 1/4 的本地实现，未提交、未部署、未调用真实模型。阶段 2/3 的跨运行诊断、按目标前缀结转、单次 carry reviewer 会话、optimizer 补位与演化报告仍须完成。以下入口不会自行恢复旧实验或启动第二批。

## 包与运行请求

`agent_company/policy/role_package.py` 定义 `role_package_v1`。P0 从既有 A/C 原文迁移，保留换行；初始角色为 master、implementer、integrator、verifier、prober，角色文本为空。目录和 A 在运行中固定；propose 只改变 C 或已有角色策略，off 全部固定。角色版本由所有当前及后来同角色实例共享，通知不决定状态是否生效。

在 profile 的 `inference` 中显式设置 `role_policy_protocol: "role_package_v1"`，各臂提供 `policy_package_path`；每个 Team 臂独立提供 `role_policy_mode: "off"` 或 `"propose"`。缺少 opt-in 的旧 A/C 路径保持原行为。Single 绑定同一输入包以启用固定交付和收尾协议，但应用的 A/C 为空，不接收团队规则或协作摘要。

`scripts/prepare_programbench_role_series.py --project-root <执行代码目录> --write-profiles <外部新目录>` 只创建 P0 包及两批配置。输出必须在执行代码目录之外；部署时使用部署下的 `profiles/`。它不会伪造 v1。

| 批次 | 两臂（同时段，并发 2） | 配置模板 |
|---|---|---|
| 1 | Single(P0) + propose(P0) | `configs/stage1/role_series_pandoc_batch1.template.json` |
| 2 | propose(v1) + off(P0) | `configs/stage1/role_series_pandoc_batch2.template.json` |

formal 的新步骤是 `role_series`，显式传 `--batch-index 1/2` 和同号 `--repeat-index`。只读 prepare 与付费 launch 仍分开，沿用锁、两臂上限、内存、清理、部署身份和唯一输出目录检查；每次只处理一个批次。第二批缺少真实已审 v1 时不能形成有效运行计划。两批编号固定为 1 和 2，必须在新的部署目录运行，不复用已占用编号的旧部署。第二批 v1.parent 必须指向 off 臂及首批实际运行的同一 P0（revision 0）；启动记录按臂保存包的 SHA256。

## 运行中证据

- 原生 task_create/update/list 的成功结果或 taskstore 事实确认任务状态；工具参数本身不当作执行成功。
- COORD 仅 report/resolve/ack，通过既有 send_message 承载；后续真实 tool result 上追加持久化回执。reported 内容不升级为已验证事实。
- pending、提案状态、3000 字节协作摘要只附最新一份。事件与收尾提醒锚定在首次送达的原生历史位置、逐字重放；压缩移除锚点后记录且不补发。transport retry 使用同一请求快照。
- Git 采样复用已有采样周期，只读取对象/路径及关系元数据。缺日志或缺关系保留 unknown/hypothesis，不把提交号不同、缺少 ACK、文件删除单独判成协作错误。
- Stop hook 使用已批准原文，仅 Single/master；≤900 秒放行收尾，worker 不安装，原生连续空转保护保留。

## 生成与派发分离

`agent_company/programbench/role_generation.py` 的 `generate_only` 接受脚本生成组件，保存完整 parent、candidate、review、report 和状态，无 runner/启动器调用。

`scripts/generate_programbench_role_policy.py --parent <包> --replay-result <结构化结果> --output <新目录>` 是零 API 回放入口。结果明示 `capture_kind=artifact_replay`、`fresh_model_review=false`、`dispatch=not_performed`。它不会把保存的 review 当作新审查，也不会批准候选或产生官方分。实际诊断/carry/optimizer 后端留给阶段 2/3；不能拿旧 followup 自动派发入口代替。

## 验证边界

本地原生 CLI 使用脚本 provider 验证可达请求、工具执行、回执与状态继承；不证明真实模型遵从、策略收益、Docker/服务器行为。既有 formal `--smoke` 与缓存探针对新角色协议明确拒绝，不能把旧五模式冒烟当作新包验证。部署前须在新协议真实 CLI 场景验证剩余时间大于 900 秒时 Single/master 的提前结束被 Stop hook 阻挡并继续工具工作，不能只用 420 秒冒烟证明。部署、真实模型调用、启动与 push 均需用户单独授权。

阶段 2/3 待办：POLICY-REVIEW 标签和事件关联、广播目标、候选 revision 与父版本关系、parent/回放源 hash 和重复 JSON 键检查，在生成 v1 前完成。
