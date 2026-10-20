# `agent_company` 模块导航

`agent_company` 保存全部可复用业务逻辑，按职责分成七个子包；`runtime_io.py` 是共享的 artifact
布局与原子 JSON 读写。`scripts/` 只做参数解析后调用这里。目录与依赖规则见
[`docs/repository_structure.md`](../docs/repository_structure.md)。

这次重组只按职责移动文件，不改代码逻辑。原来同一个包里的耦合仍在：子包之间有双向引用
（例如 `programbench` ↔ `evolution`、`runtime` ↔ `policy`），要拆开需要另外改代码。

## `runtime/`：一个臂的执行

- `qwen.py`：用发布版 Qwen CLI 运行 Single 或原生 Team，生成 Qwen 配置与启动环境。
- `relay.py`, `transport.py`：容器唯一可达的模型通道；角色工具过滤、派 worker、并发上限、
  重试与期限、逐次请求记录和用量费用。
- `provider.py`：读取 provider 配置（凭据只从环境变量取），输出不含凭据的描述。
- `team.py`, `workspace.py`：原生 Team 的 worker 身份与共享 Docker 任务环境。
- `network.py`, `netns_firewall.py`, `network_helper.Dockerfile`：只放行 relay 的网络边界。
- `node_runtime.py`：挂载指定的 Node 发行版，不改 benchmark 镜像。
- `qwen_*.mjs`, `qwen_bootstrap.cjs`：对原生 Qwen 的兼容补丁（check-in、停滞超时、工具错误、
  最终汇报），由 `qwen.py` 按同目录路径写入运行时。

## `policy/`：角色策略（研究对象）

- `role_policy.py`：relay 持有的角色级策略状态，控制消息的暂存与提交。
- `role_policy_runtime.py`：注入给模型的策略渲染与运行时输入。
- `source_boundary.py`：固定的 benchmark 完整性说明，与演化的 A/C 分开。

## `programbench/`：任务、评测与判读协议

- `task.py`, `workspace.py`：公开任务输入、cleanroom 推理容器、源码提交。
- `runner.py`, `comparison.py`, `cli.py`：Single/Team 推理加独立官方评测。
- `evaluator_compat.py`, `evaluator_pytest_timeout.py`：对复制的官方评测器的适配
  （按同目录路径装入评测容器，不在 import 图中）。
- `total_selection.py`, `total_evidence.py`：单题官方总分协议。
- `followup.py`, `study.py`, `baseline.py`, `observation.py`：演化 followup、单题研究输入、
  基线 profile 与不注入的运行观察。

## `analysis/`：只读证据与读数

- `evidence_projection.py`：把 Team 证据投影成不含推理内容的视图。
- `score_layers.py`, `baseline_summary.py`, `native_final_report.py`, `s0_report.py`：
  分数分层、基线汇总、最终汇报检查与 S0 读数。

## `evolution/`：策略演化

- `loop.py`, `selection.py`, `recovery.py`：分工与通信策略的联合演化、选择与任务边界 checkpoint。
- `optimizer.py`, `reviewer.py`, `outer_submission.py`：optimizer 与 reviewer。
- `evidence.py`, `outer_budget.py`, `harness_capabilities.py`, `efficiency.py`：训练证据读取、分页、
  目标 harness 描述与离线效率投影。

## `deploy/`：正式部署与付费启动

- `deployment.py`：按提交归档的一次部署及其零 API 核验。
- `launch.py`, `run_health.py`：对已审部署执行单个显式动作及其闸门。
- `ssh.py`, `support.py`, `code_version.py`：无凭据 SSH、启动脚本共享逻辑、运行时代码版本记录。

## `roadmap/`：早期 RoadmapBench 路径（仍被复用）

- `task.py`：Harbor 任务格式，也是 ProgramBench `Task` 的基类。
- `runner.py`, `data.py`, `cli.py`, `policies/`：RoadmapBench 运行、下载与 `scripts/run_stage1.py` 入口。

## 历史代码

2026-10-15 删除了 V4/V5、mini-SWE、MiniCORAL、proposer 与早期协议层，并把原来的
`agent_company/stage1/` 按上面的职责拆开；需要时从提交 `e6a9790` 恢复旧布局。

## 新文件放置

1. 只是 CLI 参数编排？放 `scripts/`，业务逻辑留在本包。
2. 属于哪一层就放进对应子包；不保留 re-export shim，移动模块时直接改所有调用方。
3. 仅供测试使用？放 `tests/`。
