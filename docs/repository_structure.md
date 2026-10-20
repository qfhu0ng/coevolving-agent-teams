# 仓库文件架构

本文是仓库的文件导航与边界约定。设计说明见 [`design_zh.md`](design_zh.md)，最终结果见 [`results/`](../results/)。本文只回答三个问题：
文件应放在哪里、目录之间如何依赖、哪些内容可以进入 Git。

## 顶层目录

```text
Agent Company/
├── agent_company/       可复用的 Python 实现；业务逻辑只放这里（逐模块导航见 agent_company/README.md）
│   ├── runtime/         一个臂的执行：Qwen CLI 与原生 Team、relay、断网边界
│   ├── policy/          角色级策略（研究对象）
│   ├── programbench/    任务、cleanroom 推理、官方评测、总分判读协议
│   ├── analysis/        已保存证据的只读读数
│   ├── evolution/       optimizer、reviewer、选择
│   ├── deploy/          按提交部署与付费启动
│   ├── roadmap/         早期 RoadmapBench 路径（仍被复用）
│   └── runtime_io.py    artifact 布局与原子 JSON 读写
├── configs/stage1/      provider、策略与实验配置
├── scripts/             薄 CLI 入口；参数解析后调用 agent_company
├── tests/               自动化测试与测试场景/夹具
├── docs/                方案、协议、运行边界与研究笔记
├── runs/                本地实验输出（Git 忽略）
├── local_docs/          本地报告、图表、研究笔记、开发记忆与历史文档（Git 忽略）
├── external/            本地第三方工作树/运行依赖（Git 忽略）
├── README.md            项目入口与最终结果摘要
├── results/             最终结果：分数、M0/M1 原文、曲线与图
└── LICENSE              MIT
```

V4/V5、mini-SWE、MiniCORAL、proposer、trusted verifier 与 `experiments/` 已于 2026-10-15 从主干删除，
原 `agent_company/stage1/` 同日按职责拆成上面七个子包；需要时从提交 `e6a9790` 恢复旧布局。本地还可能出现 `.venv*/`、`.pytest_cache/`、`.ruff_cache/`、
`__pycache__/` 和 `.DS_Store`，都已由 `.gitignore` 排除。

## 2026-10-15 重组对照

原 `agent_company/stage1/` 下的文件按职责移入子包；去掉与包名重复的前缀，其余文件名不变。

| 新位置 | 原 `stage1/` 文件 |
| --- | --- |
| `runtime/` | `qwen`、`team`、`workspace`、`relay`、`transport`、`network`、`netns_firewall`、`node_runtime`、`qwen_*.mjs/cjs`、`network_helper.Dockerfile`；新增 `provider.py`（原 `cli.load_provider` 与 `cli._public_provider`，后者改名 `public_provider`） |
| `policy/` | `role_policy`、`role_policy_runtime`、`source_boundary` |
| `programbench/` | `programbench` → `task`；`programbench_{workspace,runner,comparison,cli,study,followup,observation,total_selection,total_evidence,baseline}` → 去掉前缀；`evaluator_compat`、`evaluator_pytest_timeout` |
| `analysis/` | `programbench_{evidence_projection,score_layers,baseline_summary}` → 去掉前缀；`native_final_report`、`s0_report` |
| `evolution/` | `evolution` → `loop`；`selection`、`optimizer`、`reviewer`、`evidence`、`outer_budget`、`outer_submission`、`harness_capabilities`、`efficiency`、`recovery` |
| `deploy/` | `formal_deployment` → `deployment`、`formal_launch` → `launch`、`formal_run_health` → `run_health`、`launch_support` → `support`、`launch_transport` → `ssh`；`code_version` |
| `roadmap/` | `roadmap` → `task`；`runner`、`data`、`cli`、`policies/` |

测试 `tests/test_stage1_X.py` 一律改名为 `tests/test_X.py`。

## 依赖方向

```text
scripts/ ──┐
tests/ ────┴──> agent_company/
docs/  --描述/链接-->  scripts/、agent_company/
agent_company/ --写输出--> runs/（运行时路径，不允许反向 import）
```

1. `scripts/` 只负责 CLI 参数、显式 live-run 确认和结果展示；可复用逻辑进入 `agent_company/`。
2. `agent_company/` 不 import `scripts/`、`tests/`、`runs/` 或 `docs/`（由 `tests/test_architecture_boundaries.py` 检查；
   目前已知一处违例：`deploy/launch.py` 读取 `scripts/launch_programbench_s0_deepseek.py` 的常量）。
3. `tests/` 可以调用公开模块和 CLI，测试夹具不应成为生产依赖。
4. 运行生成的 trajectory、patch、log 和 summary 进入 `runs/`。
5. 不保留 re-export shim；移动模块时直接改所有调用方的 import，由全量测试证明行为中性。

## Git 与本地数据边界

| 内容 | 是否提交 | 位置 |
| --- | --- | --- |
| 源码、测试、CLI、配置 | 是 | `agent_company/`, `tests/`, `scripts/`, `configs/` |
| 方案、协议、运行边界 | 是 | `docs/*.md` |
| 报告、图表、研究笔记、历史文档 | 否 | `local_docs/` |
| 原始轨迹、patch、日志、完整结果包 | 否 | `runs/` |
| 第三方本地工作树、环境、缓存、密钥 | 否 | `external/`, `.venv*`, cache, `.env*` |

不要因为某个结果“重要”就强制加入被忽略的 `runs/` 或 `local_docs/`。
